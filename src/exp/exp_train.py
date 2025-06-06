import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.mvcnn_clip import MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP
from models.Teachers.CLIP import MV_CLIP
from models.Students.MVAlexNet import MV_AlexNet
from dataset.dataset import MultiViewDataset
from torchvision import transforms
from utils import TripletLoss, RelationDisLoss, read_tensorboard_data,plot_tensorboard_data
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from variable import TrainConfig

# 训练配置
train_config = TrainConfig(
    model_name='MVCNN_CLIP',  # 模型名称
    batch_size=8,  # 批大小
    lr=1e-6,  # 学习率
    model_num=0,  # 模型编号
    margin=1.0,  # 边距
    train_num_views=12,  # 视图数量
    num_epochs=14,  # 训练轮数
    train_dataset='OS-MN40-core',  # 训练数据集名称
    experiment_name="exp1"  # 实验名称
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN的自动优化（避免不确定性）

set_seed(2025)  # 可以改成你想要的种子值（如 2023, 1234 等）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 直接调整到 CLIP 的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # CLIP 专用参数
])

print("loading train data.......")
train_dataset = MultiViewDataset(root_dir=f"../../data/{train_config.train_data}/train",transform=transform, num_views=train_config.train_num_views)

train_loader = DataLoader(train_dataset, batch_size = train_config.batch_size, shuffle=True)

model_T = MV_CLIP(num_views = train_config.num_views).to(device)

print("finished loading train data.......")
if train_config.model_name == "MVCNN_CLIP":
    model = MVCNN_CLIP(num_views = train_config.train_num_views).to(device)
    criterion = TripletLoss(margin = 0.5)
    optimizer = optim.Adam([
    {"params": model.net_1.parameters(), "lr": 1e-6},  # 主干网络低学习率
    # {"params": model.net_2.parameters(), "lr": 1e-4}   # 新增层高学习率
])
elif train_config.model_name == "MV_AlexNet":
    print("Training MV_AlexNet")
    model = MV_AlexNet(num_views = train_config.train_num_views, is_dis = False).to(device)
    criterion = TripletLoss(margin = 0.5)
    optimizer = optim.Adam([
    {"params": model.features.features.parameters(), "lr": 1e-6},
    # {"params": model.features.fc_features.parameters(), "lr": train_config.lr},
    {"params": model.retrieval.parameters(), "lr": train_config.lr}   # 新增层高学习率
])
elif train_config.model_name == "MV_AlexNet_dis_Pre":
    print("Training MV_AlexNet_dis_Pre")
    model = MV_AlexNet(num_views = train_config.train_num_views, is_dis = True).to(device)
    criterion = RelationDisLoss(dis_margin = 1.0, ang_margin = 0.2)
    optimizer = optim.Adam([
    {"params": model.features.features.parameters(), "lr": 1e-6},
    {"params": model.retrieval.parameters(), "lr": train_config.lr}   # 新增层高学习率
])

# 添加学习率调度器
scheduler = StepLR(optimizer, step_size=4, gamma=0.5)  # 每 5 个 epoch 学习率乘以 0.1

model_path = f"../../models/train_models/{train_config.experiment_name}/{train_config.train_data}/{train_config.model_name}/epochs_{train_config.model_num}_lr_{train_config.lr}_batch_{train_config.batch_size}.pth"

# 确保保存模型的目录存在
os.makedirs(os.path.dirname(model_path), exist_ok=True)

num_epochs = train_config.num_epochs

# 检查是否存在已保存的模型文件
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step']
    print(f"Resuming training from epoch {start_epoch + 1}")
else:
    print("No saved model found. Starting training from scratch.")
    start_epoch = 0
    start_step = 0

# tensorboard
log_dir = f"../../output/tensorboard_logs/{train_config.experiment_name}/{train_config.train_data}/{train_config.model_name}/lr_{train_config.lr}_batch_{train_config.batch_size}"
os.makedirs(log_dir, exist_ok=True)

# 检查日志文件是否存在
if os.path.exists(log_dir) and os.listdir(log_dir):
    writer = SummaryWriter(log_dir=log_dir, purge_step=start_step)  # previous_step 是上次记录的步数
else:
    writer = SummaryWriter(log_dir=log_dir, purge_step=start_step)

losses = []
step = start_step
model.train()
print("before train")
for epoch in range(start_epoch + 1, num_epochs):
    # print("in the loop")
    epoch_loss = 0.0
    
    for anchor_images, positive_images, negative_images in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"):
        step += 1
        anchor_images = anchor_images.view(-1, *anchor_images.shape[-3:])  # (batch_size * num_views, C, H, W)
        positive_images = positive_images.view(-1, *positive_images.shape[-3:])  # (batch_size * num_views, C, H, W)
        negative_images = negative_images.view(-1, *negative_images.shape[-3:])  # (batch_size * num_views, C, H, W)

        anchor_images, positive_images, negative_images = anchor_images.to(device), positive_images.to(device), negative_images.to(device)

        optimizer.zero_grad()
        anchor_features = model(anchor_images)
        positive_features = model(positive_images)
        negative_features = model(negative_images)

        t_anchor_features = model_T(anchor_images)
        t_positive_features = model_T(positive_images)
        t_negative_features = model_T(negative_images)

        # print("anchor_features", anchor_features.shape) # anchor_features torch.Size([32, 768])

        if train_config.model_name == "MV_AlexNet_dis" or train_config.model_name == "MV_AlexNet_dis_Pre":
            loss = criterion(t_anchor_features, t_positive_features, t_negative_features,anchor_features, positive_features, negative_features)
        else:  
            loss = criterion(anchor_features, positive_features, negative_features)
        loss.backward()

        # 打印梯度
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient of {name}: {param.grad.norm().item():.4f}")


        # print(loss)
        optimizer.step()
        epoch_loss += loss.item()

        # 将损失值写入 TensorBoard
        writer.add_scalar("Loss/train", loss.item(), step)

        losses.append(loss.item())
        torch.cuda.empty_cache()

    epoch_loss /= len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 更新学习率
    scheduler.step()

    # 保存模型
    torch.save({
        'epoch': epoch,
        'step' : step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() 
    }, f"../../models/train_models/{train_config.experiment_name}/{train_config.train_data}/{train_config.model_name}/epochs_{epoch}_lr_{train_config.lr}_batch_{train_config.batch_size}.pth")

    # 将每个 epoch 的平均损失写入 TensorBoard
    writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)

# 关闭 SummaryWriter
writer.close()

# 定义日志路径和标签
log_dir = f"../../output/tensorboard_logs/{train_config.experiment_name}/{train_config.train_data}/{train_config.model_name}/lr_{train_config.lr}_batch_{train_config.batch_size}"
tag = "Loss/train_epoch"

# 读取 TensorBoard 数据
try:
    data = read_tensorboard_data(log_dir, tag)
    # 绘制并保存图表
    os.makedirs(f"../../output/Loss_curve/{train_config.experiment_name}/{train_config.train_data}/", exist_ok=True)
    plot_tensorboard_data(
        data,
        title="Training Loss Over Epochs",
        xlabel="Epoch",
        ylabel="Loss",
        save_path=f"../../output/Loss_curve/{train_config.experiment_name}/{train_config.train_data}/{train_config.model_name}_loss_curve_tensorboard_lr_{train_config.lr}_batch_{train_config.batch_size}.png"
    )
    print(f"Loss curve saved to ../../output/Loss_curve/{train_config.experiment_name}/{train_config.train_data}/{train_config.model_name}_loss_curve_tensorboard_lr_{train_config.lr}_batch_{train_config.batch_size}.png")
except ValueError as e:
    print(e)
