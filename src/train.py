import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.mvcnn_clip import MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP
from models.Students.MVAlexNet import MV_AlexNet
from dataset.dataset import MultiViewDataset
from torchvision import transforms
from utils import TripletLoss,read_tensorboard_data,plot_tensorboard_data
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# 定义命令行参数解析器
parser = argparse.ArgumentParser(description="Train MVCNN_CLIP model")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 10)")
parser.add_argument("--num_epochs", type=int, default=15, required=True, help="Number of epochs to train (default: 1)")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (default: 0.001)")
parser.add_argument("--margin", type=float, default=1.0, help="Margin for triplet loss (default: 1.0)")
parser.add_argument("--num_views", type=int, default=15, help="Number of views for MVCNN (default: 10)")
parser.add_argument("--data_root", type=str, default="../data/ModelNet_random_30_final/DS/train", help="Root directory of the dataset (default: ../data/ModelNet_random_30_final/DS/train)")
parser.add_argument("--model_num", type=str, default="9", required=True,help="Path to save the trained model (default: ../models/train_models/base/mvcnn_clip_01.pth)")
parser.add_argument("--model_name", type=str, default="MVCLIP_MLP", required=True, help="The name of the model")
args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 直接调整到 CLIP 的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # CLIP 专用参数
])

print("loading train data.......")
train_dataset = MultiViewDataset(root_dir="../data/ModelNet_random_30_final/DS/train",transform=transform, num_views=args.num_views)
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)

print("finished loading train data.......")
if args.model_name == "MVCNN_CLIP":
    model = MVCNN_CLIP(num_views = args.num_views).to(device)
    optimizer = optim.Adam([
    {"params": model.net_1.parameters(), "lr": 1e-6},  # 主干网络低学习率
    # {"params": model.net_2.parameters(), "lr": 1e-4}   # 新增层高学习率
])
elif args.model_name == "MVCLIP_CNN":
    model = MVCLIP_CNN(num_views = args.num_views).to(device)
    optimizer = optim.Adam([
    {"params": model.net_1.parameters(), "lr": 1e-6},  # 主干网络低学习率
    {"params": model.net_2.parameters(), "lr": 1e-4}   # 新增层高学习率
])
elif args.model_name == "MVCLIP_MLP":
    model = MVCLIP_MLP(num_views = args.num_views).to(device)
    optimizer = optim.Adam([
    {"params": model.net_1.parameters(), "lr": 1e-6},  # 主干网络低学习率
    {"params": model.net_2.parameters(), "lr": 1e-4}   # 新增层高学习率
])
elif args.model_name == "MV_AlexNet":
    model = MV_AlexNet(num_views = args.num_views).to(device)
    optimizer = optim.Adam([
    {"params": model.features.features.parameters(), "lr": 1e-6},
    {"params": model.features.fc_features.parameters(), "lr": 1e-4},
    {"params": model.retrieval.parameters(), "lr": 1e-4}   # 新增层高学习率
])

model_path = f"../models/train_models/base/{args.model_name}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pth"
criterion = TripletLoss(margin = 0.5)

# 确保保存模型的目录存在
os.makedirs(os.path.dirname(model_path), exist_ok=True)

num_epochs = args.num_epochs

# 检查是否存在已保存的模型文件
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step']
    print(f"Resuming training from epoch {start_epoch + 1}")
else:
    print("No saved model found. Starting training from scratch.")
    start_epoch = 0
    start_step = 0

# tensorboard
log_dir = f"../output/tensorboard_logs/{args.model_name}/lr_{args.lr}_batch_{args.batch_size}"
os.makedirs(log_dir, exist_ok=True)

# 检查日志文件是否存在
if os.path.exists(log_dir) and os.listdir(log_dir):
    writer = SummaryWriter(log_dir=log_dir, purge_step=start_step)  # previous_step 是上次记录的步数
else:
    writer = SummaryWriter(log_dir=log_dir)

losses = []
step = start_step
model.train()
for epoch in range(start_epoch + 1, num_epochs):
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

        loss = criterion(anchor_features, positive_features, negative_features)
        loss.backward()

        # 打印梯度
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient of {name}: {param.grad.norm().item():.4f}")

        optimizer.step()
        epoch_loss += loss.item()

        # 将损失值写入 TensorBoard
        writer.add_scalar("Loss/train", loss.item(), step)

        losses.append(loss.item())

    epoch_loss /= len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 保存模型
    torch.save({
        'epoch': epoch,
        'step' : step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"../models/train_models/base/{args.model_name}/epochs_{epoch}_lr_{args.lr}_batch_{args.batch_size}.pth")

    # 将每个 epoch 的平均损失写入 TensorBoard
    writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)

# 关闭 SummaryWriter
writer.close()

# 定义日志路径和标签
log_dir = f"../output/tensorboard_logs/{args.model_name}/lr_{args.lr}_batch_{args.batch_size}"
tag = "Loss/train_epoch"

# 读取 TensorBoard 数据
try:
    data = read_tensorboard_data(log_dir, tag)
    # 绘制并保存图表
    os.makedirs(f"../output/Loss_curve/", exist_ok=True)
    plot_tensorboard_data(
        data,
        title="Training Loss Over Epochs",
        xlabel="Epoch",
        ylabel="Loss",
        save_path=f"../output/Loss_curve/{args.model_name}_loss_curve_tensorboard_lr_{args.lr}_batch_{args.batch_size}.png"
    )
    print(f"Loss curve saved to ../output/Loss_curve/{args.model_name}_loss_curve_tensorboard_lr_{args.lr}_batch_{args.batch_size}.png")
except ValueError as e:
    print(e)
