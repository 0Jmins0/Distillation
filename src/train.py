import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.mvcnn_clip import MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP
from models.Teachers.DINOv2 import MV_DINOv2, MV_DINOv2_without_adapter
from models.Teachers.CLIP import MV_CLIP
from models.Students.MVAlexNet import MV_AlexNet
from dataset.dataset import MultiViewDataset
from torchvision import transforms
from utils import TripletLoss, RelationDisLoss, read_tensorboard_data,plot_tensorboard_data,SimpleFeatureDistillationLoss,CombineLoss
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import os
from torch.utils.data import random_split
import torch.nn.functional as F
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 定义命令行参数解析器
parser = argparse.ArgumentParser(description="Train MVCNN_CLIP model")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 10)")
parser.add_argument("--num_epochs", type=int, default=100,help="Number of epochs to train (default: 1)")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (default: 0.001)")
parser.add_argument("--margin", type=float, default=1.0, help="Margin for triplet loss (default: 1.0)")
parser.add_argument("--num_views", type=int, default=12, help="Number of views for MVCNN (default: 10)")
parser.add_argument("--data_root", type=str, default="../data/ModelNet_random_30_final/DS/train", help="Root directory of the dataset (default: ../data/ModelNet_random_30_final/DS/train)")
parser.add_argument("--model_num", type=str, default=0,help="Path to save the trained model (default: ../models/train_models/base/mvcnn_clip_01.pth)")
parser.add_argument("--model_name", type=str, default="MV_AlexNet", help="The name of the model")
parser.add_argument("--train_data", type=str, default="OS-ABO-core", help="The name of the train data")
parser.add_argument("--loss", type=str, default="SimpleFeatureDistillationLoss", help="The name of the train data")
parser.add_argument("--rGL", type=float, default=0.8, help="The rate of G:L")
parser.add_argument("--rOI", type=float, default=0.8, help="The rate of Out:In")
parser.add_argument("--model_T", type=str, default="CLIP_B32")
# parser.add_argument("--loss", type=str, default="RelationDisLoss", help="The name of the train data")


args = parser.parse_args()

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
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 直接调整到 CLIP 的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # CLIP 专用参数
])

# 加载整个训练集
print("loading train data.......")

data_root = f"../data/{args.train_data}/train"
train_dataset = MultiViewDataset(root_dir=data_root, transform=transform, num_views=args.num_views)
print("finished loading train data.......")

# 切分训练集和验证集
# 假设我们切分出 20% 的数据作为验证集


target_size = (7,7)
if args.model_name == "MV_AlexNet_dis" or args.model_name == "MV_AlexNet_dis_Pre":
    if args.model_T == "CLIP_B32":
        print("Using CLIP_B32 as the teacher model")
        model_T = MV_CLIP(num_views = args.num_views).to(device)
        target_size = (7,7)
    elif args.model_T == "DINOv2_B14":
        print("Using DINOv2_B14 as the teacher model")
        model_T = MV_DINOv2(num_views = args.num_views).to(device)
        target_size = (16,16)
    model_T.eval()
    Pretrained_model_path = f"../models/exp/train_models/{args.train_data}/MV_AlexNet/best_model_lr_1e-06_batch_{args.batch_size}.pth"

print("finished loading train data.......")
if args.model_name == "MVCNN_CLIP":
    model = MV_CLIP(num_views = args.num_views).to(device)
    criterion = TripletLoss(margin = 0.5)
    init_lr = args.lr
    optimizer = optim.Adam([
    {"params": model.net_1.parameters(), "lr": init_lr},  # 主干网络低学习率
    # {"params": model.net_2.parameters(), "lr": 1e-4}   # 新增层高学习率
])
elif args.model_name == "MVCNN_CLIP_L":
    model = MV_CLIP(num_views = args.num_views).to(device)
    criterion = TripletLoss(margin = 0.5)
    init_lr = args.lr
    optimizer = optim.Adam([
    {"params": model.net_1.parameters(), "lr": init_lr},  # 主干网络低学习率
    # {"params": model.net_2.parameters(), "lr": 1e-4}   # 新增层高学习率
])
elif args.model_name == "MV_AlexNet":
    print("Training MV_AlexNet")
    model = MV_AlexNet(num_views = args.num_views, is_dis = False).to(device)
    criterion = TripletLoss(margin = 0.5)
    init_lr = args.lr
    optimizer = optim.Adam([
    {"params": model.features.features.parameters(), "lr": init_lr},
    # {"params": model.features.fc_features.parameters(), "lr": args.lr},
    {"params": model.retrieval.parameters(), "lr": init_lr}   # 新增层高学习率
])
elif args.model_name == "MV_AlexNet_dis_Pre":
    print("Training MV_AlexNet_dis_Pre")
    if args.loss == "SimpleFeatureDistillationLoss":
        criterion = SimpleFeatureDistillationLoss()
    elif args.loss == "RelationDisLoss":
        criterion = RelationDisLoss(dis_margin = 1.0, ang_margin = 0.2, rateGL = args.rGL, rateOI = args.rOI)
    elif args.loss == "CombineLoss":
        criterion = CombineLoss(dis_margin = 1.0, ang_margin = 0.2, rateGL = args.rGL, rateOI = args.rOI)
    model = MV_AlexNet(target_size = target_size, num_views = args.num_views, is_dis = True, batch_size = args.batch_size, PretrainedModel_dir = Pretrained_model_path).to(device)
    init_lr = args.lr * 0.1
    optimizer = optim.Adam([
    {"params": model.features.features.parameters(), "lr": init_lr},
    {"params": model.retrieval.parameters(), "lr": init_lr}   # 新增层高学习率
])


if args.model_name == "MV_AlexNet_dis_Pre":
    if args.loss == "SimpleFeatureDistillationLoss":
        model_path = f"../models/exp/train_models/{args.train_data}/{args.model_name}/{args.loss}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pth"
    else:
        model_path = f"../models/exp/train_models/{args.train_data}/{args.model_name}/{args.loss}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}_rGL_{args.rGL}_rOI_{args.rOI}.pth"
else:
    model_path = f"../models/exp/train_models/{args.train_data}/{args.model_name}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pth"

# 确保保存模型的目录存在
os.makedirs(os.path.dirname(model_path), exist_ok=True)

num_epochs = args.num_epochs


# tensorboard
if args.model_name == "MV_AlexNet_dis_Pre":
    if args.loss == "SimpleFeatureDistillationLoss":
        log_dir = f"../output/tensorboard_logs/exp/{args.train_data}/{args.model_name}/{args.model_T}/{args.loss}/lr_{args.lr}_batch_{args.batch_size}"
    else:
        log_dir = f"../output/tensorboard_logs/exp/{args.train_data}/{args.model_name}/{args.model_T}/{args.loss}/lr_{args.lr}_batch_{args.batch_size}_rGL_{args.rGL}_rOI_{args.rOI}"
else:
    log_dir = f"../output/tensorboard_logs/exp/{args.train_data}/{args.model_name}/lr_{args.lr}_batch_{args.batch_size}"
os.makedirs(log_dir, exist_ok=True)




# 添加学习率调度器
# scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 每 5 个 epoch 学习率乘以 0.5
# 余弦退火调度器
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

save_path = None
if args.model_name == "MV_AlexNet_dis_Pre":
    if args.loss == "SimpleFeatureDistillationLoss":
        save_path = f"../models/exp/train_models/{args.train_data}/{args.model_name}/{args.model_T}/{args.loss}/best_model_lr_{args.lr}_batch_{args.batch_size}.pth"
    else:
        save_path = f"../models/exp/train_models/{args.train_data}/{args.model_name}/{args.model_T}/{args.loss}/best_model_lr_{args.lr}_batch_{args.batch_size}_rGL_{args.rGL}_rOI_{args.rOI}.pth"
else:
    save_path = f"../models/exp/train_models/{args.train_data}/{args.model_name}/best_model_lr_{args.lr}_batch_{args.batch_size}.pth"

# 检查是否存在已保存的模型文件
if os.path.exists(save_path):
    print(f"Loading model from {save_path}")
    checkpoint = torch.load(save_path, map_location=device)
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
# 检查日志文件是否存在
if os.path.exists(log_dir) and os.listdir(log_dir):
    writer = SummaryWriter(log_dir=log_dir, purge_step=start_step)  # previous_step 是上次记录的步数
else:
    writer = SummaryWriter(log_dir=log_dir, purge_step=start_step)

step = start_step

# 初始化最佳性能指标
best_validation_loss = float('inf')  # 假设我们以最低的验证损失为目标
best_model_path = None

model.train()
print("before train")
last_save_path = None  # 用于存储上一个 epoch 的模型路径

validation_split = 0.2
train_size = int((1 - validation_split) * len(train_dataset))
validation_size = len(train_dataset) - train_size
train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

# 创建训练集和验证集的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)


# kfold = KFold(n_splits=5, shuffle=True, random_state=2025)

# for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
#     train_sub = torch.utils.data.Subset(train_dataset, train_idx)
#     validation_sub = torch.utils.data.Subset(train_dataset, val_idx)

#     # 创建 DataLoader
#     train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True)
#     validation_loader = DataLoader(validation_sub, batch_size=args.batch_size, shuffle=False)


for epoch in range(start_epoch + 1, num_epochs):
    model.train()
    # print("in the loop")
    epoch_loss = 0.0
    torch.cuda.empty_cache()
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

        # 对特征进行 L2 归一化
        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        positive_features = F.normalize(positive_features, p=2, dim=1)
        negative_features = F.normalize(negative_features, p=2, dim=1)

        if args.model_name == "MV_AlexNet_dis" or args.model_name == "MV_AlexNet_dis_Pre":
            with torch.no_grad():
                t_anchor_features = model_T(anchor_images)
                t_positive_features = model_T(positive_images)
                t_negative_features = model_T(negative_images)

            # 对教师模型的特征也进行归一化
            t_anchor_features = F.normalize(t_anchor_features, p=2, dim=1)
            t_positive_features = F.normalize(t_positive_features, p=2, dim=1)
            t_negative_features = F.normalize(t_negative_features, p=2, dim=1)

        # print("anchor_features", anchor_features.shape) # anchor_features torch.Size([32, 768])

        if args.model_name == "MV_AlexNet_dis" or args.model_name == "MV_AlexNet_dis_Pre":
            loss = criterion(t_anchor_features, t_positive_features, t_negative_features,anchor_features, positive_features, negative_features)
        else:  
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

        torch.cuda.empty_cache()

    epoch_loss /= len(train_loader)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Validation
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for anchor_images, positive_images, negative_images in tqdm(validation_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"):
            anchor_images = anchor_images.view(-1, *anchor_images.shape[-3:])  # (batch_size * num_views, C, H, W)
            positive_images = positive_images.view(-1, *positive_images.shape[-3:])  # (batch_size * num_views, C, H, W)
            negative_images = negative_images.view(-1, *negative_images.shape[-3:])  # (batch_size * num_views, C, H, W)

            anchor_images, positive_images, negative_images = anchor_images.to(device), positive_images.to(device), negative_images.to(device)

            optimizer.zero_grad()
            anchor_features = model(anchor_images)
            positive_features = model(positive_images)
            negative_features = model(negative_images)

            # 对特征进行 L2 归一化
            anchor_features = F.normalize(anchor_features, p=2, dim=1)
            positive_features = F.normalize(positive_features, p=2, dim=1)
            negative_features = F.normalize(negative_features, p=2, dim=1)

            if args.model_name == "MV_AlexNet_dis" or args.model_name == "MV_AlexNet_dis_Pre":
                with torch.no_grad():
                    t_anchor_features = model_T(anchor_images)
                    t_positive_features = model_T(positive_images)
                    t_negative_features = model_T(negative_images)

                # 对教师模型的特征也进行归一化
                t_anchor_features = F.normalize(t_anchor_features, p=2, dim=1)
                t_positive_features = F.normalize(t_positive_features, p=2, dim=1)
                t_negative_features = F.normalize(t_negative_features, p=2, dim=1)

            # print("anchor_features", anchor_features.shape) # anchor_features torch.Size([32, 768])

            if args.model_name == "MV_AlexNet_dis" or args.model_name == "MV_AlexNet_dis_Pre":
                loss = criterion(t_anchor_features, t_positive_features, t_negative_features,anchor_features, positive_features, negative_features)
            else:  
                loss = criterion(anchor_features, positive_features, negative_features)
            validation_loss += loss.item()

    validation_loss /= len(validation_loader)
    writer.add_scalar("Loss/validation", validation_loss, epoch)
    print(f"Validation Loss: {validation_loss:.4f}")


    # 将每个 epoch 的平均损失写入 TensorBoard
    writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)


    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存最佳模型
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model_path = save_path
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, best_model_path)
        print(f"Saved best model to {best_model_path}")
    
    scheduler.step()
    # 更新学习率
    # if args.model_name == "MV_AlexNet_dis_Pre":
    # else:
    #     lr = init_lr * (1 - epoch / num_epochs) ** 0.9
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

# 关闭 SummaryWriter
writer.close()
