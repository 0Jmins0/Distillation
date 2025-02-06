import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.mvcnn_clip import MVCNN_CLIP
from dataset.dataset import MultiViewDataset
from torchvision import transforms
from utils import TripletLoss
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt


# 定义命令行参数解析器
parser = argparse.ArgumentParser(description="Train MVCNN_CLIP model")
parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training (default: 10)")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train (default: 1)")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 0.001)")
parser.add_argument("--margin", type=float, default=1.0, help="Margin for triplet loss (default: 1.0)")
parser.add_argument("--num_views", type=int, default=15, help="Number of views for MVCNN (default: 10)")
parser.add_argument("--data_root", type=str, default="../data/ModelNet_random_30_final/DS/train", help="Root directory of the dataset (default: ../data/ModelNet_random_30_final/DS/train)")
parser.add_argument("--model_path", type=str, default="../models/train_models/base/mvcnn_clip_5.pth", help="Path to save the trained model (default: ../models/train_models/base/mvcnn_clip_01.pth)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # 先将图像调整到更大的尺寸
#     transforms.RandomResizedCrop(224),  # 随机裁剪到目标尺寸
#     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为0.5
#     transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转，概率为0.5
#     transforms.RandomRotation(degrees=15),  # 随机旋转，角度范围为±15度
#     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 随机高斯模糊
#     transforms.ToTensor(),  # 转换为张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
# ])

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 直接调整到 CLIP 的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # CLIP 专用参数
])

print("loading train data.......")
train_dataset = MultiViewDataset(root_dir="../data/ModelNet_random_30_final/DS/train",transform=transform, num_views=args.num_views)
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, drop_last=True)

print("finished loading train data.......")
model = MVCNN_CLIP(num_views = args.num_views).to(device)
criterion = TripletLoss(margin = 0.5)
optimizer = optim.Adam([
    {"params": model.net_1.parameters(), "lr": 1e-6},  # 主干网络低学习率
    # {"params": model.net_2.parameters(), "lr": 1e-6}   # 新增层高学习率
])

# 确保保存模型的目录存在
os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

num_epochs = args.num_epochs

# 检查是否存在已保存的模型文件
if os.path.exists(args.model_path):
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch + 1}")
else:
    print("No saved model found. Starting training from scratch.")
    start_epoch = 0

losses = []

model.train()
for epoch in range(start_epoch + 1, num_epochs):
    epoch_loss = 0.0
    i = 0
    for anchor_images, positive_images, negative_images in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"):
        i += 1
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

        losses.append(loss.item())
        if(i % 20 == 0):
            # 确保存储路径存在
            save_path = f"../models/train_models/base/pics/loss_curve_epoch_{epoch}_batch_{i}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(losses) + 1), losses, label="Training Loss")
            plt.xlabel("Batch")
            plt.ylabel("Loss")
            plt.title(f"Training Loss Over Batches (Epoch {epoch})")
            plt.legend()
            plt.grid(True)
            plt.savefig(save_path)


    
    epoch_loss /= len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 保存模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"../models/train_models/base/mvcnn_clip_{epoch}.pth")


# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(losses) + 1), losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig("../models/train_models/base/loss_curve.png")

# 显示图像
plt.show()
