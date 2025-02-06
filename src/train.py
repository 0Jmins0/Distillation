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
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 10)")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train (default: 1)")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 0.001)")
parser.add_argument("--margin", type=float, default=1.0, help="Margin for triplet loss (default: 1.0)")
parser.add_argument("--num_views", type=int, default=8, help="Number of views for MVCNN (default: 10)")
parser.add_argument("--data_root", type=str, default="../data/ModelNet_random_30_final/DS/train", help="Root directory of the dataset (default: ../data/ModelNet_random_30_final/DS/train)")
parser.add_argument("--model_path", type=str, default="../models/train_models/base/mvcnn_clip_5.pth", help="Path to save the trained model (default: ../models/train_models/base/mvcnn_clip_01.pth)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

print("loading train data.......")
train_dataset = MultiViewDataset(root_dir="../data/ModelNet_random_30_final/DS/train",transform=transform)
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, drop_last=True)

print("finished loading train data.......")
model = MVCNN_CLIP(num_views = args.batch_size).to(device)
criterion = TripletLoss(margin = 1.0)
optimizer = optim.Adam(model.parameters(),lr = args.lr)

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
    for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        anchor_features = model(anchor)
        positive_features = model(positive)
        negative_features = model(negative)

        loss = criterion(anchor_features, positive_features, negative_features)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss /= len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

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
