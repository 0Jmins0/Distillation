import argparse
import os
import torch
import json
from models.mvcnn_clip import MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP
from models.Students.MVAlexNet import MV_AlexNet
from dataset.dataset import TestDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入 tqdm 库

# 定义命令行参数解析器
parser = argparse.ArgumentParser(description="Extract features using MVCNN_CLIP model")
parser.add_argument("--model_name", type=str, default="MVCLIP_MLP", help="Model name (MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP)")
parser.add_argument("--model_num", type=str, default="14", help="Path to save the trained model (default: ../models/train_models/base/mvcnn_clip_01.pth)")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (default: 0.001)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for feature extraction")
parser.add_argument("--num_views", type=int, default=1, help="Number of views for MVCNN (default: 1)")
parser.add_argument("--test_dataset",type=str, default="DU", help="DU or DS")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

# 加载测试数据集
test_dataset = TestDataset(root_dir=f"../data/ModelNet_random_30_final/{args.test_dataset}/retrieval", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

# 加载模型
if args.model_name == "MVCNN_CLIP":
    model = MVCNN_CLIP(num_views = args.num_views).to(device)
elif args.model_name == "MVCLIP_CNN":
    model = MVCLIP_CNN(num_views = args.num_views).to(device)
elif args.model_name == "MVCLIP_MLP":
    model = MVCLIP_MLP(num_views = args.num_views).to(device)
elif args.model_name == "MV_AlexNet":
    model = MV_AlexNet(num_views = args.num_views).to(device)
elif args.model_name == "MV_AlexNet_dis_Pre":
    model = MV_AlexNet(num_views = args.num_views, is_dis = True).to(device)
elif args.model_name == "MV_AlexNet_dis":
    model = MV_AlexNet(num_views = args.num_views, is_dis = True, is_pre = False).to(device)

model.load_state_dict(torch.load(f"../models/train_models/base/{args.model_name}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pth")['model_state_dict'])
model.eval()

# 提取特征并保存
features = []
image_paths = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Extracting features", unit="batch"):
        images, paths = batch
        images = images.to(device)
        # print(images.shape)
        batch_features = model(images)
        features.append(batch_features.cpu())
        image_paths.extend(paths)

# 将特征和路径保存到文件
features = torch.cat(features, dim=0)  # 将特征拼接成一个张量
# 确保保存模型的目录存在

path_1 = f"../features/features_{args.model_name}/{args.test_dataset}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pt"
path_2 = f"../features/image_paths_{args.model_name}/{args.test_dataset}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.json"

os.makedirs(os.path.dirname(path_1), exist_ok=True)
os.makedirs(os.path.dirname(path_2), exist_ok=True)
torch.save(features, f"../features/features_{args.model_name}/{args.test_dataset}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pt")  # 保存特征
with open(f"../features/image_paths_{args.model_name}/{args.test_dataset}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.json", "w") as f:
    json.dump(image_paths, f)  # 保存路径