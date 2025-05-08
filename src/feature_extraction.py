import argparse
import os
import numpy as np
import torch
import json
from models.mvcnn_clip import MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP
from models.Students.MVAlexNet import MV_AlexNet
from dataset.dataset import TestDataset, TestMultiViewDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入 tqdm 库
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    
    # 将图像张量堆叠成一个批次
    images = torch.stack(images, dim=0)
    
    # 确保路径列表的维度是 [batch_size, num_views]
    # 默认情况下，paths 的维度已经是 [batch_size, num_views]
    # 如果你需要转置，可以使用以下代码
    # paths = list(map(list, zip(*paths)))
    
    return images, paths


# 定义命令行参数解析器
parser = argparse.ArgumentParser(description="Extract features using MVCNN_CLIP model")
parser.add_argument("--model_name", type=str, default="MVCNN_CLIP", help="Model name (MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP)")
parser.add_argument("--model_num", type=str, default="14", help="Path to save the trained model (default: ../models/train_models/base/mvcnn_clip_01.pth)")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (default: 0.001)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for feature extraction")
parser.add_argument("--num_views", type=int, default=12, help="Number of views for MVCNN (default: 1)")
parser.add_argument("--test_dataset",type=str, default="OS-NTU-core", help="DU or DS")
parser.add_argument("--train_data", type=str, default="OS-NTU-core", help="The name of the train data")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

# 加载测试数据集
test_dataset = TestMultiViewDataset(root_dir=f"../data/{args.test_dataset}", transform=transform, num_views=args.num_views, data = "target")
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)
# 创建 DataLoader 并使用自定义的 collate_fn
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=custom_collate_fn)

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

model.load_state_dict(torch.load(f"../models/train_models/{args.train_data}/{args.model_name}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pth")['model_state_dict'])
model.eval()

# 提取特征并保存
features = []
image_paths = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Extracting features", unit="batch"):
        images, paths = batch
        # print("images",images.shape) #images torch.Size([64, 15, 3, 224, 224])
        # print("paths", np.array(paths).shape)
        images = images.view(-1, *images.shape[-3:])  # (batch_size * num_views, C, H, W)
        images = images.to(device)
        # print(images.shape)
        batch_features = model(images)  # (batch_size * num_views, feature_dim)
        # print("batch_features", batch_features.shape)  # batch_features torch.Size([64, 768])
        features.append(batch_features.cpu())
        image_paths.extend (paths)

# print(features.shape)
# 将特征和路径保存到文件
features = torch.cat(features, dim=0)  # 将特征拼接成一个张量


print(f"Extracted {features.shape} batches of features.") # Extracted torch.Size([1197, 768]) batches of features.
print(f"Extracted {len(image_paths)} image paths.")
# 确保保存模型的目录存在

path_1 = f"../features/train_{args.train_data}/features_{args.model_name}/{args.test_dataset}/{args.num_views}_epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pt"
path_2 = f"../features/train_{args.train_data}/image_paths_{args.model_name}/{args.test_dataset}/{args.num_views}_epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.json"

os.makedirs(os.path.dirname(path_1), exist_ok=True)
os.makedirs(os.path.dirname(path_2), exist_ok=True)
torch.save(features, path_1)  # 保存特征
with open(path_2, "w") as f:
    json.dump(image_paths, f)  # 保存路径