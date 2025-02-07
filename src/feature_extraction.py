import torch
import json
from models.mvcnn_clip import MVCNN_CLIP
from dataset.dataset import TestDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入 tqdm 库

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

# 加载测试数据集
test_dataset = TestDataset(root_dir="../data/ModelNet_random_30_final/DU/retrieval", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

# 加载模型
model = MVCNN_CLIP(num_views=1).to(device)
model.load_state_dict(torch.load("../models/train_models/base/mvcnn_clip_9.pth")['model_state_dict'])
model.eval()

# 提取特征并保存
features = []
image_paths = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Extracting features", unit="batch"):
        images, paths = batch
        images = images.to(device)
        batch_features = model(images)
        features.append(batch_features.cpu())
        image_paths.extend(paths)

# 将特征和路径保存到文件
features = torch.cat(features, dim=0)  # 将特征拼接成一个张量
torch.save(features, "features.pt")  # 保存特征
with open("image_paths.json", "w") as f:
    json.dump(image_paths, f)  # 保存路径