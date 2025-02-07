import json
import os
from PIL import Image
import torch 
from models.mvcnn_clip import MVCNN_CLIP
from dataset.dataset import MultiViewDataset, TestDataset
from utils import retrieve_images, extract_features
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt


# 提取类别和实例编号的函数
def extract_category_and_instance(path):
    parts = path.split(os.sep)
    category = parts[-3]  # 类别名称
    instance = parts[-2]  # 实例编号
    return category, instance

# 随机选择10张不同类别的图像作为查询图像
def get_random_query_images(root_dir, num_images=10):
    categories = os.listdir(root_dir)
    query_images = []
    selected_categories = random.sample(categories, num_images)
    
    for category in selected_categories:
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            # 获取该类别下的所有实例
            instances = os.listdir(category_path)
            if instances:
                # 随机选择一个实例
                selected_instance = random.choice(instances)
                instance_path = os.path.join(category_path, selected_instance)
                if os.path.isdir(instance_path):
                    # 获取该实例下的所有视图
                    views = os.listdir(instance_path)
                    if views:
                        # 随机选择一张视图
                        selected_view = random.choice(views)
                        query_image_path = os.path.join(instance_path, selected_view)
                        query_image = Image.open(query_image_path).convert("RGB")
                        query_images.append((query_image, query_image_path))
    
    return query_images

# 获取随机查询图像
query_images = get_random_query_images("../data/ModelNet_random_30_final/DU/retrieval", num_images=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

test_dataset = TestDataset(root_dir="../data/ModelNet_random_30_final/DU/retrieval",transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)


model = MVCNN_CLIP(num_views=1).to(device)
model.load_state_dict(torch.load("../models/train_models/base/mvcnn_clip_9.pth")['model_state_dict'])
model.eval()

# features, image_paths = extract_features(model, test_loader, device)


# 加载特征和路径
features = torch.load("features.pt")
with open("image_paths.json", "r") as f:
    image_paths = json.load(f)


from matplotlib.patches import Rectangle

# 创建一个大图来显示所有查询图像和检索结果
fig, axes = plt.subplots(len(query_images), 11, figsize=(20, 15))  # 每行11列（1个查询图像 + 10个检索结果）

for idx, (query_image, query_image_path) in enumerate(query_images):
    # 显示查询图像
    ax = axes[idx, 0]
    ax.imshow(query_image)
    category, instance = extract_category_and_instance(query_image_path)
    ax.set_title(f"Query\n{category}/{instance}", fontsize=8)
    ax.axis('off')
    
    # 对查询图像进行检索
    results = retrieve_images(query_image, features, image_paths, model, top_k=10)
    
    # 显示检索结果
    for col, (path, similarity) in enumerate(results):
        retrieved_image = Image.open(path).convert("RGB")
        ax = axes[idx, col + 1]
        retrieved_category, retrieved_instance = extract_category_and_instance(path)
        
        # 显示图像
        ax.imshow(retrieved_image)
        ax.set_title(f"{retrieved_category}/{retrieved_instance}\nSim: {similarity:.2f}", fontsize=8)
        ax.axis('off')
        
        # 如果检索实例和查询实例不一样，在图像外画一个红框
        if retrieved_instance != instance:
            # 获取图像的宽度和高度
            img_width, img_height = retrieved_image.size
            # 添加边框，边框距离图像本身有10像素
            rect = Rectangle((-10, -10), img_width + 20, img_height + 20, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # 调整显示范围，确保边框能够完整显示
            ax.set_xlim(-20, img_width + 10)  # 设置x轴范围
            ax.set_ylim(img_height + 10, -20)  # 设置y轴范围（注意y轴方向是反的）


# 调整布局
plt.tight_layout()
# 保存图像到文件
plt.savefig("retrieval_results.png", dpi=300, bbox_inches='tight')
plt.show()

