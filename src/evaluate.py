import argparse
import json
import os
from PIL import Image
import torch 
from models.mvcnn_clip import MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP
from models.Students.MVAlexNet import MV_AlexNet
from dataset.dataset import MultiViewDataset, TestDataset
from utils import retrieve_images, extract_features
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

# 定义命令行参数解析器
parser = argparse.ArgumentParser(description="Extract features using MVCNN_CLIP model")
parser.add_argument("--model_name", type=str, default="MVCLIP_MLP", help="Model name (MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP)")
parser.add_argument("--model_num", type=str, default="9", help="Path to save the trained model (default: ../models/train_models/base/mvcnn_clip_01.pth)")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (default: 0.001)")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
parser.add_argument("--num_views", type=int, default=1, help="Number of views for MVCNN (default: 1)")
parser.add_argument("--test_dataset",type=str, default="DU", help="DU or DS")
args = parser.parse_args()

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
query_images = get_random_query_images(f"../data/ModelNet_random_30_final/{args.test_dataset}/retrieval", num_images=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

test_dataset = TestDataset(root_dir=f"../data/ModelNet_random_30_final/{args.test_dataset}/retrieval",transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)


# 加载模型
if args.model_name == "MVCNN_CLIP":
    model = MVCNN_CLIP(num_views = args.num_views).to(device)
elif args.model_name == "MVCLIP_CNN":
    model = MVCLIP_CNN(num_views = args.num_views).to(device)
elif args.model_name == "MVCLIP_MLP":
    model = MVCLIP_MLP(num_views = args.num_views).to(device)
elif args.model_name == "MV_AlexNet":
    model = MV_AlexNet(num_views = args.num_views).to(device)
elif args.model_name == "MV_AlexNet_dis":
    model = MV_AlexNet(num_views = args.num_views, is_dis = True).to(device)

model.load_state_dict(torch.load(f"../models/train_models/base/{args.model_name}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pth")['model_state_dict'])
model.eval()

# features, image_paths = extract_features(model, test_loader, device)


# 加载特征和路径
features = torch.load(f"../features/features_{args.model_name}/{args.test_dataset}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pt")
with open(f"../features/image_paths_{args.model_name}/{args.test_dataset}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.json", "r") as f:
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
    
    query_image = transform(query_image).unsqueeze(0).to(device) # 因为model是4维带有批次的，所以要加一个维度
    query_features = model(query_image).detach().cpu().numpy()

    if args.model_name == "MV_AlexNet_dis":
        results = retrieve_images(query_features[:,0,:], features[:,0,:], image_paths, model, top_k=10)
    # 对查询图像进行检索
    else:
        results = retrieve_images(query_features, features, image_paths, model, top_k=10)
    
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
os.makedirs(os.path.dirname(f"../output/result/{args.test_dataset}/{args.model_name}_epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.png"), exist_ok=True)
plt.savefig(f"../output/result/{args.test_dataset}/{args.model_name}_epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.png", dpi=300, bbox_inches='tight')
plt.show()

