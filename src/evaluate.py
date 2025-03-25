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
from matplotlib.patches import Rectangle
from tqdm import tqdm

# 定义命令行参数解析器
parser = argparse.ArgumentParser(description="Extract features using MVCNN_CLIP model")
parser.add_argument("--model_name", type=str, default="MVCLIP_MLP", help="Model name (MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP)")
parser.add_argument("--model_num", type=str, default="14", help="Path to save the trained model (default: ../models/train_models/base/mvcnn_clip_01.pth)")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (default: 0.001)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for feature extraction")
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
def get_all_images(root_dir):
    """
    获取数据集中所有图像的路径和图像对象。

    :param root_dir: 数据集的根目录。
    :return: 列表，包含所有图像的元组 (image, image_path)。
    """
    all_images = []
    
    # 遍历根目录下的所有类别
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            # 遍历类别下的所有实例
            for instance in os.listdir(category_path):
                instance_path = os.path.join(category_path, instance)
                if os.path.isdir(instance_path):
                    # 遍历实例下的所有视图
                    for view in os.listdir(instance_path):
                        view_path = os.path.join(instance_path, view)
                        if os.path.isfile(view_path):  # 确保是文件
                            try:
                                # 打开图像并转换为 RGB 格式
                                image = Image.open(view_path).convert("RGB")
                                all_images.append((image, view_path))
                            except Exception as e:
                                print(f"无法加载图像 {view_path}: {e}")
    
    return all_images

def calculate_instance_accuracy(query_images, results_list):
    correct_count = 0
    total_count = 0
    category_stats = {}

    for idx, (query_image, query_image_path) in enumerate(query_images):
        category, instance = extract_category_and_instance(query_image_path)
        results = results_list(idx)
        
        if category not in category_stats:
            category_stats[category] = {"correct": 0,"total":0}

        for path, similarity in results:
            retrieved_category, retrieved_instance = extract_category_and_instance(path)

            if retrieved_instance == instance:
                correct_count += 1
                category_stats[category]["correct"] += 1
            total_count += 1
            category_stats[category]["total"] += 1
    
    if total_count == 0:
        return 0.0, {}

    category_accuracy = {}
    for category, stats in category_stats.items():
        if stats["total"] > 0:
            category_accuracy[category] = stats["corret"] / stats["total"]
        else:
            category_accuracy[category] = 0.0
    return correct_count / total_count, category_accuracy

def calculate_top1_accuracy(query_images, features, image_paths, model, device):
    """
    计算 Top-1 检索准确率。

    :param query_images: 查询图像的列表，每个元素为 (image, image_path)。
    :param features: 数据集的特征向量 (numpy 数组)。
    :param image_paths: 数据集的图像路径列表。
    :param model: 特征提取模型。
    :param device: 设备 (CPU/GPU)。
    :return: Top-1 检索准确率。
    """
    correct_count = 0
    total_count = len(query_images)

    for query_image, query_image_path in tqdm(query_images, desc="Calculating Top-1 Accuracy", unit="query"):
        # 提取查询图像的特征
        query_image_tensor = transform(query_image).unsqueeze(0).to(device)
        query_features = model(query_image_tensor).detach().cpu().numpy()

        # 检索最相似的图像（Top-1）
        if args.model_name == "MV_AlexNet_dis" or args.model_name == "MV_AlexNet_dis_Pre":
            results = retrieve_images(query_features[:,0,:], features[:,0,:], image_paths, model, top_k=1)
        # 对查询图像进行检索
        else:
            results = retrieve_images(query_features, features, image_paths, model, top_k=1)

        if len(results) > 0:
            # 获取排名第一的检索结果
            top1_path, _ = results[0]
            # 提取查询图像和检索结果的实例编号
            query_category, query_instance = extract_category_and_instance(query_image_path)
            retrieved_category, retrieved_instance = extract_category_and_instance(top1_path)

            # 判断实例是否匹配
            if retrieved_instance == query_instance:
                correct_count += 1

    # 计算 Top-1 检索准确率
    top1_accuracy = correct_count / total_count if total_count > 0 else 0.0
    return top1_accuracy

def show_pic(query_images, image_paths, model, features, device):
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

        if args.model_name == "MV_AlexNet_dis" or args.model_name == "MV_AlexNet_dis_Pre":
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
    # # 保存图像到文件
    # os.makedirs(os.path.dirname(f"../output/result/{args.test_dataset}/{args.model_name}_epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.png"), exist_ok=True)
    # plt.savefig(f"../output/result/{args.test_dataset}/{args.model_name}_epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.png", dpi=300, bbox_inches='tight')
    # plt.show()



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
elif args.model_name == "MV_AlexNet_dis_Pre":
    model = MV_AlexNet(num_views = args.num_views, is_dis = True).to(device)
elif args.model_name == "MV_AlexNet_dis":
    model = MV_AlexNet(num_views = args.num_views, is_dis = True, is_pre = False).to(device)

model.load_state_dict(torch.load(f"../models/train_models/base/{args.model_name}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pth")['model_state_dict'])
model.eval()

# features, image_paths = extract_features(model, test_loader, device)

root_dir = f"../data/ModelNet_random_30_final/{args.test_dataset}/retrieval"

# 获取随机查询图像
query_images = get_random_query_images(root_dir , num_images=10)

all_images = get_all_images(root_dir)

# 加载特征和路径
features = torch.load(f"../features/features_{args.model_name}/{args.test_dataset}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pt")

with open(f"../features/image_paths_{args.model_name}/{args.test_dataset}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.json", "r") as f:
    image_paths = json.load(f)


# show_pic(query_images, image_paths, model, features, device)

# 计算 Top-1 检索准确率
top1_accuracy = calculate_top1_accuracy(all_images, features, image_paths, model, device)
print(f"Top-1 Retrieval Accuracy: {top1_accuracy:.4f}")

import pandas as pd

# 定义 CSV 文件路径
csv_file_path = "../output/accuracy_results.csv"

# 创建一个 DataFrame 存储结果
result = {
    "Model": [args.model_name],
    "Epochs": [args.model_num],
    "Learning Rate": [args.lr],
    "Batch Size": [args.batch_size],
    "Top-1 Accuracy": [f"{top1_accuracy:.4f}"]
}
df = pd.DataFrame(result)

# 如果文件已存在，则追加数据；否则创建新文件
if os.path.exists(csv_file_path):
    df.to_csv(csv_file_path, mode="a", header=False, index=False)
else:
    df.to_csv(csv_file_path, mode="w", index=False)

print(f"结果已保存到 {csv_file_path}")