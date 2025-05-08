import argparse
import json
import os
from PIL import Image
import numpy as np
import torch 
from models.mvcnn_clip import MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP
from models.Students.MVAlexNet import MV_AlexNet
from dataset.dataset import MultiViewDataset, TestDataset,TestMultiViewDataset
from utils import retrieve_images, extract_features
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

# 定义命令行参数解析器
parser = argparse.ArgumentParser(description="Extract features using MVCNN_CLIP model")
parser.add_argument("--model_name", type=str, default="MVCNN_CLIP", help="Model name (MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP)")
parser.add_argument("--model_num", type=str, default="14", help="Path to save the trained model (default: ../models/train_models/base/mvcnn_clip_01.pth)")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (default: 0.001)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for feature extraction")
parser.add_argument("--num_views", type=int, default=1, help="Number of views for MVCNN (default: 1)")
parser.add_argument("--test_dataset",type=str, default="OS-NTU-core", help="DU or DS")
parser.add_argument("--train_data", type=str, default="OS-NTU-core", help="The name of the train data")
args = parser.parse_args()

# 提取类别和实例编号的函数
def extract_category_and_instance(path, data_dict = "q"):
    
    parts = path.split(os.sep)
    # print("A",parts)
    instance = parts[-3]  # 实例编号
    if data_dict == "q":
        category = data_dict_q[instance]  # 类别名称
    else:
        category = data_dict_t[instance]
    return category, instance

# 随机选择10张不同类别的图像作为查询图像
def get_random_query_images(root_dir, num_queries=10, num_views = 3):
    """
    随机选择多视图查询样本（每个查询包含同一实例的多个视图）。
    
    Args:
        root_dir: 数据集根目录。
        num_queries: 查询数量。
        num_views: 每个查询的视图数。
    
    Returns:
        query_groups: 格式为 [ (view_list, instance_path) ]，
                     其中 view_list 是同一实例的多个视图图像列表。
    """

    categories = os.listdir(root_dir)
    query_groups = []
    selected_categories = random.sample(categories, num_queries)
    
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
                    if len(views) >= num_views:
                        # 随机选择一张视图
                        selected_views = random.sample(views, num_views)
                        query_image_paths = [os.path.join(instance_path, selected_view) for selected_view in selected_views]
                        query_images =[Image.open(query_image_path).convert("RGB") for query_image_path in query_image_paths]
                        query_groups.append((query_images, query_image_paths))
    
    return query_groups
# 每个实例构造 30/num_views 个多视图样本
def get_all_query_groups(root_dir, num_views=15):
    query_groups = []
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            for instance in os.listdir(category_path):
                instance_path = os.path.join(category_path, instance)
                if os.path.isdir(instance_path):
                    views = sorted(os.listdir(instance_path))  # 确保顺序固定
                    if len(views) >= num_views:
                        # 按顺序分组（非重叠）
                        for i in range(0, len(views) - num_views + 1, num_views):
                            view_group = views[i:i+num_views]
                            view_paths = [os.path.join(instance_path, v) for v in view_group]
                            view_images = [Image.open(p).convert("RGB") for p in view_paths]
                            query_groups.append((view_images, view_paths))
    return query_groups

# 每个实例随机生成1组 num_views 张视图
def get_random_query_groups(root_dir, num_views=3):
    query_groups = []

    for instance in os.listdir(root_dir):
        instance_path = os.path.join(root_dir, instance)
        instance_path = os.path.join(instance_path, "image")
        if os.path.isdir(instance_path):
            views = os.listdir(instance_path)
            if len(views) >= num_views:
                selected_views = random.sample(views, num_views)
                view_paths = [os.path.join(instance_path, v) for v in selected_views]
                view_images = [Image.open(p).convert("RGB") for p in view_paths]
                query_groups.append((view_images, view_paths))
    return query_groups

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
        category, instance = extract_category_and_instance(query_image_path, "q")
        results = results_list(idx)
        
        if category not in category_stats:
            category_stats[category] = {"correct": 0,"total":0}

        for path, similarity in results:
            retrieved_category, retrieved_instance = extract_category_and_instance(path, "t")

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
            query_category, query_instance = extract_category_and_instance(query_image_path, "q")
            retrieved_category, retrieved_instance = extract_category_and_instance(top1_path,"t")

            # 判断实例是否匹配
            if retrieved_instance == query_instance:
                correct_count += 1

    # 计算 Top-1 检索准确率
    top1_accuracy = correct_count / total_count if total_count > 0 else 0.0
    return top1_accuracy

def calculate_metrics_instance(query_groups, features, image_paths, model, device, top_k = 100):
    """
    针对数据库中仅有一个正样本的场景计算指标（适配多视图查询）。
    
    Args:
        query_groups: 多视图查询列表，格式为 [(view_images, view_paths)]。
        features: 数据库特征向量（形状 [n_samples, feature_dim]）。
        image_paths: 数据库图像路径列表（长度 n_samples）。
        model: 多视图聚合模型。
        device: 设备（CPU/GPU）。
        top_k: 检索的Top-K值。
    
    Returns:
        mAP: mAP@100（正样本排名的平均倒数）。
        top1_accuracy: Top-1准确率。
        top10_accuracy: Top-10准确率。
    """
    aps = []
    top1_correct = 0
    top10_correct = 0


    for view_images, view_paths in tqdm(query_groups, desc= "Calculating metrics"):

        # print("len",(view_paths))
        view_tensors = torch.stack([transform(view) for view in view_images]).to(device)
        # 提取查询图像的特征

        query_feature = model(view_tensors).detach().cpu().numpy()
        # with torch.no_grad():
        #     query_feature = model(view_tensors)
        #     query_feature = query_feature.detach().cpu().numpy().squeeze()
        
        if args.model_name == "MV_AlexNet_dis" or args.model_name == "MV_AlexNet_dis_Pre":
            query_feature = torch.from_numpy(query_feature).float()  # 转换为 PyTorch 张量


            # # 提取全局特征和局部特征
            # query_global_features = query_feature[:, 0, :]  # 形状为 [num_queries, feature_dim]
            # query_local_features = query_feature[:, 1:, :].mean(dim=1)  # 形状为 [num_queries, feature_dim]

            # features_global_features = features[:, 0, :]  # 形状为 [num_samples, feature_dim]
            # features_local_features = features[:, 1:, :].mean(dim=1)  # 形状为 [num_samples, feature_dim]

            # # 将全局特征和局部特征拼接起来
            # query_combined_features = torch.cat([query_global_features, query_local_features], dim=1)  # 形状为 [num_queries, 2 * feature_dim]
            # features_combined_features = torch.cat([features_global_features, features_local_features], dim=1)  # 形状为 [num_samples, 2 * feature_dim]

            # results = retrieve_images(query_feature[:,0,:], features[:,0,:], image_paths, model, top_k=top_k)
            
            # 提取全局特征和局部特征
            query_global_features = query_feature[:, 0, :]  # 形状为 [num_queries, feature_dim]
            query_local_features = query_feature[:, 1:, :].reshape(query_feature.shape[0], -1)  # 形状为 [num_queries, 49 * feature_dim]
            
            # 将全局特征和局部特征拼接起来
            query_combined_features = np.concatenate([query_global_features, query_local_features], axis=1)  # 形状为 [num_queries, (1 + 49) * feature_dim]
            
            # 提取数据库特征
            features_global_features = features[:, 0, :]  # 形状为 [num_samples, feature_dim]
            features_local_features = features[:, 1:, :].reshape(features.shape[0], -1)  # 形状为 [num_samples, 49 * feature_dim]
            
            # 将全局特征和局部特征拼接起来
            features_combined_features = np.concatenate([features_global_features, features_local_features], axis=1)  # 形状为 [num_samples, (1 + 49) * feature_dim]

            
            results = retrieve_images(query_combined_features , features_combined_features, image_paths, model, top_k=top_k)

        # 对查询图像进行检索
        else:
            results = retrieve_images(query_feature, features, image_paths, model, top_k=top_k)

        # results = retrieve_images(query_feature[:,0,:], features[:,0,:], image_paths, model, top_k=top_k)

        query_instance = extract_category_and_instance(view_paths[0], "q")[1]

        positive_rank = None
        for rank, (retrieved_path, _) in enumerate(results, start = 1):
            retrieved_instance = extract_category_and_instance(retrieved_path[0],"t")[1]
            # print("retrieved_instance",retrieved_instance,query_instance)
            if retrieved_instance == query_instance :
                positive_rank = rank
                break
        
        ap = 1.0 / positive_rank if positive_rank is not None else 0.0
        aps.append(ap)

        if positive_rank is not None:
            if positive_rank <= 1:
                top1_correct += 1
            if positive_rank <= 10:
                top10_correct += 1
    map = sum(aps) / len(aps) if len(aps) > 0 else 0.0
    top1_accuracy = top1_correct / len(query_groups)
    top10_accuracy = top10_correct / len(query_groups)

    return map, top1_accuracy, top10_accuracy

def calculate_metrics_category(query_groups, features, image_paths, model, device, top_k=100):
    """
    针对数据库中相同类别作为正样本的场景计算指标（适配多视图查询）。
    
    Args:
        query_groups: 多视图查询列表，格式为 [(view_images, view_paths)]。
        features: 数据库特征向量（形状 [n_samples, feature_dim]）。
        image_paths: 数据库图像路径列表（长度 n_samples）。
        model: 多视图聚合模型。
        device: 设备（CPU/GPU）。
        top_k: 检索的Top-K值。
    
    Returns:
        mAP: mAP@100（平均精度）。
        top1_accuracy: Top-1准确率。
        top10_accuracy: Top-10准确率。
    """
    aps = []
    top1_correct = 0
    top10_correct = 0

    for view_images, view_paths in tqdm(query_groups, desc="Calculating metrics"):
        # 提取查询特征
        view_tensors = torch.stack([transform(view) for view in view_images]).to(device)
        query_feature = model(view_tensors).detach().cpu().numpy()
        
        # 获取查询类别
        query_category = extract_category_and_instance(view_paths[0],"q")[0]  # 修改：提取类别而非实例
        
        # 特殊处理MV_AlexNet_dis模型
        if args.model_name in ["MV_AlexNet_dis", "MV_AlexNet_dis_Pre"]:
            query_global = query_feature[:, 0, :]
            query_local = query_feature[:, 1:, :].reshape(query_feature.shape[0], -1)
            query_feature = np.concatenate([query_global, query_local], axis=1)
            
            db_global = features[:, 0, :]
            db_local = features[:, 1:, :].reshape(features.shape[0], -1)
            db_features = np.concatenate([db_global, db_local], axis=1)
        else:
            db_features = features

        # 执行检索
        results = retrieve_images(query_feature, db_features, image_paths, model, top_k=top_k)
        
        # 计算AP（Average Precision）
        relevant_ranks = []
        for rank, (retrieved_path, _) in enumerate(results, start=1):
            retrieved_category = extract_category_and_instance(retrieved_path[0],"t")[0]
            if retrieved_category == query_category:  # 正样本条件：类别相同
                relevant_ranks.append(rank)
        
        # 计算当前查询的AP
        ap = 0.0
        if relevant_ranks:
            precision_at_k = [ (i+1) / rank for i, rank in enumerate(relevant_ranks) ]
            ap = sum(precision_at_k) / len(precision_at_k)
            # 更新Top-K准确率
            if relevant_ranks[0] <= 1:
                top1_correct += 1
            if relevant_ranks[0] <= 10:
                top10_correct += 1
        aps.append(ap)

    # 计算全局指标
    map = sum(aps) / len(aps) if len(aps) > 0 else 0.0
    top1_accuracy = top1_correct / len(query_groups)
    top10_accuracy = top10_correct / len(query_groups)

    return map, top1_accuracy, top10_accuracy

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

# test_dataset = TestMultiViewDataset(root_dir=f"../data/{args.test_dataset}", transform=transform, num_views=args.num_views, data = "query")
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)


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

# features, image_paths = extract_features(model, test_loader, device)

root_dir = f"../data/{args.test_dataset}/query"

# # 获取随机查询图像
# query_images = get_random_query_images(root_dir , num_images=10)

# all_images = get_all_images(root_dir)

# if args.num_views == 15:

#     # 加载特征和路径
#     features = torch.load(f"../features/features_{args.model_name}/{args.test_dataset}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pt")

#     with open(f"../features/image_paths_{args.model_name}/{args.test_dataset}/epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.json", "r") as f:
#         image_paths = json.load(f)

# else:
# 加载特征和路径
path_1 = f"../features/train_{args.train_data}/features_{args.model_name}/{args.test_dataset}/12_epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.pt"
path_2 = f"../features/train_{args.train_data}/image_paths_{args.model_name}/{args.test_dataset}/12_epochs_{args.model_num}_lr_{args.lr}_batch_{args.batch_size}.json"

features = torch.load(path_1)

with open(path_2, "r") as f:
    image_paths = json.load(f)

dict_dir_t = f"../data/{args.test_dataset}/target_label.txt"
dict_dir_q = f"../data/{args.test_dataset}/query_label.txt"

# 初始化一个空字典
data_dict_t = {}
# 打开文件并读取内容
with open(dict_dir_t, "r") as file:
    for line in file:
        # 去掉每行的换行符并以逗号分隔
        key, value = line.strip().split(",")
        # 将键值对存入字典
        data_dict_t[key] = value

# 初始化一个空字典
data_dict_q = {}
# 打开文件并读取内容
with open(dict_dir_q, "r") as file:
    for line in file:
        # 去掉每行的换行符并以逗号分隔
        key, value = line.strip().split(",")
        # 将键值对存入字典
        data_dict_q[key] = value

# show_pic(query_images, image_paths, model, features, device)

# 加载多视图查询样本（每组查询包含同一实例的所有视图路径）


# query_groups = get_all_query_groups(root_dir, num_views=args.num_views)

query_groups = get_random_query_groups(root_dir, num_views=args.num_views)
# print(f"查询样本数量: {query_groups[1][1]}")

# 计算指标
mAP, top1_accuracy, top10_accuracy = calculate_metrics_category(
    query_groups=query_groups,
    features=features,
    image_paths=image_paths,
    model=model,
    device=device,
    top_k=100
)

print(f"mAP@100: {mAP:.4f}")
print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
print(f"Top-10 Accuracy: {top10_accuracy:.4f}")
import pandas as pd

# 定义 CSV 文件路径
csv_file_path = "../output/accuracy_results.csv"

# 创建一个 DataFrame 存储结果
result = {
    "Model": [args.model_name],
    "Epochs": [args.model_num],
    "Learning Rate": [args.lr],
    "Batch Size": [args.batch_size],
    "num_views": [args.num_views],
    "Train Dataset": [args.train_data],
    "test_dataset": [args.test_dataset],
    "Top-1 Accuracy": [f"{top1_accuracy:.4f}"],
    "Top-10 Accuracy": [f"{top10_accuracy:.4f}"],
    "mAP": [f"{mAP:.4f}"],
}
df = pd.DataFrame(result)

# 如果文件已存在，则追加数据；否则创建新文件
if os.path.exists(csv_file_path):
    df.to_csv(csv_file_path, mode="a", header=False, index=False)
else:
    df.to_csv(csv_file_path, mode="w", index=False)

print(f"结果已保存到 {csv_file_path}")