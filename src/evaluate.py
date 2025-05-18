import argparse
import json
import os
import time
from PIL import Image
import numpy as np
import torch 
from models.Teachers.CLIP import MV_CLIP,MV_CLIP_without_adpter
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
from retrieval_metric import eval_retrieval_metric

import pandas as pd

# 定义命令行参数解析器
parser = argparse.ArgumentParser(description="Extract features using MVCNN_CLIP model")
parser.add_argument("--model_name", type=str, default="MV_AlexNet", help="Model name (MVCNN_CLIP, MVCLIP_CNN, MVCLIP_MLP)")
parser.add_argument("--model_num", type=str, default="66", help="Path to save the trained model (default: ../models/train_models/base/mvcnn_clip_01.pth)")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (default: 0.001)")
parser.add_argument("--num_epochs", type=int, default=15,help="Number of epochs to train (default: 1)")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
parser.add_argument("--num_views", type=int, default=12, help="Number of views for MVCNN (default: 1)")
parser.add_argument("--test_dataset",type=str, default="OS-ESB-core", help="DU or DS")
parser.add_argument("--train_data", type=str, default="OS-ESB-core", help="The name of the train data")
parser.add_argument("--loss", type=str, default="SimpleFeatureDistillationLoss", help="The name of the train data")
parser.add_argument("--rGL", type=float, default=0.8, help="The rate of G:L")
parser.add_argument("--rOI", type=float, default=0.8, help="The rate of Out:In")
# parser.add_argument("--loss", type=str, default="RelationDisLoss", help="The name of the train data")
parser.add_argument("--order", type=int, default=1, help="The rate of Out:In")

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
# 提取类别和实例编号的函数
def extract_category_and_instance(path, data_dict = "q"):
    # print(path)
    parts = path.split(os.sep)
    # print("A",parts)
    instance = parts[-3]  # 实例编号
    if instance == "query" or instance == "target":
        instance = parts[-2]
    # print("B",instance)
    if data_dict == "q":
        category = data_dict_q[instance]  # 类别名称
    else:
        category = data_dict_t[instance]
    return category, instance

# 每个实例随机生成1组 num_views 张视图
def get_random_query_groups(root_dir, num_views=3):
    query_groups = []

    for instance in os.listdir(root_dir):
        instance_path = os.path.join(root_dir, instance)
        if root_dir not in ["../data/MN40-DS/query", "../data/MN40-DU/query",]: 
            instance_path = os.path.join(instance_path, "image")
        if os.path.isdir(instance_path):
            views = os.listdir(instance_path)
            if len(views) >= num_views:
                selected_views = random.sample(views, num_views)
                view_paths = [os.path.join(instance_path, v) for v in selected_views]
                view_images = [Image.open(p).convert("RGB") for p in view_paths]
                query_groups.append((view_images, view_paths))
    return query_groups



def calculate_metrics_category(query_groups, features, image_paths, model, device):
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
        metrics: 包含 mAP、NDCG@100、ANMRR 和 PR 曲线的字典。
    """
    seen_labels = [""]
    query_features = []
    query_labels = []
    inference_times = []  # 用于存储每次查询的推理时间
    
    for view_images, view_paths in tqdm(query_groups, desc="Extracting query features"):
        view_tensors = torch.stack([transform(view) for view in view_images]).to(device)

        start_time = time.time()  # 开始时间
        query_feature = model(view_tensors).detach().cpu().numpy()
        end_time = time.time()  # 结束时间
        inference_times.append(end_time - start_time)  # 记录推理时间

        if args.model_name in ["MV_AlexNet_dis", "MV_AlexNet_dis_Pre","MVCNN_CLIP_L","MV_CLIP_without_adpter"]:
            query_global = query_feature[:, 0, :]
            query_local = query_feature[:, 1:, :].reshape(query_feature.shape[0], -1)
            query_feature = np.concatenate([query_global, query_local], axis=1)
        
        query_features.append(query_feature)
        query_labels.append(extract_category_and_instance(view_paths[0], "q")[0])
    
    query_features = np.concatenate(query_features, axis=0)
    query_labels = np.array(query_labels)
    print("features", features.shape) # features (32, 768)
    # 对数据库特征进行相同的处理
    if args.model_name in ["MV_AlexNet_dis", "MV_AlexNet_dis_Pre","MVCNN_CLIP_L","MV_CLIP_without_adpter"]:
        db_features = features.reshape(features.shape[0], -1)
        # print("dill")
    else:
        db_features = features

    print("features_after", db_features.shape) # features (32, 768)
    

    # 检查 features 的形状
    # if len(db_features.shape) != 2:
    #     raise ValueError(f"Features must be a 2-dimensional array, but got shape {db_features.shape}")    
    
    target_labels = [extract_category_and_instance(path[0], "t")[0] for path in image_paths]
    target_labels = np.array(target_labels)
    print("query_labels", query_labels) # query_labels (32,)
    metrics = eval_retrieval_metric(query_features, db_features, query_labels, target_labels)
    # 添加平均推理时间到 metrics
    metrics["average_inference_time"] = np.mean(inference_times)

    return metrics

# def calculate_metrics_category(query_groups, features, image_paths, model, device, top_k=100):
#     """
#     针对数据库中相同类别作为正样本的场景计算指标（适配多视图查询）。
    
#     Args:
#         query_groups: 多视图查询列表，格式为 [(view_images, view_paths)]。
#         features: 数据库特征向量（形状 [n_samples, feature_dim]）。
#         image_paths: 数据库图像路径列表（长度 n_samples）。
#         model: 多视图聚合模型。
#         device: 设备（CPU/GPU）。
#         top_k: 检索的Top-K值。
    
#     Returns:
#         mAP: mAP@100（平均精度）。
#         top1_accuracy: Top-1准确率。
#         top10_accuracy: Top-10准确率。
#     """
#     aps = []
#     top1_correct = 0
#     top10_correct = 0

#     for view_images, view_paths in tqdm(query_groups, desc="Calculating metrics"):
#         # 提取查询特征
#         view_tensors = torch.stack([transform(view) for view in view_images]).to(device)
#         query_feature = model(view_tensors).detach().cpu().numpy()
        
#         # 获取查询类别
#         query_category = extract_category_and_instance(view_paths[0],"q")[0]  # 修改：提取类别而非实例
        
#         # 特殊处理MV_AlexNet_dis模型
#         if args.model_name in ["MV_AlexNet_dis", "MV_AlexNet_dis_Pre"]:
#             query_global = query_feature[:, 0, :]
#             query_local = query_feature[:, 1:, :].reshape(query_feature.shape[0], -1)
#             query_feature = np.concatenate([query_global, query_local], axis=1)
            
#             db_global = features[:, 0, :]
#             db_local = features[:, 1:, :].reshape(features.shape[0], -1)
#             db_features = np.concatenate([db_global, db_local], axis=1)
#         else:
#             db_features = features

#         # 执行检索
#         results = retrieve_images(query_feature, db_features, image_paths, model, top_k=top_k)
        
#         # 计算AP（Average Precision）
#         relevant_ranks = []
#         for rank, (retrieved_path, _) in enumerate(results, start=1):
#             retrieved_category = extract_category_and_instance(retrieved_path[0],"t")[0]
#             if retrieved_category == query_category:  # 正样本条件：类别相同
#                 relevant_ranks.append(rank)
        
#         # 计算当前查询的AP
#         ap = 0.0
#         if relevant_ranks:
#             precision_at_k = [ (i+1) / rank for i, rank in enumerate(relevant_ranks) ]
#             ap = sum(precision_at_k) / len(precision_at_k)
#             # 更新Top-K准确率
#             if relevant_ranks[0] <= 1:
#                 top1_correct += 1
#             if relevant_ranks[0] <= 10:
#                 top10_correct += 1
#         aps.append(ap)

#     # 计算全局指标
#     map = sum(aps) / len(aps) if len(aps) > 0 else 0.0
#     top1_accuracy = top1_correct / len(query_groups)
#     top10_accuracy = top10_correct / len(query_groups)

#     return map, top1_accuracy, top10_accuracy

import matplotlib.pyplot as plt

def plot_pr_curve(pr_data):
    """
    绘制 PR 曲线
    :param pr_data: 由 pr 函数返回的 PR 曲线数据，格式为 [precision_at_0.0, precision_at_0.1, ..., precision_at_1.0]
    """
    recall_thresholds = np.arange(0.0, 1.1, 0.1)  # 召回率阈值从 0.0 到 1.0，步长为 0.1
    precision_values = pr_data  # 精确率值

    plt.figure(figsize=(8, 6))
    plt.plot(recall_thresholds, precision_values, marker='o', linestyle='-', color='b')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

# test_dataset = TestMultiViewDataset(root_dir=f"../data/{args.test_dataset}", transform=transform, num_views=args.num_views, data = "query")
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
Pretrained_model_path = f"../models/exp/train_models/{args.train_data}/MV_AlexNet/best_model_lr_{args.lr}_batch_{args.batch_size}.pth"


# 加载模型
if args.model_name == "MVCNN_CLIP":
    model = MV_CLIP(num_views = args.num_views).to(device)
elif args.model_name == "MVCNN_CLIP_L":
    model = MV_CLIP(num_views = args.num_views).to(device)
elif args.model_name == "MVCLIP_CNN":
    model = MVCLIP_CNN(num_views = args.num_views).to(device)
elif args.model_name == "MVCLIP_MLP":
    model = MVCLIP_MLP(num_views = args.num_views).to(device)
elif args.model_name == "MV_AlexNet":
    model = MV_AlexNet(num_views = args.num_views).to(device)
elif args.model_name == "MV_AlexNet_dis_Pre":
    model = MV_AlexNet(num_views = args.num_views, is_dis = True, PretrainedModel_dir = Pretrained_model_path).to(device)
elif args.model_name == "MV_AlexNet_dis":
    model = MV_AlexNet(num_views = args.num_views, is_dis = True, is_pre = False).to(device)
elif args.model_name == "MV_CLIP_without_adpter":
    model = MV_CLIP_without_adpter(num_views = args.num_views).to(device)

if args.model_name == "MV_AlexNet_dis_Pre":
    if args.loss == "SimpleFeatureDistillationLoss":
        model.load_state_dict(torch.load(f"../models/exp/train_models/{args.train_data}/{args.model_name}/{args.loss}/best_model_lr_{args.lr}_batch_{args.batch_size}.pth")['model_state_dict'])
    elif args.loss == "RelationDisLoss":
        model.load_state_dict(torch.load(f"../models/exp/train_models/{args.train_data}/{args.model_name}/{args.loss}/best_model_lr_{args.lr}_batch_{args.batch_size}_rGL_{args.rGL}_rOI_{args.rOI}.pth")['model_state_dict'])
elif args.model_name != "MV_CLIP_without_adpter":
    model.load_state_dict(torch.load(f"../models/exp/train_models/{args.train_data}/{args.model_name}/best_model_lr_{args.lr}_batch_{args.batch_size}.pth")['model_state_dict'])

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
if args.model_name == "MV_AlexNet_dis_Pre":
    if args.loss == "SimpleFeatureDistillationLoss":
        path_1 = f"../features/exp/train_{args.train_data}/features_{args.model_name}/{args.loss}/{args.test_dataset}/{args.num_views}_lr_{args.lr}_batch_{args.batch_size}.pt"
        path_2 = f"../features/exp/train_{args.train_data}/image_paths_{args.model_name}/{args.loss}/{args.test_dataset}/{args.num_views}_lr_{args.lr}_batch_{args.batch_size}.json"
    else:
        path_1 = f"../features/exp/train_{args.train_data}/features_{args.model_name}/{args.loss}/{args.test_dataset}/{args.num_views}_lr_{args.lr}_batch_{args.batch_size}_rGL_{args.rGL}_rOI_{args.rOI}.pt"
        path_2 = f"../features/exp/train_{args.train_data}/image_paths_{args.model_name}/{args.loss}/{args.test_dataset}/{args.num_views}_lr_{args.lr}_batch_{args.batch_size}_rGL_{args.rGL}_rOI_{args.rOI}.json"
else:
    path_1 = f"../features/exp/train_{args.train_data}/features_{args.model_name}/{args.test_dataset}/{args.num_views}_lr_{args.lr}_batch_{args.batch_size}.pt"
    path_2 = f"../features/exp/train_{args.train_data}/image_paths_{args.model_name}/{args.test_dataset}/{args.num_views}_lr_{args.lr}_batch_{args.batch_size}.json"

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
ans = calculate_metrics_category(
    query_groups=query_groups,
    features=features,
    image_paths=image_paths,
    model=model,
    device=device,
)



# 定义 CSV 文件路径
csv_file_path = "../output/acc/accuracy_09.csv"

# 创建一个 DataFrame 存储结果
result = {
    "order": [args.order],
    "Model": [args.model_name],
    "Epochs": [args.model_num],
    "Learning Rate": [args.lr],
    "Batch Size": [args.batch_size],
    "num_views": [args.num_views],
    "Train Dataset": [args.train_data],
    "test_dataset": [args.test_dataset],
    "loss": [args.loss],
    "rGL": [args.rGL],
    "rOI": [args.rOI],
    "average_inference_time": [f"{ans['average_inference_time']:.4f}"],
    "mAP": [f"{ans['map']:.4f}"],
    "ndcg@100": [f"{ans['ndcg@100']:.4f}"],
    "anmrr": [f"{ans['anmrr']:.4f}"],
}

if args.model_name != "MV_AlexNet_dis_Pre":
    result['loss'] = "TripleLoss"

df = pd.DataFrame(result)

# 如果文件已存在，则追加数据；否则创建新文件
if os.path.exists(csv_file_path):
    df.to_csv(csv_file_path, mode="a", header=False, index=False)
else:
    df.to_csv(csv_file_path, mode="w", index=False)

print(f"结果已保存到 {csv_file_path}")

# pr_data = ans["s_pr"]
# plot_pr_curve(pr_data)