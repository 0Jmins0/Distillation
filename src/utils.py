import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class TripletLoss(nn.Module):
    """
    三元组损失函数，用于训练模型学习特征空间中的相似性。
    """    
    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        计算三元组损失。
        param anchor: 锚点图像的特征向量 (N, D)
        param positive: 正样本图像的特征向量 (N, D)
        param negative: 负样本图像的特征向量 (N, D)
        return: 三元组损失值
        """
        # L2 范数
        distance_positive = torch.norm(anchor - positive, p = 2, dim = 1)
        distance_negative = torch.norm(anchor - negative, p = 2, dim = 1)
        # 确保非负
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def extract_features(model, data_loader, device):
    """
    提取数据集的特征向量。
    model: 特征提取模型
    data_loader: 数据加载器
    device: 设备 (CPU/GPU)
    return: 特征向量和对应的图像路径
    """    
    model.eval()
    features = []
    image_paths = []
    
    with torch.no_grad():# 关闭梯度计算
        for images, paths in tqdm(data_loader, desc="Extracting features", unit="batch"): # 遍历数据加载器的每个批次
            images = images.to(device)
            batch_features = model(images).cpu().numpy() # 提取特征
            features.append(batch_features)
            image_paths.extend(paths)
    features = np.vstack(features) # 将所有批次的特征向量叠成一个矩阵
    # print(image_paths)
    return features, image_paths

def retrieve_images(query_image, features, image_paths, model, top_k = 5, transform = None, device = None):
    """
    根据查询图像检索最相似的图像。
    :param query_image: 查询图像 (PIL.Image)
    :param features: 数据集的特征向量 (numpy数组)
    :param image_paths: 数据集的图像路径列表
    :param top_k: 返回最相似的前K个图像
    :param transform: 查询图像的预处理变换
    :param device: 设备 (CPU/GPU)
    :return: 最相似的图像路径及其相似度
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_image = transform(query_image).unsqueeze(0).to(device) # 因为model是4维带有批次的，所以要加一个维度
    query_features = model(query_image).detach().cpu().numpy()

    #[(1,D) + (N,D) -> (1,N) -> (N,)]
    similarities = cosine_similarity(query_features, features).flatten()

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(image_paths[i], similarities[i]) for i in top_indices]