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

class TripletRelationLoss(nn.Module):
    def __init__(self, dis_margin = 1.0, ang_margin = 0.2):
        super(TripletRelationLoss, self).__init__()
        self.dis_margin = dis_margin
        self.ang_margin = ang_margin
    def distance_loss(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p = 2, dim = 1)
        distance_negative = torch.norm(anchor - negative, p = 2, dim = 1)
        losses = torch.relu(distance_positive - distance_negative + self.dis_margin)
        return losses.mean()
    def angle_loss(self, anchor, positive, negative):
        cos_positive = torch.sum(anchor * positive, dim = 1)
        cos_negative = torch.sum(anchor * negative, dim = 1)
        losses = torch.relu(cos_positive - cos_negative + self.ang_margin)
        return losses.mean()
    def forward(self, anchor, positive, negative):
        return self.distance_loss(anchor, positive, negative) + self.angle_loss(anchor, positive, negative)

class RelationDisLoss(nn.Module):
    def __init__(self, dis_margin = 1.0, ang_margin = 0.2):
        super(RelationDisLoss, self).__init__()
        self.triplet_loss = TripletRelationLoss(dis_margin, ang_margin)

    def forward(self,t_anchor, t_positive, t_negative, s_anchor, s_positive, s_negative): # (32,50,256)
        s_vector_pos = s_anchor - s_positive
        s_vector_neg = s_anchor - s_negative
        t_vector_pos = t_anchor - t_positive
        t_vector_neg = t_anchor - t_negative

        dis_pos = torch.sum(torch.norm(s_vector_pos - t_vector_pos, p = 2, dim = 2), dim = 1)
        dis_neg = torch.sum(torch.norm(s_vector_neg - t_vector_neg, p = 2, dim = 2), dim = 1)

        cos_pos = torch.sum(torch.sum(s_vector_pos * t_vector_pos, dim = 2), dim = 1)
        cos_neg = torch.sum(torch.sum(s_vector_neg * t_vector_neg, dim = 2), dim = 1)

        inner_losses = 0
        len_dim = s_anchor.size()[1]
        for t in range(0, len_dim): # 全局特征cls没有内部的距离
            anchor_t = s_anchor[:, t, :]
            positive_t = s_positive[:, t, :]
            negative_t = s_negative[:, t, :]
            inner_losses += self.triplet_loss(anchor_t, positive_t, negative_t)

        inner_losses /= len_dim

        Loss = (dis_pos + dis_neg + cos_pos + cos_neg ) / len_dim + inner_losses
        # print("Loss",Loss.shape)
        return Loss.mean()

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

def retrieve_images(query_features, features, image_paths, model, top_k = 5, transform = None, device = None):
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

    # print("q", query_features.shape, "f", features.shape)
    # query_image = transform(query_image).unsqueeze(0).to(device) # 因为model是4维带有批次的，所以要加一个维度
    # query_features = model(query_image).detach().cpu().numpy()

    #[(1,D) + (N,D) -> (1,N) -> (N,)]
    # print(query_features.shape, features.shape)
    similarities = cosine_similarity(query_features, features).flatten()

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # print(top_indices)

    # print("top_indices", len(image_paths),len(similarities), len(top_indices))

    return [(image_paths[i], similarities[i]) for i in top_indices]

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def read_tensorboard_data(log_dir, tag):
    """从 TensorBoard 日志文件中读取指定标签的数据"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    if tag in ea.Tags()["scalars"]:
        return ea.Scalars(tag)
    else:
        raise ValueError(f"Tag '{tag}' not found in TensorBoard logs.")

def plot_tensorboard_data(data, title, xlabel, ylabel, save_path):
    """绘制数据并保存为图片"""
    steps = [x.step for x in data]
    values = [x.value for x in data]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, label="Training Loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
