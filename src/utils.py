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
    def __init__(self, dis_margin = 1.0, ang_margin = 0.2, rateGL = 0.5,rateOI = 0.5):
        super(RelationDisLoss, self).__init__()
        self.triplet_loss = TripletRelationLoss(dis_margin, ang_margin)
        self.rateGL = rateGL
        self.rateOI = rateOI

    def forward(self,t_anchor, t_positive, t_negative, s_anchor, s_positive, s_negative): # (32,50,256)
        len_dim = s_anchor.size()[1]

        # 归一化特征向量
        s_anchor = torch.nn.functional.normalize(s_anchor, p=2, dim=-1)
        s_positive = torch.nn.functional.normalize(s_positive, p=2, dim=-1)
        s_negative = torch.nn.functional.normalize(s_negative, p=2, dim=-1)
        t_anchor = torch.nn.functional.normalize(t_anchor, p=2, dim=-1)
        t_positive = torch.nn.functional.normalize(t_positive, p=2, dim=-1)
        t_negative = torch.nn.functional.normalize(t_negative, p=2, dim=-1)

        s_vector_pos = s_anchor - s_positive
        s_vector_neg = s_anchor - s_negative
        t_vector_pos = t_anchor - t_positive
        t_vector_neg = t_anchor - t_negative
        
        s_vector_pos_G = s_vector_pos[:, 0, :]
        s_vector_neg_G = s_vector_neg[:, 0, :]
        t_vector_pos_G = t_vector_pos[:, 0, :]
        t_vector_neg_G = t_vector_neg[:, 0, :]

        s_vector_pos_L = s_vector_pos[:, 1:, :]
        s_vector_neg_L = s_vector_neg[:, 1:, :]
        t_vector_pos_L = t_vector_pos[:, 1:, :]
        t_vector_neg_L = t_vector_neg[:, 1:, :]

        dis_pos_G = torch.mean(torch.norm(s_vector_pos_G - t_vector_pos_G, p = 2, dim = 1))
        dis_neg_G = torch.mean(torch.norm(s_vector_neg_G - t_vector_neg_G, p = 2, dim = 1))
        dis_pos_L = torch.mean(torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2), dim = 1).mean()
        dis_neg_L = torch.mean(torch.norm(s_vector_neg_L - t_vector_neg_L, p = 2, dim = 2), dim = 1).mean()

        cos_pos_G = torch.mean(torch.sum(s_vector_pos_G * t_vector_pos_G, dim = 1))
        cos_neg_G = torch.mean(torch.sum(s_vector_neg_G * t_vector_neg_G, dim = 1)) 
        cos_pos_L = torch.mean(torch.sum(s_vector_pos_L * t_vector_pos_L, dim = 2), dim = 1).mean()
        cos_neg_L = torch.mean(torch.sum(s_vector_neg_L * t_vector_neg_L, dim = 2), dim = 1).mean()
        
        # print("dis_pos_G_before sum",torch.norm(s_vector_pos_G - t_vector_pos_G, p = 2, dim = 1))
        # print("disG_shape",s_vector_pos_G.shape) #([16, 768])
        # print("disL_shape",s_vector_pos_L.shape) #([16, 49, 768])

        # print("G_norm_shape",torch.norm(s_vector_pos_G - t_vector_pos_G, p = 2, dim = 1).shape) #([16])
        # print("L_norm_shape",torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2).shape) #([16, 49])

        # print("G_sum",torch.sum(torch.norm(s_vector_pos_G - t_vector_pos_G, p = 2, dim = 1)).shape)# []
        # print("L_sum",torch.sum(torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2), dim = 1).shape) #([16])


        # print("dis_pos_G", dis_pos_G)
        # print("dis_pos_L", dis_pos_L)
        # print("disL0",torch.sum(torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2), dim = 1)[0])
 
        # print("dis_pos_L_before mean",torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2))
        # print("dis_pos_L", dis_pos_L)
        # print("dis_pos_L_sum/dim",torch.sum(torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2), dim = 1)/ (len_dim - 1))


        # dis_pos = torch.sum(torch.norm(s_vector_pos - t_vector_pos, p = 2, dim = 2), dim = 1)
        # dis_neg = torch.sum(torch.norm(s_vector_neg - t_vector_neg, p = 2, dim = 2), dim = 1)

        # cos_pos = torch.sum(torch.sum(s_vector_pos * t_vector_pos, dim = 2), dim = 1)
        # cos_neg = torch.sum(torch.sum(s_vector_neg * t_vector_neg, dim = 2), dim = 1)

        inner_losses_L = 0
        
        for t in range(1, len_dim): # 全局特征cls没有内部的距离
            anchor_t = s_anchor[:, t, :]
            positive_t = s_positive[:, t, :]
            negative_t = s_negative[:, t, :]
            inner_losses_L += self.triplet_loss(anchor_t, positive_t, negative_t)
        inner_losses_L /= (len_dim - 1)
        
        inner_losses_G = self.triplet_loss(s_anchor[:, 0, :], s_positive[:, 0, :], s_negative[:, 0, :]) # 全局特征cls

        # print("inner_losses_G", inner_losses_G)
        # print("inner_losses_L", inner_losses_L)

        loss_G = (dis_pos_G + dis_neg_G + cos_pos_G + cos_neg_G) * self.rateOI + inner_losses_G * (1 - self.rateOI)
        loss_L = (dis_pos_L + dis_neg_L + cos_pos_L + cos_neg_L) * self.rateOI + inner_losses_L * (1 - self.rateOI)


        # 归一化损失
        # loss_G_normalized = loss_G / (loss_G.detach().max() + 1e-8)  # 防止除以零
        # loss_L_normalized = loss_L / (loss_L.detach().max() + 1e-8)  # 防止除以零

        Loss = loss_G * self.rateGL + loss_L * (1 - self.rateGL)        
        # print("Loss",Loss.shape)
        return Loss.mean()
import torch
import torch.nn as nn

class SimpleFeatureDistillationLoss(nn.Module):
    def __init__(self):
        super(SimpleFeatureDistillationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self,t_anchor, t_positive, t_negative, s_anchor, s_positive, s_negative): 
        """
        计算教师模型和学生模型特征之间的均方误差损失。
        
        参数:
        t_anchor (torch.Tensor): 教师模型的锚点特征，形状为 (batch_size, seq_len, feature_dim)
        t_positive (torch.Tensor): 教师模型的正样本特征，形状为 (batch_size, seq_len, feature_dim)
        t_negative (torch.Tensor): 教师模型的负样本特征，形状为 (batch_size, seq_len, feature_dim)
        s_anchor (torch.Tensor): 学生模型的锚点特征，形状为 (batch_size, seq_len, feature_dim)
        s_positive (torch.Tensor): 学生模型的正样本特征，形状为 (batch_size, seq_len, feature_dim)
        s_negative (torch.Tensor): 学生模型的负样本特征，形状为 (batch_size, seq_len, feature_dim)
        
        返回:
        torch.Tensor: 特征蒸馏损失
        """
        # # 初始化总损失
        # total_loss = 0.0

        # # 获取序列长度
        # seq_len = t_anchor.size(1)

        # # 逐个时间步计算损失
        # for t in range(seq_len):
        #     # 提取当前时间步的特征
        #     t_anchor_t = t_anchor[:, t, :]
        #     s_anchor_t = s_anchor[:, t, :]

        #     # 计算每个特征的均方误差损失
        #     loss_anchor = self.mse_loss(t_anchor_t, s_anchor_t)

        #     # 沿着特征维度求和
        #     loss_anchor = torch.sum(loss_anchor, dim=1)

        #     # 累加当前时间步的损失
        #     total_loss += loss_anchor

        # # 计算平均损失
        # total_loss /= seq_len
        # + self.mse_loss(t_positive, s_positive) + self.mse_loss(t_negative, s_negative)

        # 计算均方误差损失
        loss = self.mse_loss(t_anchor, s_anchor)
        return loss

class CombineLoss(nn.Module):
    def __init__(self, dis_margin = 1.0, ang_margin = 0.2, rateGL = 0.5,rateOI = 0.5):
        super(CombineLoss, self).__init__()
        self.triplet_loss = TripletRelationLoss(dis_margin, ang_margin)
        self.rateGL = rateGL
        self.rateOI = rateOI
        self.mse_loss = nn.MSELoss()

    def forward(self,t_anchor, t_positive, t_negative, s_anchor, s_positive, s_negative): # (32,50,256)
        len_dim = s_anchor.size()[1]

        # 归一化特征向量
        s_anchor = torch.nn.functional.normalize(s_anchor, p=2, dim=-1)
        s_positive = torch.nn.functional.normalize(s_positive, p=2, dim=-1)
        s_negative = torch.nn.functional.normalize(s_negative, p=2, dim=-1)
        t_anchor = torch.nn.functional.normalize(t_anchor, p=2, dim=-1)
        t_positive = torch.nn.functional.normalize(t_positive, p=2, dim=-1)
        t_negative = torch.nn.functional.normalize(t_negative, p=2, dim=-1)

        s_vector_pos = s_anchor - s_positive
        s_vector_neg = s_anchor - s_negative
        t_vector_pos = t_anchor - t_positive
        t_vector_neg = t_anchor - t_negative
        
        s_vector_pos_G = s_vector_pos[:, 0, :]
        s_vector_neg_G = s_vector_neg[:, 0, :]
        t_vector_pos_G = t_vector_pos[:, 0, :]
        t_vector_neg_G = t_vector_neg[:, 0, :]

        s_vector_pos_L = s_vector_pos[:, 1:, :]
        s_vector_neg_L = s_vector_neg[:, 1:, :]
        t_vector_pos_L = t_vector_pos[:, 1:, :]
        t_vector_neg_L = t_vector_neg[:, 1:, :]

        dis_pos_G = torch.mean(torch.norm(s_vector_pos_G - t_vector_pos_G, p = 2, dim = 1))
        dis_neg_G = torch.mean(torch.norm(s_vector_neg_G - t_vector_neg_G, p = 2, dim = 1))
        dis_pos_L = torch.mean(torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2), dim = 1).mean()
        dis_neg_L = torch.mean(torch.norm(s_vector_neg_L - t_vector_neg_L, p = 2, dim = 2), dim = 1).mean()

        cos_pos_G = torch.mean(torch.sum(s_vector_pos_G * t_vector_pos_G, dim = 1))
        cos_neg_G = torch.mean(torch.sum(s_vector_neg_G * t_vector_neg_G, dim = 1)) 
        cos_pos_L = torch.mean(torch.sum(s_vector_pos_L * t_vector_pos_L, dim = 2), dim = 1).mean()
        cos_neg_L = torch.mean(torch.sum(s_vector_neg_L * t_vector_neg_L, dim = 2), dim = 1).mean()
        
        # print("dis_pos_G_before sum",torch.norm(s_vector_pos_G - t_vector_pos_G, p = 2, dim = 1))
        # print("disG_shape",s_vector_pos_G.shape) #([16, 768])
        # print("disL_shape",s_vector_pos_L.shape) #([16, 49, 768])

        # print("G_norm_shape",torch.norm(s_vector_pos_G - t_vector_pos_G, p = 2, dim = 1).shape) #([16])
        # print("L_norm_shape",torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2).shape) #([16, 49])

        # print("G_sum",torch.sum(torch.norm(s_vector_pos_G - t_vector_pos_G, p = 2, dim = 1)).shape)# []
        # print("L_sum",torch.sum(torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2), dim = 1).shape) #([16])


        # print("dis_pos_G", dis_pos_G)
        # print("dis_pos_L", dis_pos_L)
        # print("disL0",torch.sum(torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2), dim = 1)[0])
 
        # print("dis_pos_L_before mean",torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2))
        # print("dis_pos_L", dis_pos_L)
        # print("dis_pos_L_sum/dim",torch.sum(torch.norm(s_vector_pos_L - t_vector_pos_L, p = 2, dim = 2), dim = 1)/ (len_dim - 1))


        # dis_pos = torch.sum(torch.norm(s_vector_pos - t_vector_pos, p = 2, dim = 2), dim = 1)
        # dis_neg = torch.sum(torch.norm(s_vector_neg - t_vector_neg, p = 2, dim = 2), dim = 1)

        # cos_pos = torch.sum(torch.sum(s_vector_pos * t_vector_pos, dim = 2), dim = 1)
        # cos_neg = torch.sum(torch.sum(s_vector_neg * t_vector_neg, dim = 2), dim = 1)

        inner_losses_L = 0
        
        for t in range(1, len_dim): # 全局特征cls没有内部的距离
            anchor_t = s_anchor[:, t, :]
            positive_t = s_positive[:, t, :]
            negative_t = s_negative[:, t, :]
            inner_losses_L += self.triplet_loss(anchor_t, positive_t, negative_t)
        inner_losses_L /= (len_dim - 1)
        
        inner_losses_G = self.triplet_loss(s_anchor[:, 0, :], s_positive[:, 0, :], s_negative[:, 0, :]) # 全局特征cls

        # print("inner_losses_G", inner_losses_G)
        # print("inner_losses_L", inner_losses_L)

        loss_G = (dis_pos_G + dis_neg_G + cos_pos_G + cos_neg_G) * self.rateOI + inner_losses_G * (1 - self.rateOI)
        loss_L = (dis_pos_L + dis_neg_L + cos_pos_L + cos_neg_L) * self.rateOI + inner_losses_L * (1 - self.rateOI)

        Loss1 = loss_G * self.rateGL + loss_L * (1 - self.rateGL)

        Loss2 = self.mse_loss(t_anchor, s_anchor)


        # 动态权重调整（改进版）
        alpha = Loss2 / (Loss1 + Loss2 + 1e-8)
        beta = Loss1 / (Loss1 + Loss2 + 1e-8)

        # 结合损失函数
        TotalLoss = alpha * Loss1 + beta * Loss2

        return TotalLoss

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
