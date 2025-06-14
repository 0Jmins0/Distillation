import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class MV_DINOv2(nn.Module):
    def __init__(self, num_views=12):
        super(MV_DINOv2, self).__init__()
        
        # 加载 DINOv2 的预训练模型
        model_dir = "/home/xyzhang/project/Distillation/models/pretrained/DINOv2-b14"
        self.processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False)
        self.dinov2_model = AutoModel.from_pretrained(model_dir)

        self.num_views = num_views
        
        self.net_1 = self.dinov2_model

        self.input_dim = self.dinov2_model.config.hidden_size


    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(-1, self.num_views, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)
        
        dinov2_output = self.dinov2_model(x)
        features = dinov2_output.last_hidden_state
        
        features = features.view(-1, self.num_views, features.shape[1], features.size(-1))
        features = torch.max(features, dim=1)[0]  # 对每个样本的视图特征进行最大池化
        return features

class MV_DINOv2_without_adapter(nn.Module):
    def __init__(self, num_views=12):
        super(MV_DINOv2_without_adapter, self).__init__()

        # 加载 DINOv2 的预训练模型
        model_dir = "/home/xyzhang/project/Distillation/models/pretrained/DINOv2-b14"
        self.processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False)
        self.dinov2_model = AutoModel.from_pretrained(model_dir)
        # self.dinov2_model = DINOv2Model.from_pretrained(model_dir)
        # self.dinov2_processor = DINOv2Processor.from_pretrained(model_dir)
        self.num_views = num_views

        self.net_1 = self.dinov2_model

    def forward(self, x):
        N, C, H, W = x.size()
        print(x.shape) # 12,3,244,244
        x = x.view(-1, self.num_views, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)
        
        dinov2_output = self.dinov2_model(x)
        features = dinov2_output.last_hidden_state
        # print("T_before_pool", features.shape) #12,257,768
        
        features = features.view(-1, self.num_views, features.shape[1], features.size(-1))
        features = features[:, :, 0:1, :]  # 只取第一个视图的特征
        # print("T_after_pool", features.shape) #1,12,1,768
        features = torch.max(features, dim=1)[0]  # 对每个样本的视图特征进行最大池化
        # print("T",features.shape) # 1,1,768
        return features
