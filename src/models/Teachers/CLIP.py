import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor

class CLIPAdapter(nn.Module):
    def __init__(self):
        super(CLIPAdapter, self).__init__()
        # 适配器层：将特征从768通道映射到256通道
        self.fc1 = nn.Linear(768, 256)  # 将特征维度从768映射到256

    def forward(self, x):
        # x: (batch_size, num_patches, hidden_size)
        x = self.fc1(x)  # 将特征维度从768映射到256
        return x

class MV_CLIP(nn.Module):
    def __init__(self, num_views = 12):
        # 初始化 MVCNN_CLIP 类，并设置默认的视图数量为12
        super(MV_CLIP, self).__init__()
        # 从预训练模型中加载 CLIPVisionModel
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # 从预训练模型中加载 CLIPProcessor
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # 设置视图数量
        self.num_views = num_views

        self.adapter = CLIPAdapter()  # 添加适配器模块
        
    def forward(self, x):
        # x: (batch_size * num_views, C, H, W)
        # print(x.shape)
        N, C, H, W = x.size()   
        x = x.view(-1, self.num_views,C, H, W)
        # 调整维度顺序，将视图维度（num_views）移到通道维度（C）之后，然后将张量重新整形为(N * num_views, C, H, W)。
        # 这样，每个视图都被视为一个独立的图像样本。?????????
        x = x.permute(0,2,1,3,4).contiguous().view(-1, C, H, W)
        
        # pooler_output是CLIP模型的输出特征，形状为(N * num_views, hidden_size)
        # (N * num_views, hidden_size)
        # features = self.net_1(x).pooler_output 

        clip_output = self.clip_model(x)
        features = clip_output.last_hidden_state 
        # print("T_before_pool", features.shape) #(480,50,768)
        
        features = self.adapter(features)  # (N * num_views, 50, 256)
        # 视角池化
        features = features.view(-1, self.num_views, 50, features.size(-1))
        # 对每个样本的视图特征最大池化，保留每个样本的最显著特征
        # 最终(N, hidden_size)
        features = torch.max(features, dim = 1)[0] 
        # print("T",features.shape)
        return features # (32,50,768)