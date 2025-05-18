import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor

class CLIPAdapter(nn.Module):
    def __init__(self, input_dim):
        super(CLIPAdapter, self).__init__()
        self.fc0 = nn.Linear(input_dim, 768)  # 将特征维度从768映射到768
        # 动态设置输入维度和输出维度
        self.fc1 = nn.Linear(768, 256)

    def forward(self, x):
        # x: (batch_size, num_patches, hidden_size)
        if x.size(2) != 768:
            # 如果输入的特征维度不是768，则进行线性变换
            x = self.fc0(x)
        x = self.fc1(x)  # 将特征维度从768映射到256
        return x

class MV_CLIP(nn.Module):
    def __init__(self, num_views = 12):
        # 初始化 MVCNN_CLIP 类，并设置默认的视图数量为12
        super(MV_CLIP, self).__init__()
        # model_dir = "/home/xyzhang/project/Distillation/models/pretrained/clip-vit-large-patch14"
        model_dir = "/home/xyzhang/project/Distillation/models/pretrained/clip-vit-base-patch32"

        # 从预训练模型中加载 CLIPVisionModel
        self.clip_model = CLIPVisionModel.from_pretrained(model_dir)
        # 从预训练模型中加载 CLIPProcessor
        self.clip_processor = CLIPProcessor.from_pretrained(model_dir)
        # 设置视图数量
        self.num_views = num_views

        self.net_1 = self.clip_model

        # 在 MVCNN_CLIP 的 __init__ 中解冻部分层
        for name, param in self.net_1.named_parameters():
            if "vision_model.encoder.layers.23" in name:  # 解冻最后几层
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 获取预训练模型的隐藏维度
        self.input_dim = self.clip_model.config.hidden_size
        # 动态设置适配器的输入和输出维度
        # self.adapter = CLIPAdapter(input_dim=self.input_dim)

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
        
        
        # features = self.adapter(features)  # (N * num_views, patch_size(50/257), 256)
        
        # print("T_before_pool", features.shape) #(480,50,768)
        # 视角池化
        features = features.view(-1, self.num_views, features.shape[1], features.size(-1))
        # print("T_after_pool", features.shape) #(480,50,256)
        # 对每个样本的视图特征最大池化，保留每个样本的最显著特征
        # 最终(N, hidden_size)
        features = torch.max(features, dim = 1)[0] 
        # print("T",features.shape)
        return features # (32,50,768)

class MV_CLIP_without_adpter(nn.Module):
    def __init__(self, num_views = 12):
        # 初始化 MVCNN_CLIP 类，并设置默认的视图数量为12
        super(MV_CLIP_without_adpter, self).__init__()

        # model_dir = "/home/xyzhang/project/Distillation/models/pretrained/clip-vit-large-patch14"
        model_dir = "/home/xyzhang/project/Distillation/models/pretrained/clip-vit-base-patch32"

        # 从预训练模型中加载 CLIPVisionModel
        self.clip_model = CLIPVisionModel.from_pretrained(model_dir)
        # 从预训练模型中加载 CLIPProcessor
        self.clip_processor = CLIPProcessor.from_pretrained(model_dir)
        # 设置视图数量
        self.num_views = num_views

        self.net_1 = self.clip_model


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
        # 视角池化
        features = features.view(-1, self.num_views, features.shape[1], features.size(-1))
        features = features[:,:,0:1,:] # 取第一个视图的特征
        # print("T_after_pool", features.shape) #(480,50,256)
        # 对每个样本的视图特征最大池化，保留每个样本的最显著特征
        # 最终(N, hidden_size)
        features = torch.max(features, dim = 1)[0] 
        # print("T",features.shape)
        return features # (32,50,768)