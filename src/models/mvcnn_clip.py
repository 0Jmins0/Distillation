import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor

class MVCNN_CLIP(nn.Module):
    def __init__(self, num_views = 12):
        super(MVCNN_CLIP, self).__init__()
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.num_views = num_views

        self.net_1 = self.clip_model
        self.net_2 = nn.Sequential(
            nn.Linear(self.net_1.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512,128)
        )

    def forward(self, x):
        # x: (batch_size * num_views, C, H, W)

        N, C, H, W = x.size()
        x = x.view(-1, self.num_views,C, H, W)
        # 调整维度顺序，将视图维度（num_views）移到通道维度（C）之后，然后将张量重新整形为(N * num_views, C, H, W)。
        # 这样，每个视图都被视为一个独立的图像样本。?????????
        x = x.permute(0,2,1,3,4).contigugous().view(-1, C, H, W)
        
        inputs = self.clip_processor(images=x, return_tensors="pt")
        # pooler_output是CLIP模型的输出特征，形状为(N * num_views, hidden_size)
        # (N * num_views, hidden_size)
        features = self.net_1(**inputs).pooler_output 

        # 视角池化
        features = features.view(-1, self.num_views, features.size(-1))
        # 对每个样本的视图特征最大池化，保留每个样本的最显著特征
        # 最终(N, hidden_size)
        features = torch.max(features, dim = 1)[0] 

        # 通过CNN2
        features = self.net_2(features)
        return features
