import torch
import torchvision
import torch.nn as nn
class BaseFeatureNet(nn.Module):
    def __init__(self, num_views = 15, base_model_name = "ALEXNET", pretrained = True):
        # 初始化函数，设置基础模型名称和是否使用预训练模型
        super(BaseFeatureNet, self).__init__()
        # 调用父类的初始化函数
        self.base_model = torchvision.models.alexnet(pretrained = pretrained)
        # 加载预训练的alexnet模型
        self.fc_features = None
        self.num_views = num_views
        if base_model_name == "ALEXNET":
            # 如果基础模型名称为ALEXNET
            base_model = torchvision.models.alexnet(pretrained = pretrained)
            # 加载预训练的alexnet模型
            self.feazture_len = 4096
            # 设置特征长度为4096
            self.features = base_model.features
            # 获取模型的特征部分
            # 调整全连接层以适应224x224输入
            self.fc_features = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self._get_conv_output_size(), 4096),  # 224x224输入时，特征图尺寸为6x6
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True)
            )
        def _get_conv_output_size(self):
        # 使用一个虚拟输入来计算特征图的尺寸
            dummy_input = torch.randn(1, 3, 224, 224)  # 假设输入尺寸为224x224
            dummy_output = self.features(dummy_input)
            return dummy_output.view(dummy_output.size(0), -1).size(1)
    def forward(self, x):
        # 获取输入张量的维度
        N,C,H,W = x.size()
        # 将输入张量展平，使其维度为(N*self.num_views, C, H, W)
        x = x.view(-1, self.num_views, C, H, W)
        # 将张量的维度重新排列，使其维度为(N, C, self.num_views, H, W)
        x = x.permute(0,2,1,3,4).contiguous().view(-1, C, H, W)

        # 获取特征
        x = self.features(x)
        # 将张量的维度重新排列，使其维度为(N, 256*6*6)
        x = x.view(x.size(0), 256 * 6 * 6)
        features = self.fc_features(x)
        features = features.view(-1, self.num_views, features.size(-1))
        features = torch.max(features, 1)[0]

        return features
    

class BaseRetrievalNet(nn.module):
    def __init__(self, base_model_name = "ALEXNET", feature_len = 4096):
        super(BaseRetrievalNet, self).__init__()
        base_model_name = base_model.name.upper()
        if base_model_name == "ALEXNET":
            self.feature_len = feature_len
        
        self.fc = nn.Linear(self.feature_len, 1024)
    
    def forward(self, x):
        x.self.fc(x)
        return x

class MV_AlexNet(nn.Module):
    def __init__(self, num_views = 15, pretrained = True):
        # 初始化函数，设置默认参数num_viewsm为15
        super(MV_AlexNet, self).__init__()
        base_model_name = "ALEXNET"

        print(f'\ninit {base_model_name} model...\n')
        
        self.features = BaseFeatureNet(base_model_name, num_views=num_views, pretrained = True)
        self.retrieval = BaseRetrievalNet(base_model_name)

    # 定义前向传播函数
    def forward(self, x):
        # 将输入x传入特征提取层
        x = self.features(x)
        # 将特征提取层的输出传入检索层
        x = self.retrieval(x)
        # 返回检索层的输出
        return x