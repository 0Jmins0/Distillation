import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# 224 *224 * 3 -> 13 * 13 * 256
class BaseFeatureNet(nn.Module):
    def __init__(self, num_views = 15, base_model_name = "ALEXNET", pretrained = True):
        # 初始化函数，设置基础模型名称和是否使用预训练模型
        super(BaseFeatureNet, self).__init__()
        # 调用父类的初始化函数
        self.base_model = torchvision.models.alexnet(pretrained = pretrained)
        # 加载预训练的alexnet模型
        self.num_views = num_views
        self.features = self.base_model.features
        # 移除最后一个池化层（MaxPool2d）
        self.features = nn.Sequential(*list(self.features.children())[:-1])

    def _get_conv_output_size(self):
    # 使用一个虚拟输入来计算特征图的尺寸
        dummy_input = torch.randn(1, 3, 224, 224)  # 假设输入尺寸为224x224
        dummy_output = self.features(dummy_input)
        return dummy_output.view(dummy_output.size(0), -1).size(1)
    
    def forward(self, x):
        # print(x.size())
        # 获取输入张量的维度
        N,C,H,W = x.size()
        # 将输入张量展平，使其维度为(N*self.num_views, C, H, W)

        x = x.view(-1, self.num_views, C, H, W)
        # 将张量的维度重新排列，使其维度为(N, C, self.num_views, H, W)
        x = x.permute(0,2,1,3,4).contiguous().view(-1, C, H, W)

        # 获取特征
        features = self.features(x) # 13 * 13 * 256
    
        # print("Alex——self", features.shape) # 480, 256, 13, 13

        # 将特征重新组织为 (N, num_views, 256, 13, 13)
        features = features.view(-1, self.num_views, 256, 13, 13)
    
        # print("Alex_view", features.shape) # 32, 256, 13, 13

        # 视角池化：对每个视角的特征进行全局池化
        # 使用最大池化或平均池化聚合视角特征
        features = torch.max(features, dim=1)[0]  # 使用最大池化

        # print("before_return", features.shape) # （32，256，13，13）
        return features
    

class BaseRetrievalNet(nn.Module):
    def __init__(self, base_model_name = "ALEXNET", feature_len = 4096):
        super(BaseRetrievalNet, self).__init__()
        base_model_name = base_model_name.upper()
        if base_model_name == "ALEXNET":
            self.feature_len = feature_len
        
        self.fc1 = nn.Linear(13 * 13 * 256, 1024)
        self.fc2 = nn.Linear(1024, 256)

    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        global_feature = self.fc2(x)
        return global_feature

# 13 * 13 * 256 -> 49 * 256 -> 256
class AlexNet_Adapter(nn.Module):
    def __init__(self,target_size = (7,7),batch_size = 4):
        super(AlexNet_Adapter, self).__init__()
        self.target_sieze = target_size
        if target_size == (16,16):
            self.adapter = nn.Sequential(
                nn.Upsample(size=target_size, mode='bilinear', align_corners=False),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # 保持通道数
            )
            input_size = 16 * 16 * 256
            self.feature_dim = 768
        elif target_size == (7,7):
            self.adapter = nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1)
            input_size = 7 * 7 * 256
            self.feature_dim = 768
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 256) 
        self.fc3 = nn.Linear(256, self.feature_dim)

    def forward(self, x):
        batch_size = x.size(0)
        # print("before_adapter", x.shape) # [4, 256, 13, 13]
        x = self.adapter(x) # 7 * 7 * 256
        # print("after_adapter", x.shape) # [4, 256, 7, 7
        local_feature = x.view(x.size(0), -1, 256)
        # print("local_feature", local_feature.shape) # [4, 49, 256]
        x = local_feature.view(local_feature.size(0), -1)
        # print("local_feature_view", x.shape) # [4, 49 * 256]
        x = F.relu(self.fc1(x))
        # print("after_fc1_relu", x.shape) # [4, 1024]
        global_feature = self.fc2(x)
        # print("global_feature", global_feature.shape) # [4, 256]
        combined_feature = torch.cat((local_feature, global_feature.unsqueeze(1)), dim=1)
        # print("combined_feature", combined_feature.shape) # [4, 50, 256]
        # print("combined_feature_view", combined_feature.view(-1, 256).shape) # [4, 50 * 256]
        combined_feature = F.relu(self.fc3(combined_feature.view(-1, 256)))
        combined_feature = combined_feature.view(local_feature.size(0), -1, self.feature_dim)  # 重塑回 [4, 50, 768]
        # print("after_fc3", combined_feature.shape) # [4, 768]
        # print("after_adapter", combined_feature.shape) # （32， 50， 256）
        return combined_feature 

class MV_AlexNet(nn.Module):
    def __init__(self, num_views = 15, is_dis = False, is_pre = True,target_size = (7,7),batch_size = 4, PretrainedModel_dir = None):
        # 初始化函数，设置默认参数num_viewsm为15
        super(MV_AlexNet, self).__init__()
        base_model_name = "ALEXNET"
        
        print(f'\ninit {base_model_name} model...\n')
        self.is_dis = is_dis
        # self.features = BaseFeatureNet(num_views = num_views, base_model_name = base_model_name,  pretrained = True)
        if is_dis == False:
            self.features = BaseFeatureNet(num_views = num_views, base_model_name = base_model_name,  pretrained = True)
            self.retrieval = BaseRetrievalNet(base_model_name)
        elif is_pre == False:
            self.features = BaseFeatureNet(num_views = num_views, base_model_name = base_model_name,  pretrained = True)
            self.retrieval = AlexNet_Adapter(target_size, batch_size)
        else:
            self.features = self.load_pretrained_model(num_views, PretrainedModel_dir)
            for param in self.features.parameters():
                param.requires_grad = False
            self.retrieval = AlexNet_Adapter(target_size, batch_size)

    def load_pretrained_model(self, num_views, model_path):
        # 加载预训练模型的权重
        print(f"Loading pre-trained model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 初始化 BaseFeatureNet
        features = BaseFeatureNet(num_views = num_views, base_model_name="ALEXNET", pretrained=False)

        # 只加载特征提取层的权重
        features_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith("features"):
                # new_key = k.replace("features.", "")  # 移除前缀
                features_state_dict[k[9:]] = v
        features.load_state_dict(features_state_dict, strict=True)

        print("Pre-trained features loaded successfully.")
        return features

    # 定义前向传播函数
    def forward(self, x):
        # 将输入x传入特征提取层
        x = self.features(x)
        if self.is_dis == False:
            # 将特征提取层的输出传入检索层
            x = self.retrieval(x)
            # 返回检索层的输出
            return x
        else:
            combined_feature = self.retrieval(x)
            return combined_feature

# # 测试代码
# model = MV_AlexNet(num_views=15, pretrained=True, is_dis=True)
# dummy_input = torch.randn(2 * 15, 3, 224, 224)  # 输入形状为 (batch_size, 3, 224, 224)
# local_feature, global_feature = model(dummy_input)

# print("Local feature shape:", local_feature.shape)  # 应为 (2, 49, 256)
# print("Global feature shape:", global_feature.shape)  # 应为 (2, 256)