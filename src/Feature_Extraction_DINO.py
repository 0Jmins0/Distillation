import torch
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
import timm


def preprocess_image(image_path):
    # 定义预处理步骤
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(image).unsqueeze(0).to(device="cuda")
    return input_image

def get_features(model, path, batch_size = 10):
    # 指定要检查的目录路径
    directory_path_label = path
    # 获取目录下的所有项目
    items = os.listdir(directory_path_label)
    # 过滤出文件夹名称
    labels = [item for item in items if os.path.isdir(os.path.join(directory_path_label, item))]

    # print(labels)
    features = {}
    # 遍历每一个类别
    for label in labels:
        directory_path_number = directory_path_label + '/' + label
        items = os.listdir(directory_path_number)
        numbers = [item for item in items if os.path.isdir(os.path.join(directory_path_number, item))]
        images = []
        # 遍历某个类别里的一组多视图
        for number in tqdm(numbers,desc=f"Processing {label}"):
            directory_path_picture = directory_path_number + '/' + number
            pictures = os.listdir(directory_path_picture)
            pictures_path = [directory_path_picture + '/' + pic for pic in pictures]

            # 对图像进行处理，满足模型输入格式
            input_images = [preprocess_image(picture_path) for picture_path in pictures_path]
            batches = [torch.cat(input_images[i:i + batch_size], dim=0) for i in range(0, len(input_images), batch_size)]
            
        image_features = []
        with torch.no_grad():
            for batch in batches:
                image_features.append(model.image_encoder(batch).cpu().numpy())
                torch.cuda.empty_cache()  # 清理缓存

        image_features = np.concatenate(image_features, axis=0)
        images.append(image_features)
        
        features[label] = images
    return features

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# model = timm.create_model('vit_base_patch8_224.dino', pretrained=True, num_classes=0)
# model.eval()  # 切换到评估模式

# # 加载DINO模型
# model_path = '../models/DINO/dino_deitsmall8_pretrain.pth'
# model.load_state_dict(torch.load(model_path))

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

path = '../data/ModelNet40_180_tmp'

features_DINO = get_features(dinov2_vits14, path)

print(features_DINO)
print(np.array(features_DINO['airplane']).shape)