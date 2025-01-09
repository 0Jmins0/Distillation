# python src/Feature_Extraction.py 

import clip
import torch
from PIL import Image
import os
from tqdm import tqdm
import numpy as np


def get_features(model, preprocess, path):
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
        # 遍历某个类别里的某一个点云
        for number in tqdm(numbers,desc=f"Processing {label}"):
            directory_path_picture = directory_path_number + '/' + number
            pictures = os.listdir(directory_path_picture)
            pictures_path = [directory_path_picture + '/' + pic for pic in pictures]
            image = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in pictures_path]
            image = torch.cat(image, dim=0)
            with torch.no_grad():
                image_features = model.encode_image(image)
            images.append(image_features.cpu().numpy())
        
        features[label] = images
    return features



device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32", device=device)
path = '../data/ModelNet40_180_tmp'

features_CLIP = get_features(model, preprocess, path)

print(features_CLIP)
print(np.array(features_CLIP['airplane']).shape)