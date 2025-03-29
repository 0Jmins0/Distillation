import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MultiViewDataset(Dataset):
    def __init__(self, root_dir, transform = None, num_views = 15):
        self.root_dir = root_dir
        self.transform = transform
        self.num_views = num_views
        self.instances = self._load_instances()

    def _load_instances(self): # 读取数据集到一个列表，每个元素是（类别名、实例名、实例地址）
        instances = []
        for cls in os.listdir(self.root_dir):
            cls_dir = os.path.join(self.root_dir, cls)
            for instance in os.listdir(cls_dir):
                instance_dir = os.path.join(cls_dir, instance)
                if os.path.isdir(instance_dir): # 检查是否为文件夹
                    instances.append((cls, instance, instance_dir))
        return instances
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx): # 根据索引获取某实例
        cls, instance, instance_dir = self.instances[idx]
        views = os.listdir(instance_dir)

        anchor_views = random.sample(views, self.num_views)
        remaining_views = [v for v in views if v not in anchor_views]

        # 从剩下的视图中随机选择 num_views 个视图作为负样本
        if len(remaining_views) >= self.num_views:
            positive_views = random.sample(remaining_views, self.num_views)
        else:
            # 如果剩下的视图数量不足 num_views，重复选择
            positive_views = random.choices(remaining_views, k=self.num_views)
        
        anchor_paths = [os.path.join(instance_dir, anchor_view) for anchor_view in anchor_views]
        positive_paths = [os.path.join(instance_dir, positive_view) for positive_view in positive_views]

        anchor_images = [Image.open(anchor_path).convert("RGB") for anchor_path in anchor_paths]
        positive_images = [Image.open(positive_path).convert("RGB") for positive_path in positive_paths]

        negative_instance = random.choice(self.instances)
        if random.random() < 0.5:
        # 保证不同类或者同类但不同实例
            while negative_instance[0] == cls :
                negative_instance = random.choice(self.instances)
        else:# 同类但不同实例
            while negative_instance[0] != cls or (negative_instance[0] == cls and negative_instance[1] == instance):
                negative_instance = random.choice(self.instances)
        
        negative_dir = negative_instance[2]

        negative_views = random.sample(os.listdir(negative_dir), self.num_views)
        negative_paths = [os.path.join(negative_dir, negative_view) for negative_view in negative_views]
        negative_images = [Image.open(negative_path).convert("RGB") for negative_path in negative_paths]

        # 应用数据增强
        if self.transform:
            anchor_images = [self.transform(image) for image in anchor_images]
            positive_images = [self.transform(image) for image in positive_images]
            negative_images = [self.transform(image) for image in negative_images]

        # 将图片列表转换为张量
        anchor_images = torch.stack(anchor_images, dim=0)  # (num_views, C, H, W)
        positive_images = torch.stack(positive_images, dim=0) 
        negative_images = torch.stack(negative_images, dim=0)  

        return anchor_images, positive_images, negative_images


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        image_paths = []
        for cls in os.listdir(self.root_dir):
            cls_dir = os.path.join(self.root_dir, cls)
            for instance in os.listdir(cls_dir):
                instance_dir = os.path.join(cls_dir, instance)
                if os.path.isdir(instance_dir):
                    for view in os.listdir(instance_dir):
                        image_path = os.path.join(instance_dir, view)
                        image_paths.append(image_path)
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# train_dataset = MultiViewDataset(root_dir = '../../data/ModelNet_random_30_final/DS/train', transform=transform)
# train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
