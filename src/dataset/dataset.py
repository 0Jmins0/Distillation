import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MultiViewDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
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
        anchor_view = random.choice(views)
        positive_view = random.choice([v for v in views if v != anchor_view])

        anchor_path = os.path.join(instance_dir, anchor_view)
        positive_path = os.path.join(instance_dir, positive_view)

        anchor_image = Image.open(anchor_path).convert("RGB")
        positive_image = Image.open(positive_path).convert("RGB")

        negative_instance = random.choice(self.instances)
        # 保证不同类或者同类但不同实例
        while negative_instance[0] == cls and negative_instance[1] == instance:
            negative_instance = random.choice(self.instances)
        
        negative_dir = negative_instance[2]
        negative_view = random.choice(os.listdir(negative_dir))
        negative_path = os.path.join(negative_dir, negative_view)
        negative_image = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
        return anchor_image, positive_image, negative_image


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