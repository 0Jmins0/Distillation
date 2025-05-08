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
                instance_dir = os.path.join(instance_dir, "image")
                if os.path.isdir(instance_dir): # 检查是否为文件夹
                    instances.append((cls, instance, instance_dir))
        return instances
    
    def __len__(self):
        return len(self.instances)
    
    def getviews(self, idx): # 根据索引获取某实例的视图
        cls, instance, instance_dir = self.instances[idx]
        views = os.listdir(instance_dir)
        selet_views = []
        step = len(views) // self.num_views
        for idx in range(0, len(views), step):
            # print("idx",idx)
            selet_views.append(views[idx])
        # print("selet_views",selet_views.sort)
        return selet_views

    
    def __getitem__(self, anchor_idx): # 根据索引获取某实例
        cls, instance, instance_dir = self.instances[anchor_idx]
        # views = os.listdir(instance_dir)
        # anchor_views = []
        # step = len(views) // self.num_views
        # for idx in range(len(views), step):
        positive_idx = random.randint(0, len(self.instances) - 1)
        while self.instances[positive_idx][0] != cls or (self.instances[positive_idx][0] == cls and self.instances[positive_idx][1] == instance):
            positive_idx = random.randint(0, len(self.instances) - 1)

        negative_idx = random.randint(0, len(self.instances) - 1)
        while self.instances[negative_idx][0] == cls :
            negative_idx = random.randint(0, len(self.instances) - 1)        

        anchor_views = self.getviews(anchor_idx)
        anchor_instance = self.instances[anchor_idx]

        # print("anchor",anchor_instance[0],anchor_instance[1],anchor_instance[2])

        positive_views = self.getviews(positive_idx)
        positive_instance = self.instances[positive_idx]

        negative_views = self.getviews(negative_idx)
        negative_instance = self.instances[negative_idx]
        
        anchor_paths = [os.path.join(instance_dir, anchor_view) for anchor_view in anchor_views]
        anchor_images = [Image.open(anchor_path).convert("RGB") for anchor_path in anchor_paths]        

        positive_dir = positive_instance[2]
        positive_paths = [os.path.join(positive_dir, positive_view) for positive_view in positive_views]
        positive_images = [Image.open(positive_path).convert("RGB") for positive_path in positive_paths]
        
        negative_dir = negative_instance[2]
        negative_paths = [os.path.join(negative_dir, negative_view) for negative_view in negative_views]
        negative_images = [Image.open(negative_path).convert("RGB") for negative_path in negative_paths]

        # 应用数据增强
        if self.transform:
            anchor_images = [self.transform(image) for image in anchor_images]
            positive_images = [self.transform(image) for image in positive_images]
            negative_images = [self.transform(image) for image in negative_images]

        # 将图片列表转换为张量
        # print("anchor",anchor_images[0].shape) # anchor torch.Size([3, 224, 224])
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

class TestMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_views=15, data = "target"):
        self.root_dir = root_dir
        self.transform = transform
        self.num_views = num_views
        self.data = data
        self.instances = self._load_instances()
        

    def _load_instances(self): # 读取数据集到一个列表，每个元素是（类别名、实例名、实例地址）
        data_dir = os.path.join(self.root_dir, self.data)
        dict_dir = os.path.join(self.root_dir, f"{self.data}_label.txt")

        # 初始化一个空字典
        data_dict = {}
        # 打开文件并读取内容
        with open(dict_dir, "r") as file:
            for line in file:
                # 去掉每行的换行符并以逗号分隔
                key, value = line.strip().split(",")
                # 将键值对存入字典
                data_dict[key] = value

        instances = []
        for instance in os.listdir(data_dir):
            instance_dir = os.path.join(data_dir, instance)
            instance_dir = os.path.join(instance_dir, "image")
            cls = data_dict[instance]
            if os.path.isdir(instance_dir): # 检查是否为文件夹
                instances.append((cls, instance, instance_dir))
        return instances


    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):  # 根据索引获取某实例
        cls, instance, instance_dir = self.instances[idx]
        views = os.listdir(instance_dir)

        test_views = []
        step = len(views) // self.num_views
        for idx in range(0, len(views), step):
            # print("idx",idx)
            test_views.append(views[idx])
        
        # 随机选择 num_views 个视图作为测试样本
        # test_views = random.sample(views, self.num_views)
        test_paths = [os.path.join(instance_dir, test_view) for test_view in test_views]

        # 加载并处理每个视图
        test_images = []
        for test_path in test_paths:
            image = Image.open(test_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            test_images.append(image)

        # 将图片列表转换为张量
        test_images = torch.stack(test_images, dim=0)  # (num_views, C, H, W) 
        # print("test",test_images.shape) # test torch.Size([15, 3, 224, 224])
        return test_images, test_paths
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# train_dataset = MultiViewDataset(root_dir = '../../data/ModelNet_random_30_final/DS/train', transform=transform)
# train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
