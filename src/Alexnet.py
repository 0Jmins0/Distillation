class MultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.instances = {}  # 用于存储实例标签
        self.data = []

        # 遍历每个类别和实例
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            for instance_name in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_name)
                if os.path.isdir(instance_path):
                    instance_images = [os.path.join(instance_path, img) for img in os.listdir(instance_path)]
                    instance_label = len(self.instances)  # 为每个实例分配一个唯一标签
                    self.instances[instance_name] = instance_label
                    self.data.extend([(img_path, instance_label) for img_path in instance_images])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, instance_label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, instance_label
class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.instance_to_indices = {}
        for idx, (_, instance_label) in enumerate(self.dataset.data):
            if instance_label not in self.instance_to_indices:
                self.instance_to_indices[instance_label] = []
            self.instance_to_indices[instance_label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor_image, anchor_instance = self.dataset[idx]
        positive_idx = random.choice(self.instance_to_indices[anchor_instance])
        positive_image, _ = self.dataset[positive_idx]

        # 随机选择一个负样本（不同实例）
        negative_instance = random.choice(list(self.instance_to_indices.keys()))
        while negative_instance == anchor_instance:
            negative_instance = random.choice(list(self.instance_to_indices.keys()))
        negative_idx = random.choice(self.instance_to_indices[negative_instance])
        negative_image, _ = self.dataset[negative_idx]

        return anchor_image, positive_image, negative_image

# 加载预训练的 AlexNet 模型并修改最后的全连接层
model = models.alexnet(pretrained=True)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 128)  # 输出 128 维特征向量
model = model.to(device)

# 定义三元组损失函数
criterion = TripletLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
dataset = MultiViewDataset(root_dir="path/to/your/dataset", transform=transform)
triplet_dataset = TripletDataset(dataset)
train_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True, num_workers=4)

# 训练模型
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            # 计算损失
            loss = criterion(anchor_out, positive_out, negative_out)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * anchor.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 开始训练
train_model(model, train_loader, criterion, optimizer, num_epochs=10)