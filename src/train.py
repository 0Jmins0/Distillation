import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.mvcnn_clip import MVCNN_CLIP
from dataset.dataset import MultiViewDataset
from torchvision import transforms
from utils import TripletLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

train_dataset = MultiViewDataset(root_dir="../data/ModelNet_random_30_final/DS/train",transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = MVCNN_CLIP(num_views=1).to(device)
criterion = TripletLoss(margin = 1.0)
optimizer = optim.Adam(model.parameters(),lr = 0.001)

num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for anchor, positive, negative in train_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        anchor_features = model(anchor)
        positive_features = model(positive)
        negative_features = model(negative)

        loss = criterion(anchor_features, positive_features, negative_features)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "../models/train_models/base/mvcnn_clip_01.pth")