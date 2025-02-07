from PIL import Image
import torch 
from models.mvcnn_clip import MVCNN_CLIP
from dataset.dataset import MultiViewDataset, TestDataset
from utils import retrieve_images, extract_features
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

test_dataset = TestDataset(root_dir="../data/ModelNet_random_30_final/DU/retrieval",transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

query_image = Image.open("../data/ModelNet_random_30_final/DU/retrieval/sofa/sofa_0186/sofa_0186.2.png").convert("RGB")

model = MVCNN_CLIP(num_views=1).to(device)
model.load_state_dict(torch.load("../models/train_models/base/mvcnn_clip_9.pth")['model_state_dict'])
model.eval()

features, image_paths = extract_features(model, test_loader, device)

# query_image = transform(query_image).unsqueeze(0).unsqueeze(1).to(device) 
results = retrieve_images(query_image, features, image_paths, model, top_k=10)

for path, similarity in results:
    print(f"Image: {path}, Similarity: {similarity:.4f}")
