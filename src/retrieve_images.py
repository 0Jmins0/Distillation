import numpy as np
from Alexnet import DataLoader

def extract_features(model, dataloader):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feature_vectors = model(images)
            features.append(feature_vectors.cpu().numpy())
    return np.vstack(features)

def retrieve_images(query_features, database_features, top_k=5):
    distances = np.linalg.norm(database_features - query_features, axis=1)
    indices = np.argsort(distances)[:top_k]
    return indices

# 提取数据库特征
database_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
database_features = extract_features(model, database_loader)

# 提取查询图像特征
query_image_path = "path/to/query/image.jpg"
query_image = Image.open(query_image_path).convert("RGB")
query_image = transform(query_image).unsqueeze(0).to(device)
query_features = model(query_image).cpu().numpy()

# 检索最相似的图像
top_k_indices = retrieve_images(query_features, database_features, top_k=5)
print("Top K similar images indices:", top_k_indices)