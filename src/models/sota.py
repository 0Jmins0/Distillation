import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import clip

# 加载ResNet50模型
resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Identity()  # 移除ResNet50的分类层，以便输出特征向量

# 如果有自己的权重文件，加载它
model_path = "path_to_your_model/CLIP-KD/ResNet50_model.pt"
resnet50.load_state_dict(torch.load(model_path))
resnet50.eval()

# 加载CLIP的文本处理器
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# 图像预处理
image_path = "path_to_your_image.jpg"
image = Image.open(image_path)
image_input = processor(images=image, return_tensors="pt")

# 文本预处理
text = "A photo of a cat."
text_input = processor(text=text, return_tensors="pt")

# 计算图像特征
with torch.no_grad():
    image_features = resnet50(image_input['pixel_values'])

# 加载CLIP的文本模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
with torch.no_grad():
    text_features = clip_model.get_text_features(input_ids=text_input['input_ids'])

# 计算图像和文本之间的余弦相似度
cosine_similarity = F.cosine_similarity(image_features, text_features)

print(f"Cosine similarity between image and text: {cosine_similarity.item()}")
