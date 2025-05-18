import torch
import os
from transformers import CLIPVisionModel, CLIPProcessor
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像
# 定义保存路径
model_dir = "/home/xyzhang/project/Distillation/models/pretrained/openai/clip-vit-large-patch14"
os.makedirs(model_dir, exist_ok=True)

# 预加载模型并保存到本地
model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPVisionModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# 保存模型和处理器
clip_model.save_pretrained(model_dir)
clip_processor.save_pretrained(model_dir)