from transformers import CLIPProcessor, CLIPModel
import os

# 设置 Hugging Face 镜像源（使用清华镜像）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("模型和处理器已下载完成并缓存。")

