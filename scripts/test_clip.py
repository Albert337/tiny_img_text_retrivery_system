from PIL import Image
import matplotlib.pyplot as plt
import requests
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# plt.imshow(image)
# plt.axis('off')
# plt.show()
# breakpoint()

# # 对图像进行预处理
# inputs = processor(images=image, return_tensors="pt")

# # 获取图像编码
# with torch.no_grad():
#     image_embeddings = model.get_image_features(**inputs)


# txt_input = "a photo of a cat"
# text_inputs = processor(text=txt_input, return_tensors="pt")

# with torch.no_grad():    
#     text_embeddings = model.get_text_features(**text_inputs)
#     print(text_embeddings)


# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
# print(probs)



####test with multiple images and multiple texts
# all_images = [Image.open(i.strip()) for i in open("result.txt","r").readlines()]
# inputs = processor(text=["a photo of a apple", "a photo of a banana"],images=all_images, return_tensors="pt", padding=True)
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
# print(probs)


###test images match

def embeding_img(img_data):
    img_embeding = processor(images=img_data, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_embeddings = model.get_image_features(**img_embeding)
    return image_embeddings.flatten().detach().numpy().tolist()

def cal_consine_distance(img_data1, img_data2):
    img_embeding1 = embeding_img(img_data1)
    img_embeding2 = embeding_img(img_data2)
    return 1-float(np.dot(img_embeding1, img_embeding2)/(np.linalg.norm(img_embeding1)*np.linalg.norm(img_embeding2)))

all_images = [Image.open(i.strip()) for i in open("result.txt","r").readlines()]
src_image = Image.open("/Users/albert/project/others/database_milvus/fruit01/f_05_04_0001.jpg")
all_images.append(src_image)
dists=[cal_consine_distance(src_image, i) for i in all_images]
        
print(dists)