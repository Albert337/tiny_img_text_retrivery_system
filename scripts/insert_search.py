import torch
from torch import nn
from PIL import Image
import argparse
import os
import yaml
from transformers import CLIPProcessor, CLIPModel
from pymilvus import Collection, connections, db


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def embeding_text(text):
    # clip编码
    text_inputs = processor(text=text, return_tensors="pt")

    with torch.no_grad():    
        text_embeddings = clip_model.get_text_features(**text_inputs)

    return text_embeddings.flatten().detach().numpy().tolist()


def embeding_img(img_data):
    img_embeding = processor(images=img_data, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_embeddings = clip_model.get_image_features(**img_embeding)
    return image_embeddings.flatten().detach().numpy().tolist()

def update_image_vector(data_path, operator):
    idxs, embedings, paths = [], [], []
    total_count = 0
    try:
        for dir_name in os.listdir(data_path):
            sub_dir = os.path.join(data_path, dir_name)
            if not os.path.isdir(sub_dir):
                continue
            for file in os.listdir(sub_dir):
                if not file.endswith('.jpg'):
                    continue
                print(f'{dir_name}/{file} is processing')
                image = Image.open(os.path.join(sub_dir, file))
                embeding = embeding_img(image)

                idxs.append(total_count)
                embedings.append(embeding)
                paths.append(os.path.join(sub_dir, file))
                total_count += 1

                if total_count % 50 == 0:
                    data = [idxs, embedings, paths]
                    operator.insert(data)

                    print(f'success insert {total_count} items')
                    idxs, embedings, paths = [], [], []

            if len(idxs):
                data = [idxs, embedings, paths]
                operator.insert(data)
                print(f'success insert {total_count} items')
    except Exception as e:
        print(e)
    print(f'finish update items: {total_count}')


def search_data(input_embeding):
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=[input_embeding],
        anns_field="embeding_img",
        param=search_params,
        limit=6,
        expr=None,
        output_fields=["path"],
        timeout=None,
        round_decimal=-1
    )
    img_paths = [hit.entity.get("path") for hit in results[0]] 
    return img_paths

def image_search(text,img_input):
    # breakpoint()
    if text == '' and img_input is None:
        return None
    elif text != '':
        input_embeding = embeding_text(text)
    elif img_input is not None:
        input_embeding = embeding_img(img_input)
    else:
        pass
    # results = text_image_vector.search_data(imput_embeding)
    results=search_data(input_embeding)
    with open('result.txt', 'w') as f:
        for path in results:
            f.write(path+'\n')

    return results


if __name__ == '__main__':
    cfg=load_config('./cfg/config.yaml')
    data_dir = cfg['data_dir']
 
    clip_model = CLIPModel.from_pretrained(cfg["model"])
    processor = CLIPProcessor.from_pretrained(cfg["model"])

    conn = connections.connect(host="0.0.0.0", port=19530)
    db.using_database("text_image_db")
    
    collection = Collection("text_image_vector")
    collection.load()

    update_image_vector(data_dir,collection)
