import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
)

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "What are these?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
# raw_image = Image.open(requests.get(image_file, stream=True).raw)
image_file="/Users/albert/project/others/database_milvus/fruit01/apple/f_10_01_0001.jpg"
raw_image = Image.open(image_file)
# breakpoint()
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
