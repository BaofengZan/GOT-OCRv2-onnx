import os 
import torch
import random
import numpy as np 
import sys
sys.path.insert(0, r"D:\LearningCodes\GithubRepo\shouxieAI\GOT-OCR2.0\OnnxLLM")


DEFAULT_RANDOM_SEED = 1024

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
    
seedEverything()

from transformers import AutoTokenizer
from onnxllm import AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("/data/llm/llm-export/onnx-standard/", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("/data/llm/llm-export/onnx-standard/", trust_remote_code=True)

# prompt = '蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是'

# inputs = tokenizer(prompt, return_tensors='pt')

# output = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(output[0], skip_special_tokens=True))
import os
import requests
from PIL import Image
from io import BytesIO
import torch
from blip_process import *
from conversation import  conv_templates, SeparatorStyle

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'



def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image



tokenizer = AutoTokenizer.from_pretrained(r"D:\LearningCodes\GithubRepo\shouxieAI\GOT-OCR2.0\GOT-OCR-2.0-master\GOT_weights", trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(r"D:\LearningCodes\GithubRepo\shouxieAI\GOT-OCR2.0\llm-export\onnx", trust_remote_code=True)

use_im_start_end = True

image_token_len = 256
    
image = load_image(r"D:\LearningCodes\GithubRepo\shouxieAI\GOT-OCR2.0\GOT-OCR-2.0-master\2.jpg")
image_processor = BlipImageEvalProcessor(image_size=1024)
image_tensor = image_processor(image)
img_numpy  = image_tensor.cpu()

qs = 'OCR: '
qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
conv_mode = "mpt"
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

inputs = tokenizer(prompt)

print(inputs)

output = model.generate(**inputs, images=img_numpy,  max_new_tokens=100)
print(tokenizer.decode(output, skip_special_tokens=True))