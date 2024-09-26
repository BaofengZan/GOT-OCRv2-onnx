import os 
import random
import numpy as np 

DEFAULT_RANDOM_SEED = 1024

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
import torch
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
    
seedEverything()

prompt = '蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是'

from transformers import AutoTokenizer
from onnxllm import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/data/llm/llm-export/llama3-onnx", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/data/llm/llm-export/llama3-onnx", trust_remote_code=True)

inputs = tokenizer(prompt, return_tensors='pt')

output = model.chat(tokenizer, prompt)
# pred = model.generate(**inputs)
print(output)
