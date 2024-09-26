from collections import OrderedDict
from transformers import AutoConfig,Qwen2Config

class GOTConfig(Qwen2Config):
    model_type = "GOT"
    
CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("qwen", "QWenConfig"),
        ("llama3", "LlamaConfig"),
        ("chatglm", "ChatGLMConfig"),
        ("GOT", "GOTConfig"),
    ]
)


AutoConfig.register("GOT", GOTConfig)