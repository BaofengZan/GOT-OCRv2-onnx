# OnnxLLM

<p align="center">
    <a href="https://pypi.org/project/onnxllm">
        <img src="https://badgen.net/pypi/v/onnxllm?color=blue" />
    </a>
</p>

This is an example project for learning purpose

# Installation

## Using Prebuilt

```bash
pip install onnxllm
```

## Install From Source

```bash
pip install git+https://github.com/inisis/OnnxLLM@main
```

## Install From Local

```bash
git clone https://github.com/inisis/OnnxLLM && cd OnnxLLM/
pip install .
```

# How to use

```
from transformers import AutoTokenizer
from onnxllm import AutoModelForCausalLM

# you should download onnx models from https://huggingface.co/inisis-me first
tokenizer = AutoTokenizer.from_pretrained("/data/llm/llm-export/onnx-standard/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/data/llm/llm-export/onnx-standard/", trust_remote_code=True)

prompt = '蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是'

inputs = tokenizer(prompt, return_tensors='pt')

output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

# References

> - [mnn-llm](https://github.com/wangzhaode/mnn-llm)
> - [onnxruntime](https://github.com/microsoft/onnxruntime)
> - [huggingface](https://github.com/huggingface)


<details>
  <summary>referenced models</summary>

- [chatglm-6b](https://modelscope.cn/models/ZhipuAI/chatglm-6b/summary)
- [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary)
- [chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary)
- [codegeex2-6b](https://modelscope.cn/models/ZhipuAI/codegeex2-6b/summary)
- [Baichuan2-7B-Chat](https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary)
- [Qwen-7B-Chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary)
- [Qwen-VL-Chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary)
- [Qwen-1.8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary)
- [Llama-2-7b-chat-ms](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary)
- [internlm-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b/summary)
- [phi-2](https://modelscope.cn/models/AI-ModelScope/phi-2/summary)
- [bge-large-zh](https://modelscope.cn/models/AI-ModelScope/bge-large-zh/summary)
- [TinyLlama-1.1B-Chat-v0.6](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.6)
- [Yi-6B-Chat](https://modelscope.cn/models/01ai/Yi-6B-Chat/summary)
- [Qwen1.5-0.5B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary)
- [Qwen1.5-1.8B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat/summary)
- [Qwen1.5-4B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat/summary)
- [Qwen1.5-7B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat/summary)

</details>