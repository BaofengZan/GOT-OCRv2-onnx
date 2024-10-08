- 只是为了学习GOT-ocr原理以及onnxLLM这两个库。
- 没有加速效果，master分支是带有kvcache的onnx推理
- 先使用llm-export\llm_export.py导出onnx, 然后将原始hf model文件夹下的几个json文件和qwen.tiktoken 拷贝到生成的onnx文件夹里面，  最后运行OnnxLLM\examples\got.py
- v1分支为第一版本没有使用kvcache的推理
- c++版本：使用mnn-llm 进行推理 https://github.com/BaofengZan/mnn-llm-GOT-OCR2.0

参考链接：

[Ucas-HaoranWei/GOT-OCR2.0: Official code implementation of General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model (github.com)](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)

[wangzhaode/llm-export: llm-export can export llm model to onnx. (github.com)](https://github.com/wangzhaode/llm-export)

[inisis/OnnxLLM: Large Language Model Onnx Inference Framework (github.com)](https://github.com/inisis/OnnxLLM)

