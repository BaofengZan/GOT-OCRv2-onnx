from collections import OrderedDict
from onnxllm.models.auto_config import CONFIG_MAPPING_NAMES
from onnxllm.models.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update


MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("qwen", "QwenForCausalLM"),
        ("llama3", "LlamaForCausalLM"),
        ("chatglm", "ChatGLMForCausalLM"),
        ("GOT", "GotForCausalLM"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)

class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")
