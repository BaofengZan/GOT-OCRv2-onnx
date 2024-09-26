import types
import functools
import importlib
from collections import OrderedDict
from transformers import AutoConfig, AutoTokenizer
from onnxllm.backends.onnxruntime.backend import OnnxRunner


class OnnxLLMForCausalLM:
    def __init__(self, model_path, **kwargs):
        self.max_length = 20
        self.config = AutoConfig.from_pretrained(model_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.load(model_path)

    def load_module(self, path):
        return OnnxRunner(path)

    def load(self, model_path):
        raise NotImplementedError

    def get_attention_mask(self):
        raise NotImplementedError

    def get_position_ids(self):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        return cls(model_path, **kwargs)

## Below are codes copied and modified from transformers library

def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


class _BaseAutoModelClass:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        model_class = _get_model_class(config, cls._model_mapping)
        return model_class.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )


class _LazyAutoMapping(OrderedDict):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self):
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "onnxllm.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key, value, exist_ok=False):
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys() and not exist_ok:
                raise ValueError(f"'{key}' is already used by a Transformers model.")

        self._extra_content[key] = value


def auto_class_update(cls, checkpoint_for_example="google-bert/bert-base-cased", head_doc=""):
    # Create a new class with the right name from the base class
    # Now we need to copy and re-register `from_pretrained` as class methods otherwise we can't
    # have a specific docstrings for them.
    from_pretrained = copy_func(_BaseAutoModelClass.from_pretrained)
    cls.from_pretrained = classmethod(from_pretrained)
    return cls


def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    # Special treatment
    key = key.replace("-", "_")

    return key


def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    onnxllm_module = importlib.import_module("onnxllm")

    if module != onnxllm_module:
        try:
            return getattribute_from_module(onnxllm_module, attr)
        except ValueError:
            raise ValueError(f"Could not find {attr} neither in {module} nor in {onnxllm_module}!")
    else:
        raise ValueError(f"Could not find {attr} in {onnxllm_module}!")


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
