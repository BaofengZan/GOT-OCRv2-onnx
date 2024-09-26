import os
import numpy as np
from onnxllm.models.auto_factory import OnnxLLMForCausalLM


class LlamaForCausalLM(OnnxLLMForCausalLM):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        self.context_len = 0
        self.token_len = 0
        self.past_kv_shape = [2, 1, 0, self.config.num_key_value_heads, self.config.hidden_size// self.config.num_attention_heads]

    def load(self, model_path):
        self.block_nums = self.config.num_hidden_layers
        self.lm = self.load_module(os.path.join(model_path, 'lm.onnx'))
        self.embed = self.load_module(os.path.join(model_path, 'embedding.onnx'))
        self.blocks = []
        for i in range(self.block_nums):
            self.blocks.append(self.load_module(os.path.join(model_path, f'block_{i}.onnx')))

    def get_attention_mask(self):
        if self.token_len:
            return np.zeros((1, 1, 1, self.seq_len), dtype=np.float32)
        return (1 - np.tril(np.ones((1, 1, self.seq_len, self.seq_len), dtype=np.float32))) * np.finfo(np.float32).min

    def get_position_ids(self):
        if self.token_len:
            return np.array([[self.seq_len - 1]], dtype=np.int64)
        return np.array([np.arange(self.seq_len, dtype=np.int64)])

    def stop_id(self):
        return self.tokenizer.eos_token_id

    def build_prompt(self, query):
        if hasattr(self.tokenizer, 'build_prompt'):
            prompt = self.tokenizer.build_prompt(query)
        else:
            prompt = query
        return prompt

    def str_to_ids(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        input_ids = input_ids.numpy()
        return input_ids

    def id_to_str(self, token_id):
        word = self.tokenizer._convert_id_to_token(int(token_id))
        word = self.tokenizer.convert_tokens_to_string([word])
        return word

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        hidden_states = self.embed(input_ids)
        presents = []
        for i in range(self.block_nums):
            hidden_states, kv = self.blocks[i](hidden_states, attention_mask, position_ids, past_key_values[i])
            presents.append(kv)
        logits = self.lm(hidden_states)
        self.seq_len += 1
        self.token_len += 1
        return logits, presents

    def response(self, query, stream = False):
        prompt = self.build_prompt(query)
        input_ids = self.str_to_ids(prompt)
        self.seq_len = input_ids.size
        self.context_len = self.seq_len - 2
        self.token_len = 0
        past_key_values = [np.zeros(self.past_kv_shape, dtype=np.float32) for i in range(self.block_nums)]
        token_id = input_ids
        res = ''
        while self.token_len < self.max_length:
            attention_mask = self.get_attention_mask()
            position_ids = self.get_position_ids()
            token_id, past_key_values = self.forward(token_id, attention_mask, position_ids, past_key_values)
            if token_id == self.stop_id():
                if stream:  print("", end='\n')
                res += '\n'
                break
            word = self.id_to_str(token_id)
            res += word
            if stream: print(word, end="", flush=True)
        return res

    def chat(self, tokenizer, query, history = None):
        self.tokenizer = tokenizer
        return self.response(query), None

    def stream_chat(self, tokenizer, query, history = None):
        self.tokenizer = tokenizer
        return self.response(query, True), None

    def eval(self):
        pass

    def generate(self, **kwargs):
        input_ids = kwargs['input_ids'].numpy()
        self.seq_len = input_ids.size
        self.context_len = self.seq_len - 2
        self.token_len = 0
        past_key_values = [np.zeros(self.past_kv_shape, dtype=np.float32) for i in range(self.block_nums)]
        token_id = input_ids
        res = ''
        while self.token_len < self.max_length:
            attention_mask = self.get_attention_mask()
            position_ids = self.get_position_ids()
            token_id, past_key_values = self.forward(token_id, attention_mask, position_ids, past_key_values)
            if token_id == self.stop_id():
                res += '\n'
                break
            word = self.id_to_str(token_id)
            print("word", word, token_id)
            res += word
        return res
