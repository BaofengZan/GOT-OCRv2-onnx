#coding=utf-8 
'''
尝试实现got的推理
先大概理清楚逻辑，再实现。
'''

import os
import numpy as np
from tqdm import tqdm
import torch
from onnxllm.models.qwen.llm_qwen import QwenForCausalLM



class GotForCausalLM(QwenForCausalLM):
    '''
    got的推理
    '''
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        self.im_patch_token = 151859
        self.im_start_token = 151857
        self.im_end_token = 151858
        # 用于流式输出
        self.token_cache = []
        self.print_len = 0
    
    # load函数加载onnx
    def load(self, model_path):
        self.block_nums = self.config.num_hidden_layers
        total_modules = self.config.num_hidden_layers + 2

        self.blocks = []

        with tqdm(total=total_modules, desc="Loading model shards") as pbar:
            self.lm = self.load_module(os.path.join(model_path, 'lm.onnx'))
            pbar.update(1)

            self.embed = self.load_module(os.path.join(model_path, 'embedding.onnx'))
            pbar.update(1)

            self.visual = self.load_module(os.path.join(model_path, 'visual.onnx'))
            pbar.update(1)

            self.mm_projector_vary = self.load_module(os.path.join(model_path, 'mm_projector_vary.onnx'))
            pbar.update(1)
        
            self.norm = self.load_module(os.path.join(model_path, 'norm.onnx'))
            pbar.update(1)
            for i in range(self.block_nums):
                block_path = os.path.join(model_path, f'block_{i}.onnx')
                self.blocks.append(self.load_module(block_path))
                pbar.update(1)

    def get_attention_mask(self):
        if self.token_len:
            return torch.zeros([1, self.seq_len], dtype=torch.float32)
        # 上三角矩阵
        causal_mask = torch.full(( self.seq_len, self.seq_len), fill_value=torch.finfo(torch.float32).min, dtype=torch.float32)
        
        causal_mask = torch.triu(causal_mask, diagonal=1)
        return causal_mask  
        #return (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min
    
    # forward 推理主流程
    def forward(self, input_ids, attention_mask, position_ids, past_key_caches, past_value_caches, image):

        # 1 先计算visual
        if self.first: # input_ids.shape[0] != 1:
            cnn_feature = self.visual(image[None,...])
            cnn_feature = torch.from_numpy(cnn_feature).flatten(2).permute(0, 2, 1).numpy() # 256*1024
            image_feature = self.mm_projector_vary(cnn_feature)
            
            inputs_embeds = self.embed(input_ids.astype(np.int64))
            #cur_input_embeds = np.transpose(inputs_embeds, [1,0,2])
            cur_input_embeds = inputs_embeds
            image_start_token_pos = np.where(input_ids == self.im_start_token)[0][0]
            num_patches = image_feature.shape[1]
            cur_input_embeds = np.hstack(
                (
                    cur_input_embeds[:,:image_start_token_pos+1, ...], 
                    image_feature, 
                    cur_input_embeds[:, image_start_token_pos + num_patches + 1:, ...]
                )
            )
            self.image_feature=image_feature
            self.image_start_token_pos = image_start_token_pos
        else:
            cur_input_embeds = self.embed(input_ids.astype(np.int64))
            # num_patches = self.image_feature.shape[1]
            # cur_input_embeds = np.hstack(
            #     (
            #         cur_input_embeds[:,:self.image_start_token_pos+1, ...], 
            #         self.image_feature, 
            #         cur_input_embeds[:, self.image_start_token_pos + num_patches + 1:, ...]
            #     )
            # )
            #cur_input_embeds = np.transpose(cur_input_embeds, [1,0,2])
        # -------
        presents_key = []
        presents_value = []
        #hidden_states = np.transpose(cur_input_embeds, [1, 0, 2]) # 【286，1，1024】--> [1,286,1024]
        hidden_states = cur_input_embeds  # [1,286,1024]
        for i in range(self.block_nums):
            #hidden_states = np.transpose(cur_input_embeds, [1, 0, 2])
            hidden_states, key_states, value_states = self.blocks[i](hidden_states, attention_mask.astype(np.float32), position_ids,past_key_caches[i], past_value_caches[i])
            presents_key.append(key_states)
            presents_value.append(value_states)
        #hidden_states = np.transpose(cur_input_embeds, [1, 0, 2])
        hidden_states = self.norm(hidden_states)
        logits = self.lm(hidden_states[0])  # lmhead
        logits = torch.argmax(torch.from_numpy(logits[-1, :]), dim=-1)
        self.seq_len += 1
        self.token_len += 1
        return logits,presents_key,presents_value
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False
    # generate函数  调用推理函数
    def generate(self, **kwargs):
        tokenizer = kwargs["tokenizer"]
        input_ids = np.array(kwargs['input_ids'])
        imgs = kwargs['images'].numpy()
        max_new_tokens= kwargs.get('max_new_tokens', 50)
        self.seq_len = input_ids.size
        self.context_len = self.seq_len - 2
        self.token_len = 0
        self.first=True
        #past_key_values = [np.zeros(self.past_kv_shape, dtype=np.float32) for i in range(self.block_nums)]
        past_key_caches = [np.zeros((1,16,0,64), dtype=np.float32) for i in range(self.block_nums)]
        past_value_caches = [np.zeros((1,16,0,64), dtype=np.float32) for i in range(self.block_nums)]
        token_id = np.copy(input_ids)
        output_ids = input_ids
        while self.token_len < max_new_tokens:
            attention_mask = self.get_attention_mask()[None][None]
            position_ids = self.get_position_ids()
            token_id, past_key_caches, past_value_caches = self.forward(token_id, attention_mask.numpy(), position_ids, past_key_caches, past_value_caches,  imgs)
            token_id = np.array([token_id.item()])
            output_ids = np.concatenate([output_ids, token_id])
            self.first=False
            #print(f"token-id--> {token_id} --> {tokenizer.decode(token_id, skip_special_tokens=True, errors='ignore')}")

            self.token_cache.extend(token_id.tolist())
            text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
             # After the symbol for a new line, we flush the cache.
            if text.endswith("\n"):
                printable_text = text[self.print_len :]
                self.token_cache = []
                self.print_len = 0
            # If the last token is a CJK character, we print the characters.
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len :]
                self.print_len += len(printable_text)
            # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
            # which may change with the subsequent token -- there are probably smarter ways to do this!)
            else:
                printable_text = text[self.print_len:text.rfind(' ')+1]
                self.print_len += len(printable_text)
            print(printable_text, end="", flush=True)
            #print(tokenizer.decode(token_id, skip_special_tokens=True, errors='ignore'), flush=True, end="")
            if token_id == self.stop_id():
                break
        if self.print_len==0:
            print(tokenizer.decode(output_ids[input_ids.size:], skip_special_tokens=True, errors='ignore'), flush=True, end="")
        return output_ids
    