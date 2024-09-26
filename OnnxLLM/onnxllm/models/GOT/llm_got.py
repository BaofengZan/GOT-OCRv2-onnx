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
        #if self.token_len:
        #    return torch.zeros([1, 1, 1, self.seq_len], dtype=torch.float32)
        # 上三角矩阵
        causal_mask = torch.full(( self.seq_len, self.seq_len), fill_value=torch.finfo(torch.float32).min, dtype=torch.float32)
        
        causal_mask = torch.triu(causal_mask, diagonal=1)
        return causal_mask  
        #return (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min
    
    # forward 推理主流程
    def forward(self, input_ids, attention_mask, position_ids, image):

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
            num_patches = self.image_feature.shape[1]
            cur_input_embeds = np.hstack(
                (
                    cur_input_embeds[:,:self.image_start_token_pos+1, ...], 
                    self.image_feature, 
                    cur_input_embeds[:, self.image_start_token_pos + num_patches + 1:, ...]
                )
            )
            #cur_input_embeds = np.transpose(cur_input_embeds, [1,0,2])
        # -------
        presents = []
        #hidden_states = np.transpose(cur_input_embeds, [1, 0, 2]) # 【286，1，1024】--> [1,286,1024]
        hidden_states = cur_input_embeds  # [1,286,1024]
        for i in range(self.block_nums):
            #hidden_states = np.transpose(cur_input_embeds, [1, 0, 2])
            hidden_states = self.blocks[i](hidden_states, attention_mask.astype(np.float32), position_ids)
            #presents.append(kv)
        #hidden_states = np.transpose(cur_input_embeds, [1, 0, 2])
        hidden_states = self.norm(hidden_states)
        logits = self.lm(hidden_states[0])  # lmhead
        logits = torch.argmax(torch.from_numpy(logits[-1, :]), dim=-1)
        self.seq_len += 1
        self.token_len += 1
        return logits
    # generate函数  调用推理函数
    def generate(self, **kwargs):
        input_ids = np.array(kwargs['input_ids'])
        imgs = kwargs['images'].numpy()
        max_new_tokens= kwargs.get('max_new_tokens', 50)
        self.seq_len = input_ids.size
        self.context_len = self.seq_len - 2
        self.token_len = 0
        self.first=True
        #past_key_values = [np.zeros(self.past_kv_shape, dtype=np.float32) for i in range(self.block_nums)]
        token_id = np.copy(input_ids)
        output_ids = input_ids
        while self.token_len < max_new_tokens:
            attention_mask = self.get_attention_mask()[None][None]
            position_ids = self.get_position_ids()
            token_id = self.forward(output_ids, attention_mask.numpy(), position_ids,imgs)
            token_id = np.array([token_id.item()])
            output_ids = np.concatenate([output_ids, token_id])
            self.first=False
            if token_id == self.stop_id():
                break

        return output_ids
    