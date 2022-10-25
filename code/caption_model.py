import torch

from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class CaptionModel(nn.Module):
    def __init__(self, prefix_len=4, input_shape=2048):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_dim = self.gpt.transformer.wte.weight.shape[1]
        self.prefix_len = prefix_len

        for param in self.gpt.parameters():
            param.requires_grad=False

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.align = nn.Linear(in_features=input_shape, out_features=prefix_len*self.gpt_dim)
    
    def forward(self, img: torch.tensor, tokens: (list, str), mask=None, labels=None):
        
        prefix = self.align.forward(img)
        tokens = self.tokenizer(tokens, return_tensors='pt')
        embed = self.gpt.transformer.wte(tokens['input_ids'])

        prefix = prefix.view(-1, self.prefix_len, self.gpt_dim)
        concat = 
