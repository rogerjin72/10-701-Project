import torch
import hyperparams as hp
from torch import nn
from mlp import MLP
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class CaptionModel(nn.Module):
    def __init__(self, prefix_len=4, input_shape=2048, align_layer=None):
        """
        Parameters
        ----------
        prefix_len : int, optional
            length of prefix token, default 4
        input_shape : int, optional
            length of input, default 2048
        align_layer : torch.nn.Module, optional
            model used to connect img and text representations, linear layer
            if no value is passed, default None
        """
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt.to(hp.DEVICE)
        self.gpt_dim = self.gpt.transformer.wte.weight.shape[1]
        self.prefix_len = prefix_len

        # Freeze the GPT2 weights
        # for param in self.gpt.parameters():
        #     param.requires_grad=False

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if align_layer is not None:
            self.align = align_layer
        else:   
            in_features = input_shape
            out_features = prefix_len * self.gpt_dim
            self.align = MLP(in_features, out_features, hp.N_HIDDEN)
    

    def generate_prefix(self, img: torch.Tensor):
        """
        Parameters
        ----------
        img: torch.tensor
            tensor of images

        Returns
        -------
        torch.tensor
            prefix tensor with dimensions (batch_size, prefix_length, GPT embed size)
        """
        prefix = self.align.forward(img)
        prefix = prefix.view(-1, self.prefix_len, self.gpt_dim)
        return prefix

    def forward(self, img: torch.tensor, tokens, use_labels=False):
        """
        Parameters
        ----------
        img : torch.tensor
            tensor of images
        tokens : {array-like, string}
            list of input strings or input string
        use_labels : bool, optional
            returns a loss if true, returns only logits otherwise, default False
        
        Returns
        -------
        transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
            outputs of the GPT2 model, includes loss if use_labels is true
        """
        prefix = generate_prefix(img)

        # tokenize input
        tokens = self.tokenizer(tokens, return_tensors='pt', padding=True)
        tokens = tokens.to(hp.DEVICE)

        # generate embeddings
        embed = self.gpt.transformer.wte(tokens['input_ids'])
        embed = embed.to(hp.DEVICE)

        # concatenate prefix and embeddigns
        inp_embed = torch.cat([prefix, embed], dim=1)
        
        mask = tokens['attention_mask']
        labels = None

        # create labels
        if use_labels:
            # pad attention
            mask_pad = torch.ones((tokens['input_ids'].shape[0], self.prefix_len), device = hp.DEVICE)
            mask_pad = mask_pad.long()
            mask = torch.cat([mask_pad, mask], axis=1)

            # pad labels
            label_pad = torch.zeros((tokens['input_ids'].shape[0], self.prefix_len), device = hp.DEVICE)
            label_pad = label_pad - 100
            label_pad = label_pad.long()
            labels = torch.cat([dummy, tokens['input_ids']], axis=1)
        
        output = self.gpt(inputs_embeds=inp_embed, labels=labels, attention_mask=mask)
        return output
