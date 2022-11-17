import torch
import numpy as np

from torch import nn


class PositionalEncodingLayer(nn.Module):
    """
    Implement positional encoding
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, X: torch.Tensor):
        """
        Parameters
        ----------
        X :
            input
        
        Returns
        -------
        torch.Tensor : 
            input with positional embeddings
        """
        _, seq_len, _ = X.shape
        pos_embedding = torch.zeros((seq_len, self.embedding_dim))
        
        # get positional vectors
        pos = torch.arange(seq_len).tile(self.embedding_dim // 2)
        pos = pos.reshape(self.embedding_dim // 2, seq_len).T

        # get location in embedding dimension
        denom = torch.arange(0, self.embedding_dim, 2).tile(seq_len)
        denom = denom.reshape(seq_len, self.embedding_dim // 2) / self.embedding_dim

        sin_term = torch.sin(pos / 10000 ** denom)
        cos_term = torch.cos(pos / 10000 ** denom)

        # fill positional embedding tensor
        pos_embedding[:, np.arange(0, self.embedding_dim, 2)] = sin_term
        pos_embedding[:, np.arange(0, self.embedding_dim, 2) + 1] = cos_term

        return X + pos_embedding


class Encoder(nn.Module):
    """
    Implement Transformer encoder block for prefix generation
    """
    def __init__(self, embedding_dim: int, heads: int, layers: int, inp_seq: int, out_seq: int):
        """
        Parameters
        ----------
        embedding_dim : 
            dimension of text and image embeddings
        heads : 
            number of heads to use for multihead attention
        inp_seq : 
            sequence length of input (197 for ViT)
        out_seq : 
            sequence length of the prefix
        
        Returns
        -------
        None
        """
        super().__init__()
        self.pos_embedding = PositionalEncodingLayer(embedding_dim)
        block = nn.TransformerEncoderLayer(embedding_dim, heads)
        self.encoder = nn.TransformerEncoder(block, layers)
        # linear combination of final embeddings
        self.combine = nn.Conv1d(in_channels=inp_seq, out_channels=out_seq, kernel_size=1)
    
    def forward(self, X: torch.Tensor):
        """
        Parameters
        ----------
        X : 
            input

        Returns
        -------
        torch.Tensor :
            mapped prefix
        """
        # reshape if single instance
        if len(input.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
            
        X = self.pos_embedding.forward(X)
        X = self.encoder.forward(X)
        X = self.combine(X)
        return X