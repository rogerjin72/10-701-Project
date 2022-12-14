import torch
import numpy as np

from torch import nn
import hyperparams as hp


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
        pos_embedding = pos_embedding.to(hp.DEVICE)

        return X + pos_embedding


class EncoderConv1D(nn.Module):
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
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
            
        X = self.pos_embedding.forward(X)
        X = self.encoder.forward(X)
        X = self.combine(X)
        return X

class EncoderConv2D(nn.Module):
    """
    Implement Transformer encoder block for prefix generation
    """
    def __init__(self, embedding_dim: int, heads: int, layers: int):
        """
        Parameters
        ----------
        embedding_dim : 
            dimension of text and image embeddings
        heads : 
            number of heads to use for multihead attention
        layers:
            number of blocks in the transformer

        Returns
        -------
        None
        """
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels = embedding_dim, out_channels = embedding_dim, kernel_size = 6)
        if hp.GPT == 'gpt2':
            self.conv2 = nn.Conv2d(in_channels = embedding_dim, out_channels = 768, kernel_size = 6)
        elif hp.GPT == 'gpt2-medium':
            self.conv2 = nn.Conv2d(in_channels = embedding_dim, out_channels = 1024, kernel_size = 6)
        elif hp.GPT == 'gpt2-large':
            self.conv2 = nn.Conv2d(in_channels = embedding_dim, out_channels = 1280, kernel_size = 6)
        else:
            raise(ValueError('{0} is an invalid GPT-2 size parameter').format(hp.GPT))
        self.tanh = nn.Tanh()

        # Transformer layers
        self.pos_embedding = PositionalEncodingLayer(embedding_dim)
        block = nn.TransformerEncoderLayer(embedding_dim, heads)
        self.encoder = nn.TransformerEncoder(block, layers)
    
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
        # Reshape if single instance
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])

        # Transformer forward pass
        X = self.pos_embedding.forward(X)
        X = self.encoder.forward(X)

        # Separate the class embedding
        X_img = X[:, 1:, :]

        # Reshape to 4D tensor
        X_img = X_img.reshape(X_img.shape[0], hp.VIT_DIM, hp.VIT_DIM, X_img.shape[-1])
        X_img = torch.permute(X_img, (0, 3, 1, 2))

        # Convolutional layers
        X_img = self.conv1(X_img)
        X_img = self.tanh(X_img)
        X_img = self.conv2(X_img)

        # Flatten back to 3D tensor
        X_img = torch.permute(X_img, (0, 2, 3, 1))
        X_img = X_img.reshape(X_img.shape[0], X_img.shape[1] * X_img.shape[2], X_img.shape[-1])
        
        return X_img