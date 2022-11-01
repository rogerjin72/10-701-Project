import os
import tqdm
import torch
from embed_dataset import EmbedDataset

'''
Concatenate image encodings into a single block for easier transfer
'''

dataset = EmbedDataset('..', train = False)
encoding_block = torch.empty((len(dataset), 2048))
annots = [None] * len(dataset)
print(encoding_block.size())

with tqdm.tqdm(range(len(dataset))) as bar:
    for i in bar:
        enc, annot = dataset[i]
        encoding_block[i, :] = enc
        annots[i] = annot

torch.save({'encoding': encoding_block, 'captions': annots}, os.path.join('data', 'encoding_data', 'val_encoding_block.pt'))