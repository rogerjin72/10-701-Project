import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from coco_dataset import COCODataset_ImageOnly
from img_caption_model import ImageCaptionModel
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

# Load validation data
train = False
dataset = COCODataset_ImageOnly(os.path.join('data', 'coco_data'), train)
dataset = torch.utils.data.Subset(dataset, range(500))
dataloader = DataLoader(dataset, shuffle = False, batch_size = 1)

# Load caption generator
img_cap_model = ImageCaptionModel(os.path.join('models', 'ViT_conv1d_frozen_gpt2', 'model_epoch10.pt'))
img_cap_model.eval()
img_cap_model.attach_hook()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_embeddings = img_cap_model.txt_decoder.gpt.transformer.wte.weight
gpt2_embeddings = gpt2_embeddings.unsqueeze(1)
gpt2_embeddings = gpt2_embeddings.to(torch.device('cpu'))
cosine_sim = nn.CosineSimilarity(dim = 2)

# Saving
prefixes = dict()

# Sample images
for img, id in iter(dataloader):

    # Pass through caption generator
    img = img.squeeze()
    tokens, _ = img_cap_model(img)

    # Compute cosine similarity on prefix
    prefix_embed = img_cap_model.prefix_embed
    prefix_embed = prefix_embed.to(torch.device('cpu'))
    with torch.no_grad():
        sim = cosine_sim(gpt2_embeddings, prefix_embed)
    sim = torch.argmax(sim, dim = 0)

    # Decode caption and prefix
    caption = tokenizer.decode(tokens).split('<|endoftext|>')[0]
    prefix = tokenizer.decode(sim)
    print(repr(prefix))
    print(caption)
    print('=================================================================')

    # Save
    prefixes[id] = repr(prefix)

torch.save(prefixes, 'data/eval_data/prefixes/conv1d.pt')