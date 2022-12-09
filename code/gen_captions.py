import os
import torch
import matplotlib.pyplot as plt
from coco_dataset import COCODataset_ImageOnly
from img_caption_model import ImageCaptionModel
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

'''
Forward pass the image caption model and save results
'''

train = False
num_imgs = 100

# Path to save captioned images:
if train:
    img_path = os.path.join('data', 'eval_data', 'train')
else:
    img_path = os.path.join('data', 'eval_data', 'val')

# Load validation data
dataset = COCODataset_ImageOnly(os.path.join('data', 'coco_data'), train)
dataloader = DataLoader(dataset, shuffle = True)

# Load caption generator
img_cap_model = ImageCaptionModel(os.path.join('models', 'ViT_conv2d_frozen_gpt2_allcaps_8x8', 'model_epoch5.pt'))
img_cap_model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Sample images
i = 0
text = None
for img, id in iter(dataloader):
    i += 1
    if i > num_imgs:
        break

    # Pass through caption generator
    img = img.squeeze()
    tokens, _ = img_cap_model(img)
    caption = tokenizer.decode(tokens).split('<|endoftext|>')[0]

    # Plot/show
    if text:
        text.remove()
    id = id.squeeze().item()
    img = img.permute(1, 2, 0).to(torch.uint8)
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    text = plt.figtext(0.5, 0.9, caption, fontsize = 16, wrap = True, ha = 'center')
    plt.imshow(img)
    plt.savefig(os.path.join(img_path, '{0}.jpg'.format(id)))

