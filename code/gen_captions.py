import os
import torch
import matplotlib.pyplot as plt
from coco_dataset import COCODataset_ImageOnly
from img_caption_model import ImageCaptionModel
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from textwrap import wrap
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
Forward pass the image caption model and save results
'''

train = False
num_imgs = 5000

# Path to save captioned images:
if train:
    img_path = os.path.join('data', 'captions', 'train_cap')
else:
    img_path = os.path.join('data', 'captions', 'val_cap')

# Load validation data
dataset = COCODataset_ImageOnly(os.path.join('data', 'coco_data'), train)
dataloader = DataLoader(dataset, shuffle = False)

# Load caption generator
img_cap_model = ImageCaptionModel(os.path.join('models', 'ViT_conv2d_frozen_gpt2_allcaps_4x4_large', 'model_epoch10.pt'))
img_cap_model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Sample images
i = 0
for img, id in iter(dataloader):
    i += 1
    if i > num_imgs:
        break

    # Pass through caption generator
    img = img.squeeze()
    tokens, _ = img_cap_model(img)
    caption = tokenizer.decode(tokens).split('<|endoftext|>')[0]

    # Plot/show
    id = id.squeeze().item()
    img = img.permute(1, 2, 0).to(torch.uint8)
    plt.figure()
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.figtext(0.5, 0.9, '\n'.join(wrap(caption, 30)), fontsize = 16, ha = 'center')
    plt.imshow(img)
    plt.savefig(os.path.join(img_path, '{0}.jpg'.format(id)))
    plt.close()

