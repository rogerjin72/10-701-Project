import os
import hyperparams as hp
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from img_caption_model import ImageCaptionModel
from transformers import GPT2Tokenizer
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
Forward pass the image caption model and save results
'''

# Path to save captioned images:
img_save_path = os.path.join('data', 'captions', 'ood_cap')
img_path = os.path.join('data', 'ood_images')

# Load caption generator
img_cap_model = ImageCaptionModel(os.path.join('models', 'ViT_conv2d_frozen_gpt2_allcaps_4x4_large', 'model_epoch10.pt'))
img_cap_model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Sample images
fns = os.listdir(img_path)
text = None
for fn in fns:

    # Load and transform image
    img = torchvision.io.read_image(os.path.join(img_path, fn))
    img_size = list(img.size())
    crop = transforms.CenterCrop(size = min(img_size[1:]))
    resize = transforms.Resize(hp.RESNET_50_INPUT_SIZE)
    img = transforms.Compose([crop, resize])(img)

    # Pass through caption generator
    tokens, _ = img_cap_model(img)
    caption = tokenizer.decode(tokens).split('<|endoftext|>')[0]

    # Plot/show
    if text:
        text.remove()
    img = img.permute(1, 2, 0).to(torch.uint8)
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    text = plt.figtext(0.5, 0.9, caption, fontsize = 16, wrap = True, ha = 'center')
    plt.imshow(img)
    plt.savefig(os.path.join(img_save_path, fn))

