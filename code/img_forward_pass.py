import os
import torch
from torchvision.models import resnet50, ResNet50_Weights
from coco_dataset import *

'''
Perform the forward encoding pass on all images to generate representations of images
'''

if __name__ == '__main__':

    # Set up CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the model
    model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
    model.to(device)

    # Load the image-only dataset
    dataset = COCODataset_ImageOnly(os.path.join('data', 'coco_data'), train = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1024, shuffle = True, num_workers = 0)
    
    # Testing iteration over an epoch
    for imgs, ids in iter(dataloader):
        print(imgs.size(), ids.size())