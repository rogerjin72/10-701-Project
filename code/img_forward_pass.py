import os
import torch
from torchvision.models import resnet50, ResNet50_Weights
from coco_dataset import *

'''
Perform the forward encoding pass on all images to generate representations of images
'''

if __name__ == '__main__':

    train = True

    # Get path to image save location
    img_encoding_dir = os.path.join('data', 'encoding_data', 'img_encodings')
    if train:
        img_encoding_dir = os.path.join(img_encoding_dir, 'encoding_train')
    else:
        img_encoding_dir = os.path.join(img_encoding_dir, 'encoding_val')

    # Set up CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the model
    model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)

    # Delete global pooling and fully connected layers
    resnet50_layers = list(model.children())
    model = torch.nn.Sequential(*resnet50_layers[:-2])
    model.to(device)

    # Load the image-only dataset 
    dataset = COCODataset_ImageOnly(os.path.join('data', 'coco_data'), train = train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = False, num_workers = 0)
    
    # Iterate over batches
    for imgs, ids in iter(dataloader):

        # Load GPU memory
        imgs_gpu = imgs.to(device)
        # Forward pass
        encodings = model(imgs_gpu)

        # Save forward pass results to disk
        ids = torch.flatten(ids).tolist()
        fns = ids2fns(ids, '.pt')
        for i in range(len(fns)):
            torch.save(encodings[i, ...].clone(), os.path.join(img_encoding_dir, fns[i]))

        # Clear GPU memory
        del imgs_gpu
        torch.cuda.empty_cache()