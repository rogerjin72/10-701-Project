import torch
import torchvision
import json
import os
import hyperparams as hp
from torchvision import transforms

'''
Expected file structure for MS COCO data:

root
    |- coco_annotations
        |- coco_train_annot.json
        |- coco_val_annot.json
    |- coco_test
        |- [COCO test images]
    |- coco_train
        |- [COCO train images]
    |- coco_val
        |- [COCO validation images]
'''

def remove_duplicate(annot):
    '''
    Remove multiple captions for the same image. Keeps the first caption for each image. 
        annot: A list of annotations. The annotations must be sorted in ascending order by image id.
        return: list(dict). List of unique annotations
    '''
    last_id = -1
    unique_annot = []
    for i in range(len(annot)):
        if annot[i]['image_id'] != last_id:
            last_id = annot[i]['image_id']
            unique_annot.append(annot[i])
    return unique_annot

def ids2fns(ids, extension = '.jpg'):
    '''
    Convert a list of numeric ids to a list of valid filenames
        ids: list(int)
        extension: str
        return: list(str)
    '''
    return [str(id).zfill(hp.DATA_FILENAME_LEN) + extension for id in ids]

# Standard COCO dataset with image caption pairs
class COCODataset(torch.utils.data.Dataset):

    def __init__(self, root, train):
        '''
        Reads in raw data from disk to create the dataset
            root (str): directory relative to cwd where raw data is stored
            train (bool): whether or not to load training data
            return: COCODataset
        '''
        self.train = train

        # Get paths
        self.root = root
        self.train_data_dir = os.path.join(root, 'coco_train')
        self.train_annot_dir = os.path.join(root, 'coco_annotations', 'coco_train_annot.json')
        self.val_data_dir = os.path.join(root, 'coco_val')
        self.val_annot_dir = os.path.join(root, 'coco_annotations', 'coco_val_annot.json')

        # Load labels
        with open(self.train_annot_dir) as f:
            self.train_annot = json.load(f)['annotations']
            self.train_annot.sort(key = lambda x: x['image_id'])

        with open(self.val_annot_dir) as f:
            self.val_annot = json.load(f)['annotations']
            self.val_annot.sort(key = lambda x: x['image_id'])

        # Remove duplicate captions
        self.train_annot = remove_duplicate(self.train_annot)
        self.val_annot = remove_duplicate(self.val_annot)

        # Get list of image-caption indicies
        self.train_img_ids = [x['image_id'] for x in self.train_annot]
        self.train_labels = [x['caption'] for x in self.train_annot]
        
        self.val_img_ids = [x['image_id'] for x in self.val_annot]
        self.val_labels = [x['caption'] for x in self.val_annot]

        # Convert image ids to image filenames
        self.train_img_fns = ids2fns(self.train_img_ids)
        self.val_img_fns = ids2fns(self.val_img_ids)

    def __getitem__(self, index):
        '''
        Returns a single image-caption pair. There are multiple captions for each image, so multiple indicies may map to the same image
            index: which sample to return
            return: image (torchtensor), caption (str)
        '''
        if self.train:
            img = torchvision.io.read_image(os.path.join(self.train_data_dir, self.train_img_fns[index]))
            caption = self.train_labels[index]
        else:
            img = torchvision.io.read_image(os.path.join(self.val_data_dir, self.val_img_fns[index]))
            caption = self.val_labels[index]

        # Define transforms for cropping and resizing image
        img_size = list(img.size())
        crop = transforms.CenterCrop(size = min(img_size[1:]))
        resize = transforms.Resize(hp.RESNET_50_INPUT_SIZE)
        img = transforms.Compose([crop, resize])(img)

        # If the image is a single channel BW image, stack it three times
        if img_size[0] == 1:
            img = torch.cat((img, img, img), axis = 0)
        
        # Convert image to float
        img = img.type(torch.FloatTensor)

        return img, caption
    
    def __len__(self):
        '''
        Total number of unique image-caption pairs in the dataset
            return: length (int)
        '''
        if self.train:
            return len(self.train_img_fns)
        else:
            return len(self.val_img_fns)

# Modified COCO dataset with images only
class COCODataset_ImageOnly(COCODataset):

    def __init__(self, root, train):
        '''
        Same signature and return as superclass constructor
        '''
        super().__init__(root, train)

        # Delete repeated image ids and filenames
        self.train_img_ids = sorted(list(set(self.train_img_ids)))
        self.train_img_fns = ids2fns(self.train_img_ids)

        self.val_img_ids = sorted(list(set(self.val_img_ids)))
        self.val_img_fns = ids2fns(self.val_img_ids)
    
    def __getitem__(self, index):
        '''
        Returns a single image and its corresponding id. There is a one-to-one relation between indicies and images/ids
            index: which image to return
            return: image (torchtensor), id (int)
        '''
        img, caption = super().__getitem__(index)

        # Return the image id instead of the caption
        if self.train:
            id = self.train_img_ids[index]
        else:
            id = self.val_img_ids[index]
        return img, torch.tensor([id])

# Modified COCO dataset with captions only
class COCODataset_CaptionOnly(COCODataset):

    def __init__(self, root, train):
        '''
        Same signature and return as superclass constructor
        '''
        super().__init__(root, train)
        
    def __getitem__(self, index):
        '''
        Returns a single caption and its corresponding id. There is a one-to-one relation between indicies and captions, but a 
        one-to-many relation between indicies and ids
            index: which caption to return
            return: caption (str), id (int)
        '''
        if self.train:
            caption = self.train_labels[index]
            id = self.train_img_ids[index]
        else:
            caption = self.val_labels[index]
            id = self.val_img_ids[index]
        return caption, torch.tensor([id])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from embed_dataset import EmbedDataset
    from torchvision.models import resnet50, ResNet50_Weights

    # Load the model
    model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)

    # Delete fully connected layer
    resnet50_layers = list(model.children())
    model = torch.nn.Sequential(*resnet50_layers[:-1])
    model.to(hp.DEVICE)
    model.eval()

    # Get some datasets
    dataset_coco = COCODataset(os.path.join('data', 'coco_data'), False)
    dataset_embed = EmbedDataset('data', False)

    # Verify that the embeddings match
    for i in range(10):
        img, id = dataset_coco[i]
        embed, id = dataset_embed[i]
        img = img[None, :].to(hp.DEVICE)
        embed_ref = model(img)
        
        p = embed.squeeze()
        q = embed_ref.squeeze()
        _, q_top = torch.topk(q, k = 5)
        _, p_top = torch.topk(p, k = 5)

        print(p_top)
        print(q_top)
        print()
    
