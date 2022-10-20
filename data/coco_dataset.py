import torch
import torchvision
import json
import os
import matplotlib.pyplot as plt

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

class COCODataset(torch.utils.data.Dataset):

    '''
    Reads in raw data from disk to create the dataset
        root (str): directory relative to cwd where raw data is stored
        train (bool): whether or not to load training data
        
        return: COCODataset
    '''
    def __init__(self, root, train):

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

        # Get list of image-caption indicies
        train_img_ids = [x['image_id'] for x in self.train_annot]
        self.train_labels = [x['caption'] for x in self.train_annot]
        
        val_img_ids = [x['image_id'] for x in self.val_annot]
        self.val_labels = [x['caption'] for x in self.val_annot]

        # Convert image ids to image filenames
        self.train_img_fns = [str(x).zfill(12) + '.jpg' for x in train_img_ids]
        self.val_img_fns = [str(x).zfill(12) + '.jpg' for x in val_img_ids]

    '''
    Returns a single sample of data. There are multiple captions for each image, so multiple indicies may map to the same image
        index: which sample to return

        return: image (torchtensor), caption (str)
    '''
    # TODO: Caption sentence needs to be embedded as a torchtensor
    def __getitem__(self, index):
        if self.train:
            img = torchvision.io.read_image(os.path.join(self.train_data_dir, self.train_img_fns[index]))
            caption = self.train_labels[index]
        else:
            img = torchvision.io.read_image(os.path.join(self.val_data_dir, self.val_img_fns[index]))
            caption = self.val_labels[index]
        return img, caption
    
    '''
    Total number of unique image-caption pairs in the dataset
        return: length (int)
    '''
    def __len__(self):
        if self.train:
            return len(self.train_img_fns)
        else:
            return len(self.val_img_fns)

if __name__ == "__main__":

    # Basic test to load image
    dataset = COCODataset('coco_data', train = False)

    img, caption = dataset[100]
    print(caption)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()