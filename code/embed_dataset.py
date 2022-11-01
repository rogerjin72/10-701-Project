from enum import unique
import torch
import json
import os
import hyperparams as hp

'''
Expected file structure for MS COCO data:

root
    |- coco_data
        |- coco_annotations
            |- coco_train_annot.json
            |- coco_val_annot.json
    |- encoding_data
        |- img_encodings
            |- encoding_train
                |- [Training image encodings].pt
            |- encoding_val
                |- [Validation image encodings].pt
'''

def ids2fns(ids, extension = '.jpg'):
    '''
    Convert a list of numeric ids to a list of valid filenames
        ids: list(int)
        extension: str
        return: list(str)
    '''
    return [str(id).zfill(hp.DATA_FILENAME_LEN) + extension for id in ids]

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

def collate_labels(ids, labels):
    '''
    Groups labels corresponding to the same id
        ids: List of ids
        labels List of labels
        return: 
            list(int) list of unique ids
            list(list(str)) list of grouped labels
    '''
    collated_labels = []
    last_id = -1
    block = []
    for i in range(len(ids)):
        if ids[i] != last_id:
            last_id = ids[i]
            collated_labels.append(block)
            block = []
        block.append(labels[i])
    return list(set(ids)), collated_labels[1:]

# Dataset of image embedding - text string pairs
class EmbedDataset(torch.utils.data.Dataset):

    def __init__(self, root, train, all_caps = False, collate = False):
        '''
        Reads in raw data from disk to create the dataset
            root (str): directory relative to cwd where raw data is stored
            train (bool): whether or not to load training data
            all_caps (bool): whether to keep all captions or use only one
            collate (bool): whether to group captions corresponding to the same image id
            return: COCODataset
        '''
        self.train = train

        # Get paths
        self.root = root
        self.train_data_dir = os.path.join(root, 'encoding_data', 'img_encodings', 'encoding_train')
        self.train_annot_dir = os.path.join(root, 'coco_data', 'coco_annotations', 'coco_train_annot.json')
        self.val_data_dir = os.path.join(root, 'encoding_data', 'img_encodings', 'encoding_val')
        self.val_annot_dir = os.path.join(root, 'coco_data', 'coco_annotations', 'coco_val_annot.json')

        # Load labels
        with open(self.train_annot_dir) as f:
            self.train_annot = json.load(f)['annotations']
            self.train_annot.sort(key = lambda x: x['image_id'])

        with open(self.val_annot_dir) as f:
            self.val_annot = json.load(f)['annotations']
            self.val_annot.sort(key = lambda x: x['image_id'])

        # Remove duplicate captions
        if not all_caps:
            self.train_annot = remove_duplicate(self.train_annot)
            self.val_annot = remove_duplicate(self.val_annot)

        # Get list of image-caption indicies
        self.train_img_ids = [x['image_id'] for x in self.train_annot]
        self.train_labels = [x['caption'] for x in self.train_annot]
        
        self.val_img_ids = [x['image_id'] for x in self.val_annot]
        self.val_labels = [x['caption'] for x in self.val_annot]

        # Collate ids
        if collate:
            self.train_img_ids, self.train_labels = collate_labels(self.train_img_ids, self.train_labels)
            self.val_img_ids, self.val_labels = collate_labels(self.val_img_ids, self.val_labels)

        # Convert image ids to embedding filenames
        self.train_embed_fns = ids2fns(self.train_img_ids, extension = '.pt')
        self.val_embed_fns = ids2fns(self.val_img_ids, extension = '.pt')

    # TODO: Caption sentence needs to be embedded as a torchtensor
    def __getitem__(self, index):
        '''
        Returns a single image-caption pair. There are multiple captions for each image, so multiple indicies may map to the same image
            index: which sample to return
            return: image (torchtensor), caption (str)
        '''
        if self.train:
            embed = torch.load(os.path.join(self.train_data_dir, self.train_embed_fns[index]))
            caption = self.train_labels[index]
        else:
            embed = torch.load(os.path.join(self.val_data_dir, self.val_embed_fns[index]))
            caption = self.val_labels[index]

        return embed, caption
    
    def __len__(self):
        '''
        Total number of unique image-caption pairs in the dataset
            return: length (int)
        '''
        if self.train:
            return len(self.train_embed_fns)
        else:
            return len(self.val_embed_fns)