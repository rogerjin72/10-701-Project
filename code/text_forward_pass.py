import os
import torch
import numpy as np

from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from coco_dataset import COCODataset_CaptionOnly, ids2fns

'''
Perform the forward encoding pass on all images to generate representations of images
'''

if __name__ == '__main__':

    train = False
    base_model = 't5-small'
    seq_len = 256
    
    # Get path to image save location
    text_encoding_dir = os.path.join('data', 'encoding_data', 'text_encodings')
    if train:
        text_encoding_dir = os.path.join(text_encoding_dir, 'encoding_train')
    else:
        text_encoding_dir = os.path.join(text_encoding_dir, 'encoding_val')

    # Set up CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the model
    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=seq_len)
    model = T5EncoderModel.from_pretrained(base_model)
    model.to(device)

    # Load the image-only dataset 
    dataset = COCODataset_CaptionOnly(os.path.join('data', 'coco_data'), train = train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 25, shuffle = False, num_workers = 0)

    # Iterate over batches
    for sentences, ids in tqdm(iter(dataloader)):
        # Forward pass
        tokenized = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True)
        embeddings = model(**tokenized)['last_hidden_state'].detach()
        print(embeddings.shape)
        # Save forward pass results to disk
        ids = torch.flatten(ids).tolist()
        fns = np.array(ids2fns(ids, '.pt'))
        for im in np.unique(fns):
            flag = fns == im
            torch.save(embeddings[flag], os.path.join(text_encoding_dir, im))

        # Clear GPU memory
        torch.cuda.empty_cache()
