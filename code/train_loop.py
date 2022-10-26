import tqdm
import torch
import time
import hyperparams as hp
from embed_dataset import EmbedDataset
from caption_model import CaptionModel

if __name__ == '__main__':

    # Load datasets
    dataset_train = EmbedDataset('data', True)
    dataset_val = EmbedDataset('data', False)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = hp.BATCH_SIZE, shuffle = True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = hp.BATCH_SIZE, shuffle = False)

    # Model and optimizer
    model = CaptionModel()
    model.to(hp.DEVICE)
    optim = torch.optim.Adam(model.parameters(), hp.LEARN_RATE)

    # Epoch loop
    for epoch in range(hp.NUM_EPOCHS):
        
        train_loss = 0

        # Iterate over train batches
        with tqdm.tqdm(dataloader_val, unit = 'batch') as tepoch:

            tepoch.set_description(f"Epoch {epoch}")
            batch_num = 0
            for embeds, captions in tepoch:
                batch_num +=1

                # Forward
                optim.zero_grad()
                embeds = embeds.to(hp.DEVICE)
                out = model(embeds, captions, use_labels = True)
                loss = out.loss

                # Backward
                loss.backward()
                optim.step()

                # Log statistics
                train_loss += loss.item()
                tepoch.set_postfix(loss = train_loss / batch_num)
                time.sleep(0.05)

                # Clear GPU
                del embeds
                del captions
                torch.cuda.empty_cache()

