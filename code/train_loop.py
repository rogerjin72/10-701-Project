import tqdm
import torch
import time
import copy
import os
import hyperparams as hp
from embed_dataset import EmbedDataset
from caption_model import CaptionModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ADJUST THESE VALUES FOR TRAINING:
model_save_path = os.path.join('models', 'unfrozen_gpt2')
resume_model = None
resume = False

if __name__ == '__main__':

    # Load datasets
    dataset_train = EmbedDataset('data', True)
    dataset_val = EmbedDataset('data', False)

    N_train = len(dataset_train)
    N_val = len(dataset_val)

    dataloader_train = DataLoader(dataset_train, batch_size = hp.BATCH_SIZE, shuffle = True)
    dataloader_val = DataLoader(dataset_val, batch_size = hp.BATCH_SIZE, shuffle = False)

    # Resume training
    if resume:
        
        # Model
        model = CaptionModel()
        model.to(hp.DEVICE)
        model_checkpoint = torch.load(os.path.join(model_save_path, resume_model))
        model.load_state_dict(model_checkpoint['model_state_dict'])
        
        # Optimizer
        optim = torch.optim.Adam(model.parameters(), hp.LEARN_RATE)
        optim.load_state_dict(model_checkpoint['optimizer_state_dict'])

        # Statistics
        loss_checkpoint = torch.load(os.path.join(model_save_path, 'losses.pt'))
        epoch_train_losses = loss_checkpoint['train_losses']
        epoch_val_losses = loss_checkpoint['val_losses']
        start_epoch = int(''.join(c for c in resume_model if c.isdigit())) + 1

        del model_checkpoint, loss_checkpoint

    # Start new training
    else:
        model = CaptionModel()
        model.to(hp.DEVICE)
        optim = torch.optim.Adam(model.parameters(), hp.LEARN_RATE)

        epoch_train_losses = []
        epoch_val_losses = []
        start_epoch = 0

    # Epoch loop
    for epoch in range(start_epoch, hp.NUM_EPOCHS):
        
        train_loss = 0
        val_loss = 0

        # Iterate over train batches
        model.train()
        with tqdm.tqdm(dataloader_train, unit = 'batch') as tepoch:
            
            N_trained = 0   
            tepoch.set_description(f"Epoch {epoch}")
            for embeds, captions in tepoch:

                # Forward
                optim.zero_grad()
                embeds = embeds.to(hp.DEVICE)
                out = model(embeds, captions, use_labels = True)
                loss = out.loss

                # Backward
                loss.backward()
                optim.step()

                # Log batch statistics
                N_trained += len(captions)
                train_loss += loss.item() * len(captions)
                tepoch.set_postfix(train_loss = train_loss / N_trained)
                time.sleep(0.01)

        # Iterate over validation batches
        model.eval()
        for embeds, captions in iter(dataloader_val):

            # Forward
            embeds = embeds.to(hp.DEVICE)
            out = model(embeds, captions, use_labels = True)
            loss = out.loss

            # Log batch statistics
            val_loss += loss.item() * len(captions)

        # Log epoch statistics
        epoch_train_losses.append(train_loss / N_train)
        epoch_val_losses.append(val_loss / N_val)

        # Save model and epoch statistics
        torch.save({
            'epoch': epoch,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optim.state_dict())}, os.path.join(model_save_path, 'model_epoch{0}.pt'.format(epoch)))
        torch.save({'train_losses': epoch_train_losses,
                    'val_losses': epoch_val_losses}, os.path.join(model_save_path, 'losses.pt'))
        torch.save({'batch_size': hp.BATCH_SIZE,
                    'learn_rate': hp.LEARN_RATE,
                    'hidden_nodes': hp.N_HIDDEN}, os.path.join(model_save_path, 'hyperparams.pt'))

    # Plot loss curves
    plt.plot(epoch_train_losses)
    plt.plot(epoch_val_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('GPT2 Weights Frozen')
    plt.show()

