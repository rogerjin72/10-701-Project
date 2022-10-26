import torch

#  Set up CUDA
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Dataset and dataloading
DATA_FILENAME_LEN = 12          # Number of digits in data filenames
RESNET_50_INPUT_SIZE = 224      # Input size of the Resnet 50 architecture

# Training loop
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARN_RATE = 0.01