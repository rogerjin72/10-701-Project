import torch

# CUDA
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Dataset and dataloading
DATA_FILENAME_LEN = 12          # Number of digits in data filenames
RESNET_50_INPUT_SIZE = 224      # Input size of the Resnet 50 architecture

# Training loop
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARN_RATE = 1E-6

# Network structure
N_HIDDEN = 2560