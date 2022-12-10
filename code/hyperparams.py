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
BATCH_SIZE = 32
LEARN_RATE = 2E-5
NUM_EPOCHS = 10
WARMUP_EPOCHS = 1
WEIGHT_DECAY = 1E-2

# Network structure
ATTN_LAYERS = 4
ATTN_HEADS = 4
PREFIX_LEN = 16
EMBED_LEN = 197
EMBED_DIM = 768
VIT_DIM = 14
GPT = 'gp2'
FREEZE_GPT2 = True