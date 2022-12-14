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
ATTN_LAYERS = 4                 # Number of attention layers in the align network
ATTN_HEADS = 4                  # Number of attention heads in the align network
PREFIX_LEN = 16                 # Prefix length input to GPT2
EMBED_LEN = 197                 # The length of the ViT embedding
EMBED_DIM = 768                 # The dimensionality of the embedding
VIT_DIM = 14                    # The edge length of the ViT embedding
GPT = 'gpt2-large'              # Specify size of GPT2 model
FREEZE_GPT2 = True              # Freeze gpt-2 weights during training?