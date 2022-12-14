from embed_dataset import *
from lexical_diversity import lex_div as ld
from transformers import GPT2Tokenizer, GPT2LMHeadModel

dataset = EmbedDataset('data', 'encoding_vit', train = True, all_caps = True)
captions = [x['caption'] for x in dataset.train_annot]
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
tokenizer.pad_token = tokenizer.eos_token

max_len = 0
max_cap = ''
idx = 0
for cap in captions:
    tok = tokenizer(cap, return_tensors='pt', padding=True)
    cap_len = tok['input_ids'].numel()
    if cap_len > max_len and idx not in [312977, 567870]:
        max_len = cap_len
        max_cap = cap
        max_idx = idx
    idx += 1

print(max_idx)
print(max_len)
print(max_cap)
print(tokenizer(max_cap, return_tensors='pt', padding=True))
