import os
import torch
import json
import pandas as pd
from eval_utils import get_bleu_corpus
import matplotlib.pyplot as plt
from lexical_diversity import lex_div as ld
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# ======================= Compute and save MTLD ======================= 
pred_path = os.path.join('data', 'predictions', 'conv2_all_caps_4x4_large')
mtlds = []
for i in range(11):
    preds = json.load(open(os.path.join(pred_path, '{0}_epoch'.format(i), 'beam.json')))
    all_pred = ''
    for val in preds.values():
        val = val.replace('<|endoftext|>', '')
        all_pred = all_pred + ' ' + val
    tok = ld.tokenize(all_pred)
    mtlds.append(ld.mtld_ma_bid(tok))
torch.save(mtlds, os.path.join(pred_path, 'MTLDS_best.pt'))

# ======================= Compute and save BLEU ======================= 
pred_path = os.path.join('data', 'predictions', 'conv2_all_caps_4x4_large')
targets = json.load(open('data/coco_data/coco_annotations/coco_val_annot.json'))
targets = pd.DataFrame(targets['annotations'])
targets = targets.groupby('image_id').agg({'caption': list})
targets = targets['caption'].to_list()
bleus = []
for i in range(11):
    preds = json.load(open(os.path.join(pred_path, '{0}_epoch'.format(i), 'beam.json')))
    bleus.append(100 * round(get_bleu_corpus(preds.values(), targets, 4), 4))
torch.save(bleus, os.path.join(pred_path, 'BLEUS_best.pt'))

# ======================== Plot for best model ======================== 

# Losses
model_path = os.path.join('models', 'ViT_conv2d_frozen_gpt2_allcaps_4x4_large')
losses = torch.load(os.path.join(model_path, 'losses.pt'))
train_loss = losses['train_losses']
val_loss = losses['val_losses']

fig, ax = plt.subplots(1, 2)
ax[0].plot(train_loss, marker = '.', label = 'Training Loss')
ax[0].plot(val_loss, marker = '.', label = 'Validation Loss')
ax[0].legend()
ax[0].set_ylabel('Auto-Regressive Loss')
ax[0].set_xlabel('Epochs')

# BLEU
pred_path = os.path.join('data', 'predictions', 'conv2_all_caps_4x4_large')
bleus = torch.load(os.path.join(pred_path, 'BLEUS_best.pt'))

# MTLD
mtlds = torch.load(os.path.join(pred_path, 'MTLDS_best.pt'))
print(mtlds[-1])

ax[1].plot(bleus, marker = '.')
ax[1].set_ylabel('BLEU-4 Score (Validation)', color = 'C0')
ax[1].set_xlabel('Epochs')
ax[1].tick_params(axis ='y', labelcolor = 'C0')

ax_r = ax[1].twinx()
ax_r.plot(mtlds, marker = '.', color = 'C1')
ax_r.set_ylabel('MTLD (Validation)', color = 'C1')
ax_r.tick_params(axis ='y', labelcolor = 'C1')
plt.show()

# ======================== Plot for all models ======================== 
fig, ax = plt.subplots(1, 4)
top = 2.25
bottom = 1.4

losses = torch.load('models/ViT_conv1d_frozen_gpt2/losses.pt')
train_loss = losses['train_losses']
val_loss = losses['val_losses']
line_train = ax[0].plot(train_loss[:21], marker = '.')
line_val = ax[0].plot(val_loss[:21], marker = '.')
ax[0].set_ylim(top = top, bottom = bottom)
print(min(val_loss))

losses = torch.load('models/ViT_conv2d_frozen_gpt2/losses.pt')
train_loss = losses['train_losses']
val_loss = losses['val_losses']
ax[1].plot(train_loss[:21], marker = '.')
ax[1].plot(val_loss[:21], marker = '.')
ax[1].set_ylim(top = top, bottom = bottom)
ax[1].set_yticklabels([])
print(min(val_loss))

losses = torch.load('models/ViT_conv2d_frozen_gpt2_allcaps_4x4/losses.pt')
train_loss = losses['train_losses']
val_loss = losses['val_losses']
ax[2].plot(train_loss, marker = '.')
ax[2].plot(val_loss, marker = '.')
ax[2].set_ylim(top = top, bottom = bottom)
ax[2].set_yticklabels([])
print(min(val_loss))

losses = torch.load('models/ViT_conv2d_frozen_gpt2_allcaps_8x8/losses.pt')
train_loss = losses['train_losses']
val_loss = losses['val_losses']
ax[3].plot(train_loss, marker = '.')
ax[3].plot(val_loss, marker = '.')
ax[3].set_ylim(top = top, bottom = bottom)
ax[3].set_yticklabels([])
print(min(val_loss))

plt.tight_layout()
plt.show()

