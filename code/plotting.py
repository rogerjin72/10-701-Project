import os
import torch
import numpy as np
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# LOSS PLOTS
fig, ax = plt.subplots(nrows = 1, ncols = 2)
c1 = u'#1f77b4'
c2 = u'#ff7f0e'

data = torch.load(os.path.join('models', 'ViT_conv1d_frozen_gpt2', 'losses.pt'))
train_loss = data['train_losses']
val_loss = data['val_losses']
print(train_loss[20])
print(val_loss[20])
ax[0].plot(train_loss, color = c1, linestyle = '-',  label = 'Conv1D | Training Loss',    marker = '.')
ax[0].plot(val_loss,   color = c1, linestyle = '--', label = 'Conv1D | Validation Loss',  marker = '.')

data = torch.load(os.path.join('models', 'ViT_conv2d_frozen_gpt2', 'losses.pt'))
train_loss = data['train_losses']
val_loss = data['val_losses']
print(train_loss[20])
print(val_loss[20])
ax[0].plot(train_loss, color = c2, linestyle = '-',  label = 'Conv2D | Training Loss',    marker = '.')
ax[0].plot(val_loss,   color = c2, linestyle = '--', label = 'Conv2D | Validation Loss',  marker = '.')

ax[0].set_xlim(left = 0, right = 20)
ax[0].set_ylim(bottom = 1.5, top = 2.5)
ax[0].set_title('Conv 2D All Captions 4x4')
ax[0].legend()
plt.show()

# BLEU PLOTS
# data = torch.load(os.path.join('data', 'eval_data', 'bleu', 'unfrozen_gpt2_val_bleu.pt'))
# epochs = []
# greedy_bleus = []
# top_k_bleus = []
# for model in data.keys():
#     epoch = ''.join(x for x in model if x.isdigit())
#     bleu = data[model]
#     greedy_bleu = bleu['greedy']
#     top_k_bleu = bleu['top_k']

#     epochs.append(epoch)
#     greedy_bleus.append(greedy_bleu)
#     top_k_bleus.append(top_k_bleu)
# plt.plot(epochs, greedy_bleus, color = u'#1f77b4', marker= '.', label = 'Unfrozen Greedy')
# plt.plot(epochs, top_k_bleus, color = u'#1f77b4', linestyle = '--', marker= '.', label = 'Unfrozen Top-5')

# data = torch.load(os.path.join('data', 'eval_data', 'bleu', 'frozen_gpt2_val_bleu.pt'))
# epochs = []
# greedy_bleus = []
# top_k_bleus = []
# for model in data.keys():
#     epoch = ''.join(x for x in model if x.isdigit())
#     bleu = data[model]
#     greedy_bleu = bleu['greedy']
#     top_k_bleu = bleu['top_k']

#     epochs.append(epoch)
#     greedy_bleus.append(greedy_bleu)
#     top_k_bleus.append(top_k_bleu)
# plt.plot(epochs, greedy_bleus, color = u'#ff7f0e', marker= '.', label = 'Frozen Greedy')
# plt.plot(epochs, top_k_bleus, color = u'#ff7f0e', linestyle = '--', marker= '.', label = 'Frozen Top-5')


# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('BLEU Score')
# plt.show()