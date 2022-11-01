import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Overfitted model
# data = torch.load(os.path.join('models', 'overfit', 'losses.pt'))
# train_loss = data['train_losses']
# val_loss = data['val_losses']

# plt.plot(train_loss, label = 'Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss on Small Dataset')
# plt.show()

fig, ax = plt.subplots(nrows = 1, ncols = 2)
c1 = [i / 255 for i in [98, 191, 110]]
c2 = [i / 255 for i in [255, 176, 120]]


# General model
data = torch.load(os.path.join('models', 'unfrozen_gpt2', 'losses.pt'))
train_loss = data['train_losses']
val_loss = data['val_losses']

ax[0].plot(train_loss, color = c1, label = 'Training Loss')
ax[0].plot(val_loss, color = c2, label = 'Validation Loss')
ax[0].axvline(np.argmin(val_loss), color = c2, linestyle = '--', linewidth = 1)
ax[0].axhline(np.min(val_loss), color = c2, linestyle = '--', linewidth = 1)
ax[0].set_xlim(left = 0, right = 20)

ax[0].legend(loc = 'upper left')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Finetuned GPT-2')

data = torch.load(os.path.join('models', 'frozen_gpt2', 'losses.pt'))
train_loss = data['train_losses']
val_loss = data['val_losses']
print(min(train_loss))
print(min(val_loss))

ax[1].plot(train_loss, color = c1, label = 'Training Loss')
ax[1].plot(val_loss, color = c2, label = 'Validation Loss')
ax[1].axvline(np.argmin(val_loss), color = c2, linestyle = '--', linewidth = 1)
ax[1].axhline(np.min(val_loss), color = c2, linestyle = '--', linewidth = 1)
ax[1].set_xlim(left = 0, right = 20)

ax[1].legend(loc = 'upper left')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].set_title('Frozen GPT-2')
plt.show()