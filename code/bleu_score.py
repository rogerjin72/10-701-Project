import torch
import os
import json
import hyperparams as hp

from eval_utils import get_bleu
from pathlib import Path
from caption_model import CaptionModel, Predictor
from transformers import GPT2Tokenizer
from embed_dataset import EmbedDataset
from tqdm import tqdm

DIR = Path(os.getcwd())
MODEL_PATH = DIR / 'models' / 'unfrozen_gpt2'
SAVE_PATH = DIR / 'data' / 'eval_data' / 'bleu' 

scores = {}
dataset = EmbedDataset('data', train = False, all_caps = True, collate = True)
fns = os.listdir(MODEL_PATH)

for i in range(0, len(fns), 4):
    model = fns[i]
    if 'model' not in model:
        continue
    checkpoint = torch.load(MODEL_PATH / model, map_location=torch.device(hp.DEVICE))
    decoder = CaptionModel()
    decoder.load_state_dict(checkpoint['model_state_dict'])
    predictor = Predictor(decoder)
    
    greedy_predictions = []
    top_k_predictions = []

    greedy_bleu = []
    top_k_bleu = []

    # val caption is a list of lists
    with tqdm(iter(dataset), total = len(dataset)) as tpass:
        for img, captions in tpass:
            token_ids, _ = predictor.greedy_predict(img, limit=20)
            greedy_pred = decoder.tokenizer.decode(token_ids)
            greedy_predictions.append(decoder.tokenizer.decode(token_ids))
            greedy_bleu.append(get_bleu(greedy_pred, captions))

            token_ids, _ = predictor.top_k_predict(img, limit=20)
            top_k_pred = decoder.tokenizer.decode(token_ids)
            top_k_predictions.append(decoder.tokenizer.decode(token_ids))
            top_k_bleu.append(get_bleu(greedy_pred, captions))

    scores[model] = {'greedy': sum(greedy_bleu) / len(greedy_bleu), 'top_k': sum(top_k_bleu) / len(top_k_bleu)}
    
torch.save(scores, SAVE_PATH / 'unfrozen_gpt2_val_bleu.pt')
