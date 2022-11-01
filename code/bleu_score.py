import torch
import os
import json
import hyperparams as hp

from eval_utils import get_bleu
from pathlib import Path
from caption_model import CaptionModel, Predictor
from transformers import GPT2Tokenizer
from tqdm import tqdm

DIR = Path(os.getcwd())
MODEL_PATH = DIR / 'models' / 'unfrozen'
IMG_PATH = DIR / 'data' / 'encoding_blocks'

scores = {}

for model in os.listdir(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH / model, map_location=torch.device(hp.DEVICE))
    decoder = CaptionModel()
    decoder.load_state_dict(checkpoint['model_state_dict'])
    predictor = Predictor(decoder)
    
    greedy_predictions = []
    top_k_predictions = []

    greedy_bleu = []
    top_k_bleu = []

    # val caption is a list of lists
    for img, captions in tqdm(zip(val_embeddings, val_captions), total=len(val_embeddings)):
        token_ids, scores = predictor.greedy_predict(img, limit=20)
        greedy_pred = decoder.tokenizer.decode(token_ids)
        greedy_predictions.append()
        greedy_bleu.append(get_bleu(greedy_pred, captions))

        token_ids, scores = predictor.top_k_predict(img, limit=20)
        top_k_pred = decoder.tokenizer.decode(token_ids)
        top_k_predictions.append(decoder.tokenizer.decode(token_ids))
        top_k_bleu.append(get_bleu(greedy_pred, captions))

    scores[model] = {'greedy': sum(greedy_bleu)/len(greedy_bleu), 'top_k': sum(top_k_bleu) / len(top_k_bleu)}
