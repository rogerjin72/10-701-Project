import torch
import os
import hyperparams as hp

from eval_utils import get_bleu
from pathlib import Path
from caption_model import CaptionModel, Predictor
from embed_dataset import EmbedDataset
from tqdm import tqdm

DIR = Path(os.getcwd())
MODEL_PATH = DIR / 'models' / 'frozen_gpt2'
SAVE_PATH = DIR / 'data' / 'eval_data' / 'bleu'

scores = {}
preds = {}
dataset = EmbedDataset('data', train = False, all_caps = True, collate = True)
fns = os.listdir(MODEL_PATH)
fns = [fn for fn in fns if 'model' in fn]
fns.sort(key = lambda fn: int(''.join(x for x in fn if x.isdigit())))

for i in range(0, len(fns), 2):
    model = fns[i]
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
            greedy_predictions.append(greedy_pred)
            greedy_bleu.append(get_bleu(greedy_pred, captions))

            token_ids, _ = predictor.top_k_predict(img, limit=20, k = 5)
            top_k_pred = decoder.tokenizer.decode(token_ids)
            top_k_predictions.append(top_k_pred)
            top_k_bleu.append(get_bleu(top_k_pred, captions))

    scores[model] = {'greedy': sum(greedy_bleu) / len(greedy_bleu), 'top_k': sum(top_k_bleu) / len(top_k_bleu)}
    preds[model] = {'greedy': greedy_predictions, 'top_k': top_k_predictions}
        
torch.save(preds, SAVE_PATH / 'frozen_pgt2_val_pred.pt')
torch.save(scores, SAVE_PATH / 'frozen_gpt2_val_bleu.pt')
