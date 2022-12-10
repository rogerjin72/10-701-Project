import os
import json
import torch
import pandas as pd
import hyperparams as hp

from pathlib import Path
from caption_model import Predictor, CaptionModel
from eval_utils import get_bleu_corpus
from tqdm import tqdm 


DIR = Path(os.getcwd())
SAVE_PATH = DIR / 'data' / 'predictions'
MODEL_PATH = DIR / 'models' / 'ViT_conv2d_frozen_gpt2_allcaps_8x8'

checkpoint = torch.load(MODEL_PATH / 'model_epoch10.pt', map_location = torch.device(hp.DEVICE))

decoder = CaptionModel(conv=2)
decoder.load_state_dict(checkpoint['model_state_dict'])
predictor = Predictor(decoder)

greedy_prediction = {}
beam_prediction = {} 

# val caption is a list of lists
for file in tqdm(sorted(os.listdir('data/encoding_vit/val'))[1:]):
    if file.split('.')[-1] != 'pt':
        continue 

    img_id = file.split('.')[0]
    img_id = img_id.lstrip('0')

    img = torch.load('data/encoding_vit/val/' + file)

    token_ids, _ = predictor.greedy_predict(img, limit=50)
    greedy_pred = decoder.tokenizer.decode(token_ids)
    greedy_prediction[img_id] = greedy_pred

    token_ids, _ = predictor.beam_predict(img, limit=50, k = 5)
    beam_pred = decoder.tokenizer.decode(token_ids)
    beam_prediction[img_id] = beam_pred.split('.')[0] + '.'

json.dump(greedy_prediction, open(SAVE_PATH / '8x8/10_epoch/greedy.json', 'w'), indent=4)
json.dump(beam_prediction, open(SAVE_PATH / '8x8/10_epoch/beam.json', 'w'), indent=4)

DIR = Path(os.getcwd())
PRED_PATH = DIR / 'data' / 'predictions' 
SAVE_PATH = DIR / 'data' / 'eval_data' / 'bleu'

targets = json.load(open('data/coco_data/coco_annotations/coco_val_annot.json'))
targets = pd.DataFrame(targets['annotations'])
targets = targets.groupby('image_id').agg({'caption': list})
targets = targets['caption'].to_list()

r = {}

r['greedy_bleu_4'] = round(get_bleu_corpus(greedy_prediction.values(), targets, 4), 4)
r['beam_bleu_4'] = round(get_bleu_corpus(beam_prediction.values(), targets, 4), 4)

r['greedy_bleu_3'] = round(get_bleu_corpus(greedy_prediction.values(), targets, 3), 4)
r['beam_bleu_3'] = round(get_bleu_corpus(beam_prediction.values(), targets, 3), 4)

r['greedy_bleu_2'] = round(get_bleu_corpus(greedy_prediction.values(), targets, 2), 4)
r['beam_bleu_2'] = round(get_bleu_corpus(beam_prediction.values(), targets, 2), 4)

r['greedy_bleu_1'] = round(get_bleu_corpus(greedy_prediction.values(), targets, 1), 4)
r['beam_bleu_1'] = round(get_bleu_corpus(beam_prediction.values(), targets, 1), 4)

print(r)