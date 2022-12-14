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
MODEL_PATH = DIR / 'models' / 'ViT_conv2d_frozen_gpt2_allcaps_4x4_large'

for i in range(1):

    checkpoint = torch.load(MODEL_PATH / 'model_epoch{0}.pt'.format(i), map_location = torch.device(hp.DEVICE))

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

        token_ids, _ = predictor.beam_predict(img, limit=50, k = 5)
        beam_pred = decoder.tokenizer.decode(token_ids)
        beam_prediction[img_id] = beam_pred.split('.')[0] + '.'

    json.dump(beam_prediction, open(SAVE_PATH / 'conv2_all_caps_4x4_large/{0}_epoch/beam.json'.format(i), 'w'), indent=4)

for i in range(1, 11):
    beam_prediction = json.load(open(SAVE_PATH / 'conv2_all_caps_4x4_large/{0}_epoch/beam.json'.format(i), 'r'))
    targets = json.load(open('data/coco_data/coco_annotations/coco_val_annot.json'))
    targets = pd.DataFrame(targets['annotations'])
    targets = targets.groupby('image_id').agg({'caption': list})
    targets = targets['caption'].to_list()

    r = {}
    r['beam_bleu_4'] = round(get_bleu_corpus(beam_prediction.values(), targets, 4), 4)
    r['beam_bleu_3'] = round(get_bleu_corpus(beam_prediction.values(), targets, 3), 4)
    print(r)