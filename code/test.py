from coco_dataset import *
from lexical_diversity import lex_div as ld

dataset = COCODataset('data/coco_data', False)
captions = [x['caption'] for x in dataset.val_annot]
all_caps = ''
for cap in captions:
    all_caps += cap + ' '
tok = ld.tokenize(all_caps)
print(ld.mtld_ma_bid(tok))