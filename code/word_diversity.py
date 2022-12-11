from lexical_diversity import lex_div as ld
import os
import json

model = 'conv2_all_caps_4x4'
epoch = 10
method = 'beam'

prediction_path = os.path.join('data', 'predictions', model, '{0}_epoch'.format(epoch), '{0}.json'.format(method))
with open(prediction_path) as f:
    data = json.load(f)

all_pred = ''
for val in data.values():
    val = val.replace('<|endoftext|>', '')
    all_pred = all_pred + ' ' + val
tok = ld.tokenize(all_pred)
print(ld.mtld_ma_bid(tok))
