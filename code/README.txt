NOTE: We have preserved the directory structure used when loading data/saving models, but have deleted all files from them to save space.

bleu_score_prediction.py
    This script computes the forward pass of a selected model, and computes the BLEU-k score between the generated captions and the ground truth
caption_model.py
    This file contains the class defining the align network and GPT-2 language decoder, along with functions (greedy, top-k, top-p, beam) to decode probabilities into English sentences
coco_dataset.py
    This file loads and pairs images and corresponding captions from the MS-COCO dataset, and provides a convenient way to index them
embed_dataset.py
    This file loads and pairs image EMBEDDINGS (as computed by a Vision Transformer) and corresponding captions, and provides a convenient way to index them
eval_utils.py
    This file provides utility functions used to evaluate model performance, such as computed BLEU score
gen_captions_ood.py
    This file reads in out-of-distribution images and generates captions for them, and saves a captioned image
gen_captions.py
    This file reads in images from the MS COCO dataset, generates captions for them, and saves a captioned image
gen_prefixes.py
    This file performs half of the foward pass on the model in order to compute the prefix inputs to GPT2
hyperparams.py
    This file contains hyperparameters used throughout all files. It is manually changed for each experiment to set desireable hyperparameters
img_caption_model.py
    This file contains the entire image encoder, align network, and text decoder used in the caption generation model
make_img_matrix.py
    This file creates a grid of images and captions for generating figures in the report
plotting.py:
    This file creates loss plots and other result plots for the report
train_loop.py:
    This is the main training loop used to train the caption generation model
transformer.py:
    This file defines the architecture used in the align network. It includes both the transformer and the convolutional aggregation layers
word_diversity.py:
    This file takes in a corpus of generated captions and computes the bidirectional moving average MTLD on them.

