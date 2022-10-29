import nltk

try:
    nltk.data.find('tokenizers/punkt.zip')
except LookupError:
    import ssl
    
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu


def get_bleu(predictions: list, targets: list):
    """
    get bleu score for a corpus

    Parameters
    ----------
    predictions : list
        list of model predictions as strings
    targets : list
        list of lists, inner list is the ground truth captions
    
    Returns
    -------
    float
        BLEU score
    """
    p_tokens = [word_tokenize(p) for p in predictions]
    t_tokens = []

    for target in targets:
        t_tokens.append([word_tokenize(t) for t in target])

    bleu = corpus_bleu(t_tokens, p_tokens)
    return bleu

