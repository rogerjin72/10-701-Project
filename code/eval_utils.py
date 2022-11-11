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
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction


def get_bleu(prediction: str, targets: list):
    """
    get bleu score for a sentence

    Parameters
    ----------
    predictions : 
        a string
    targets : 
        list of candidates
    
    Returns
    -------
    float
        BLEU score
    """
    p_tokens = word_tokenize(prediction)
    t_tokens = [word_tokenize(t) for t in targets]

    # added smoothing in case 3 or 4 grams not present
    chencherry = SmoothingFunction()
    bleu = sentence_bleu(t_tokens, p_tokens, smoothing_function=chencherry.method1)
    return bleu


def get_bleu_corpus(prediction: list, targets: list, n_grams=4):
    """
    get bleu score for a corpus

    Parameters
    ----------
    predictions : 
        a list of predictions
    targets : 
        list of lists of candidates
    n_grams : int, optional
        max n-gram size for bleu, default 4
    
    Returns
    -------
    float
        BLEU score
    """
    p_tokens = [word_tokenize(p) for p in prediction]
    t_tokens = [[word_tokenize(t)  for t in ts] for ts in targets]

    # added smoothing in case 3 or 4 grams not present
    chencherry = SmoothingFunction()
    weights = (1/n_grams, ) * n_grams
    bleu = corpus_bleu(t_tokens, p_tokens, weights=weights, smoothing_function=chencherry.method1)
    return bleu

