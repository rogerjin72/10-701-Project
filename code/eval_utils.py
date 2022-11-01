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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def get_bleu(prediction: str, targets: list):
    """
    get bleu score for a corpus

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

