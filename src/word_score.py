#!/usr/bin/env python
# coding: utf-8

"""
"""

try:
    import cPickle as pickle
except ImportError:
    import pickle


PICKLED_DATA_FILEPATH = "translations.pickle"
TRANSLATION = pickle.load(open(PICKLED_DATA_FILEPATH))

def score_word(src, tgt):
    """
    Scores a word to word alignment using the translation
    probability.
    """
    src = src.lower()
    tgt = tgt.lower()

    if src == tgt:
        if src in TRANSLATION and tgt in TRANSLATION[src]:
            return TRANSLATION[src][tgt]
        else:
            return 1.0
    else:
        if src not in TRANSLATION:
            return 0.0
        return TRANSLATION[src].get(tgt, 0.0)


print score_word("you", "usted")
print score_word("Machinalis", "Machinalis")
