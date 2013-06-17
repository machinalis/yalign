#!/usr/bin/env python
# coding: utf-8

"""
"""

import math
from svm import SVMClassifier

try:
    import cPickle as pickle
except ImportError:
    import pickle


class WordScore(object):
    def __init__(self, filepath):
        self.translations = pickle.load(open(filepath))
        self.min_bound = 0.0
        self.max_bound = 1.0

    def __call__(self, src, tgt):
        """
        Scores a word to word alignment using the translation
        probability.
        Scores range from 0 to 1.
        0 means that the words ARE likely translations of each other.
        1 means that the words AREN'T likely translations of each other.
        """

        if not isinstance(src, unicode) or not isinstance(tgt, unicode):
            raise ValueError("Source and target words must be unicode")
        if src.count(u" ") or tgt.count(u" "):
            raise ValueError("Words cannot have spaces")

        src = src.lower()
        tgt = tgt.lower()

        if src not in self.translations and src == tgt:
            return 0.0
        elif src not in self.translations:
            return 1.0
        return 1.0 - self.translations[src][tgt]


class TUScore(object):
    def __init__(self, filepath):
        self.classifier = SVMClassifier.load(filepath)
        self.dimention = len(self.classifier.problem.attributes)
        self.min_bound = 0
        self.max_bound = 2 * math.sqrt(self.dimention)

    def __call__(self, tu):
        """
        Returns the score of a sentence.
        The result will always be a in (self.min_bound, self.max_bound)
        """
        score = self.classifier.score(tu)
        constant = math.sqrt(self.dimention)
        # We add sqrt(n) to avoid negative numbers.
        result = score + constant

        assert self.min_bound <= result <= self.max_bound
        return result
