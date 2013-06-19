#!/usr/bin/env python
# coding: utf-8

"""
"""

import csv
import math
from yalign.svm import SVMClassifier


class WordScore(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.min_bound = 0.0
        self.max_bound = 1.0
        self.translations = {}

        self._parse_words_file()

    def _parse_words_file(self):
        data = csv.reader(open(self.filepath))
        for elem in data:
            word_a, word_b, prob = elem
            if word_a not in self.translations:
                self.translations[word_a] = {}
            self.translations[word_a][word_b] = float(prob)

    def __call__(self, word_a, word_b):
        """
        Scores a word to word alignment using the translation
        probability.
        Scores range from 0 to 1.
        0 means that the words ARE likely translations of each other.
        1 means that the words AREN'T likely translations of each other.
        """

        if not isinstance(word_a, unicode) or not isinstance(word_b, unicode):
            raise ValueError("Source and target words must be unicode")
        if word_a.count(u" ") or word_b.count(u" "):
            raise ValueError("Words cannot have spaces")

        word_a = word_a.lower()
        word_b = word_b.lower()

        if word_a not in self.translations and word_a == word_b:
            return 0.0
        elif word_a not in self.translations:
            return 1.0
        elif word_b not in self.translations[word_a]:
            return 1.0
        return 1.0 - self.translations[word_a][word_b]


class TUScore(object):
    def __init__(self, filepath):
        self.classifier = SVMClassifier.load(filepath)
        self.min_bound = 0.0
        self.max_bound = 1.0

    def __call__(self, tu):
        """
        Returns the score of a sentence.
        The result will always be a in (self.min_bound, self.max_bound)
        """
        score = self.classifier.score(tu)
        result = logistic_function(score * 3)
        assert self.min_bound <= result <= self.max_bound
        return result


def logistic_function(x):
    """
    See: http://en.wikipedia.org/wiki/Logistic_function
    """
    return 1 / (1 + math.e ** (-x))
