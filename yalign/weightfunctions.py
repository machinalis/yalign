#!/usr/bin/env python
# coding: utf-8

"""
"""

import csv
import math
from yalign.svm import SVMClassifier


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
