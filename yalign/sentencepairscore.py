# -*- coding: utf-8 -*-

import math
from simpleai.machine_learning import ClassificationProblem, is_attribute

from yalign.svm import SVMClassifier
from yalign.datatypes import ScoreFunction, SentencePair


class SentencePairScore(ScoreFunction):

    SCORE_MULTIPLIER = 3

    def __init__(self):
        super(SentencePairScore, self).__init__(0, 1)
        self.classifier = None
        self.sign = 1

    def train(self, pairs, word_score_function):
        """
        Trains the sentence pair likelihood score using examples.
        `pairs` is an interable of `SentencePair` instances.
        `word_score_function` is an instance of ScoreFunction, perhaps even an
        instance of `WordPairScore`.
        """
        pairs = list(pairs)
        self.problem = SentencePairScoreProblem(word_score_function)
        self.classifier = SVMClassifier(pairs, self.problem)
        class_ = None
        for a, b in pairs:
            sent = SentencePair(a, b)
            score = self.classifier.score(sent)
            if score != 0:
                class_ = bool(self.classifier.classify(sent)[0])
                if (score > 0 and class_ is True) or \
                   (score < 0 and class_ is False):
                    self.sign = -1
                break
        if class_ is None:
            raise ValueError("Cannot infer sign with this data")

    def load(self, filepath):
        self.classifier = SVMClassifier.load(filepath)

    def save(self, filepath):
        self.classifier.save(filepath)

    def __call__(self, a, b):
        """
        Returns the score of a sentence.
        """
        if self.classifier is None:
            raise LookupError("Score not trained or loaded yet")
        a = SentencePair(a, b)
        score = self.classifier.score(a) * self.sign
        result = logistic_function(score * SentencePairScore.SCORE_MULTIPLIER)
        assert self.min_bound <= result <= self.max_bound
        return result

    @property
    def word_pair_score(self):
        return self.classifier.problem.word_pair_score


class SentencePairScoreProblem(ClassificationProblem):
    def __init__(self, word_pair_score):
        super(SentencePairScoreProblem, self).__init__()
        self.word_pair_score = CacheOfSizeOne(word_pair_score)

    @is_attribute
    def linear_word_match(self, alignment):
        total = sum(self.word_pair_score(alignment.a, alignment.b))
        return total / float(max(len(alignment.a), len(alignment.b)))

    @is_attribute
    def linear_word_count(self, alignment):
        total = len(self.word_pair_score(alignment.a, alignment.b))
        return total / float(max(len(alignment.a), len(alignment.b)))

    @is_attribute
    def character_count_ratio(self, alignment):
        length_1 = len([c for word in alignment.a for c in word])
        length_2 = len([c for word in alignment.b for c in word])
        return ratio(length_1, length_2)

    def target(self, alignment):
        return alignment.aligned


def ratio(a, b):
    if max(a, b) == 0:
        return 0.0
    return min(a, b) / float(max(a, b))


def logistic_function(x):
    """
    See: http://en.wikipedia.org/wiki/Logistic_function
    """
    return 1 / (1 + math.e ** (-x))


class CacheOfSizeOne(object):
    f = None

    def __init__(self, f):
        self.f = f
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        if args != self.args or kwargs != self.kwargs:
            self.result = self.f(*args, **kwargs)
            self.args = args
            self.kwargs = kwargs
        return self.result

    def __getattr__(self, name):
        return getattr(self.f, name)
