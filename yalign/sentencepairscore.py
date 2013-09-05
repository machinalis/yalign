# -*- coding: utf-8 -*-
"""
Module for code related to scoring sentence pairs.
"""

import math
from simpleai.machine_learning import ClassificationProblem, is_attribute

from yalign.svm import SVMClassifier
from yalign.datatypes import ScoreFunction, SentencePair
from yalign.utils import CacheOfSizeOne


class SentencePairScore(ScoreFunction):
    """
    This class provides a score of how close two sentences are to being translations
    of each other.
    """
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

    def __call__(self, a, b):
        """
        Returns a score representing how good a
        translation sentence b is of sentence a.
        """
        if self.classifier is None:
            raise LookupError("Score not trained or loaded yet")
        a = SentencePair(a, b)
        score = self.classifier.score(a) * self.sign
        result = self.logistic_function(score * SentencePairScore.SCORE_MULTIPLIER)
        assert self.min_bound <= result <= self.max_bound
        return result

    def logistic_function(self, x):
        """ See: http://en.wikipedia.org/wiki/Logistic_function"""
        return 1 / (1 + math.e ** (-x))

    @property
    def word_pair_score(self):
        return self.classifier.problem.word_pair_score


class SentencePairScoreProblem(ClassificationProblem):
    """
    Provides the classifier attributes.
    """
    def __init__(self, word_pair_score):
        """
        Some attributes need a WordPairScore.
        """
        super(SentencePairScoreProblem, self).__init__()
        self.word_pair_score = CacheOfSizeOne(word_pair_score)

    @is_attribute
    def sum_of_word_pair_scores(self, sentence_pair):
        """
        The sum of the word pair scores divided by
        the word count of the longest sentence.
        """
        scores = self.word_pair_score(sentence_pair.a, sentence_pair.b)
        return sum(scores) / self._max_word_count(sentence_pair)

    @is_attribute
    def number_of_word_pair_scores(self, sentence_pair):
        """
        The number of the word pair scores divided by
        the number of words of the longest sentence.
        """
        scores = self.word_pair_score(sentence_pair.a, sentence_pair.b)
        return len(scores) / self._max_word_count(sentence_pair)

    @is_attribute
    def ratio_of_character_count(self, sentence_pair):
        """
        The ratio of the sentence with the least characters
        over the sentence with the most characters.
        """
        char_count_a = self._number_of_characters(sentence_pair.a)
        char_count_b = self._number_of_characters(sentence_pair.b)
        return self._ratio(char_count_a, char_count_b)

    def target(self, sentence_pair):
        """ Returns if these sentences are translations of each other """
        return sentence_pair.aligned

    def _max_word_count(self, sentence_pair):
        word_count_a = len(sentence_pair.a)
        word_count_b = len(sentence_pair.b)
        return float(max(word_count_a, word_count_b))

    def _number_of_characters(self, sentence):
        return len([c for word in sentence for c in word])

    def _ratio(self, a, b):
        if max(a, b) == 0:
            return 0.0
        return min(a, b) / float(max(a, b))
