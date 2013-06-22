# -*- coding: utf-8 -*-

import math
from simpleai.machine_learning import ClassificationProblem, is_attribute

from yalign.datatypes import ScoreFunction, Alignment
from yalign.sequencealigner import SequenceAligner
from yalign.svm import SVMClassifier


class SentencePairScore(ScoreFunction):
    def __init__(self):
        super(SentencePairScore, self).__init__(0, 1)
        self.classifier = None

    def train(self, alignments, word_score_function):
        """
        Trains the sentence pair likelihood score using examples.
        `alignments` is an interable of `Alignment` instances.
        `word_score_function` is an instance of ScoreFunction, perhaps even an
        instance of `WordPairScore`.
        """
        self.classifier = SVMClassifier(alignments,
                                 SentencePairScoreProblem(word_score_function))

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
        a = Alignment(a, b)
        score = self.classifier.score(a)
        result = logistic_function(score * 3)
        # FIXME: Consider moving this to a test
        assert self.min_bound <= result <= self.max_bound
        return result

    @property
    def sentence_pair_aligner(self):
        return self.classifier.problem.aligner

    @property
    def word_pair_score(self):
        return self.classifier.problem.word_pair_score


class SentencePairScoreProblem(ClassificationProblem):
    def __init__(self, word_pair_score):
        super(SentencePairScoreProblem, self).__init__()
        # If gap > 0.5 then the returned value could be > 1.
        self.word_pair_score = word_pair_score
        self.aligner = SequenceAligner(word_pair_score, 0.4999)

    @is_attribute
    def word_score(self, alignment):
        aligns = self.aligner(alignment.a, alignment.b)
        N = max(len(alignment.a), len(alignment.b))
        word_score = sum(x[2] for x in aligns) / float(N)
        # FIXME: Consider moving this to a test
        assert 0 <= word_score <= 1
        return word_score

    @is_attribute
    def position_difference(self, alignment):
        d = alignment.a.position - alignment.b.position
        return abs(d)

    @is_attribute
    def word_length_difference(self, alignment):
        a = len(alignment.a)
        b = len(alignment.b)
        return ratio(a, b)

    @is_attribute
    def uppercase_words_difference(self, alignment):
        a = len([x for x in alignment.a if x.isupper()])
        b = len([x for x in alignment.b if x.isupper()])
        return ratio(a, b)

    @is_attribute
    def capitalized_words_difference(self, alignment):
        a = len([x for x in alignment.a if x.istitle()])
        b = len([x for x in alignment.b if x.istitle()])
        return ratio(a, b)

    def target(self, alignment):
        return alignment.are_really_aligned


def ratio(a, b):
    if max(a, b) == 0:
        return 1.0
    return min(a, b) / float(max(a, b))


def logistic_function(x):
    """
    See: http://en.wikipedia.org/wiki/Logistic_function
    """
    return 1 / (1 + math.e ** (-x))
