# -*- coding: utf-8 -*-


class Sentence(list):
    def __init__(self, iterable=None, position=None):
        if position is not None:
            if not(0 <= position <= 1):
                raise ValueError("Position must be between 0 and 1")
            self.position = position
        if iterable is not None:
            super(Sentence, self).__init__(iterable)
        else:
            super(Sentence, self).__init__()


class SentencePair(list):
    def __init__(self, sentence_a, sentence_b, aligned=None):
        super(SentencePair, self).__init__([sentence_a, sentence_b])
        self.a = sentence_a
        self.b = sentence_b
        self.aligned = aligned


class ScoreFunction(object):
    def __init__(self, min_bound, max_bound):
        self.min_bound = min_bound
        self.max_bound = max_bound
