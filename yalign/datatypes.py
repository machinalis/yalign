# -*- coding: utf-8 -*-


class AnotatedList(list):
    def __init__(self, iterable=None, **kwargs):
        self.__dict__.update(kwargs)
        if iterable is not None:
            super(AnotatedList, self).__init__(iterable)
        else:
            super(AnotatedList, self).__init__()


class Word(unicode):
    pass


class Sentence(AnotatedList):
    pass


class Document(AnotatedList):
    pass


# FIXME: Consider renaming to AlignedSentences, of SentencePair
class Alignment(list):
    def __init__(self, sentence_a, sentence_b, are_really_aligned=None):
        super(Alignment, self).__init__()
        self.a = sentence_a
        self.b = sentence_b
        self.append(self.a)
        self.append(self.b)
        self.are_really_aligned = are_really_aligned


class ScoreFunction(object):
    def __init__(self, min_bound, max_bound):
        self.min_bound = min_bound
        self.max_bound = max_bound
