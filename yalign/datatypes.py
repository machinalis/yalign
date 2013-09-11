# -*- coding: utf-8 -*-

"""
Module of some basic data types.
"""


def _is_tokenized(word):
    """
    Runs some checks to see if the word is tokenized.
    Note: if this functions returns True doesn't mean is really tokenized, but
    if returns False you know it's not tokenized propperly.
    """
    return not ((word.endswith(".") or word.endswith(",")) and
                word[:-1].isalpha())


class Sentence(list):
    def __init__(self, iterable=None, text=None):
        self.text = text
        if iterable is not None:
            super(Sentence, self).__init__(iterable)
        else:
            super(Sentence, self).__init__()

    def check_is_tokenized(self):
        message = u"Word {!r} is not tokenized"
        for word in self:
            if not _is_tokenized(word):
                raise ValueError(message.format(word))

    def to_text(self):
        return self.text.encode('utf-8') if self.text else ' '.join(self).encode('utf-8')


class SentencePair(list):
    """
    An association of two sentences with one attribute
    to indicate if they are considered aligned.
    """
    def __init__(self, sentence_a, sentence_b, aligned=None):
        super(SentencePair, self).__init__([sentence_a, sentence_b])
        self.a = sentence_a
        self.b = sentence_b
        self.aligned = aligned


class ScoreFunction(object):
    """
    Abstract Base class for callable objects that provide a real value score.
    The min_bound and max_bound are used to assert the score range.
    """
    def __init__(self, min_bound, max_bound):
        self.min_bound = min_bound
        self.max_bound = max_bound
