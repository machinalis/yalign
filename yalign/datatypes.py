# -*- coding: utf-8 -*-


def _is_tokenized(word):
    """
    Runs some checks to see if the word is tokenized.
    Note: if this functions returns True doesn't mean is really tokenized, but
    if returns False you know it's not tokenized propperly.
    """
    # FIXME: add harder checks
    return not ((word.endswith(".") or word.endswith(",")) and
                word[:-1].isalpha())


class Sentence(list):
    def __init__(self, iterable=None):
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
        return ' '.join(self).encode('utf-8')


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
