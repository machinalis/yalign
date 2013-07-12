# -*- coding: utf-8 -*-

import csv
from yalign.datatypes import ScoreFunction


class WordPairScore(ScoreFunction):
    """
    Scores the likelihood of `word_a` being a trainslation of `word_b` and
    viceversa using using the translation probability of those words given
    in a dictionary file.
    """
    def __init__(self, dictionary_file):
        super(WordPairScore, self).__init__(0, 1)
        self.filepath = dictionary_file
        self.translations = {}
        self._parse_words_file()

    def _parse_words_file(self):
        # FIXME: Add support for .csv.gz
        # FIXME: Question: Why not (word_a, word_b) as keys?
        # FIXME: Implement the reverse on the outside
        data = csv.reader(open(self.filepath))
        for elem in data:
            word_a, word_b, prob = elem
            word_a = word_a.decode("utf-8").lower()
            word_b = word_b.decode("utf-8").lower()
            if word_a not in self.translations:
                self.translations[word_a] = {}
            self.translations[word_a][word_b] = float(prob)

    def __call__(self, word_a, word_b):
        """
        Scores the likelihood of `word_a` being a trainslation of `word_b` and
        viceversa.
        Scores range from 0 to 1.
        0 means that the words ARE likely translations of each other.
        1 means that the words AREN'T likely translations of each other.
        """

        # FIXME: Consider moving this to a test
        if not isinstance(word_a, unicode) or not isinstance(word_b, unicode):
            raise ValueError("Word A and word B words must be unicode")
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
