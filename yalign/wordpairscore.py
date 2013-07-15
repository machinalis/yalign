# -*- coding: utf-8 -*-

import csv
from yalign.datatypes import ScoreFunction


class WordPairScore(ScoreFunction):
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

    def __call__(self, sentence_a, sentence_b):
        result = []
        values = {}
        for word_a in sentence_a:
            word_a = word_a.lower()
            if word_a in self.translations:
                values.update(self.translations[word_a])
        for word_b in sentence_b:
            word_b = word_b.lower()
            if word_b in values:
                result.append(values[word_b])
        return result
