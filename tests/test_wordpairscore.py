#!/usr/bin/env python
# coding: utf-8

import os
import unittest
from yalign.wordpairscore import WordPairScore


class TestWordPairScore(unittest.TestCase):

    def setUp(self):
        self.word_pair_score = self._create_word_pair_score('test_word_scores.csv')

    def _create_word_pair_score(self, filename):
        base_path = os.path.dirname(os.path.abspath(__file__))
        translations = os.path.join(base_path, "data", filename)
        return WordPairScore(translations)

    def test_load_translations_in_gz_format(self):
        word_pair_score = self._create_word_pair_score('test_word_scores.csv.gz')
        translations = word_pair_score.translations
        self.check_translations(translations)

    def test_translations(self):
        translations = self.word_pair_score.translations
        self.check_translations(translations)

    def check_translations(self, translations):
        self.assertEqual(3, len(translations))
        self.assertEqual(translations[u'house'], {u'casa': 1.0})
        self.assertEqual(translations[u'you'], {u'ustedes': 0.625,
                                                u'vosotros': 0.375,
                                                u'vos': 0.75})
        self.assertEqual(translations[u'yourselves'], {u'vosotros': 0.75})

if __name__ == "__main__":
    unittest.main()
