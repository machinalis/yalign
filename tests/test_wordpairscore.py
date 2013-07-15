#!/usr/bin/env python
# coding: utf-8

import os
import unittest
from yalign.wordpairscore import WordPairScore


class TestWordPairScore(unittest.TestCase):
    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        translations = os.path.join(base_path, "data", "test_word_scores.csv")
        self.score_word = WordPairScore(translations)

    # FIXME: write tests


if __name__ == "__main__":
    unittest.main()
