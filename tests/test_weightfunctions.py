#!/usr/bin/env python
# coding: utf-8

"""
Tests for the weight functions
"""

import os
import csv
import unittest

from yalign.datatypes import Sentence
from yalign.sentencepairscore import SentencePairScore
from yalign.wordpairscore import WordPairScore
from yalign.tuscore import parse_training_data


class TestSentencePairScore(unittest.TestCase):
    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        word_scores = os.path.join(base_path, "data", "test_word_scores.csv")
        word_pair_score = WordPairScore(word_scores)
        self.alignments_file = os.path.join(base_path, "data", "test_tus.csv")
        training_data = parse_training_data(self.alignments_file)
        self.score = SentencePairScore()
        self.score.train(training_data, word_pair_score)

    def test_does_not_raises_errors(self):
        # Since I can't test if its a good or a bad alignment at this level
        # I'll just run the code to check that no exceptions are raised
        a = Sentence(u"house you".split(), position=0.5)
        b = Sentence(u"casa usted".split(), position=0.5)
        self.score(a, b)
        a = Sentence(u"Valar Morghulis".split(), position=0.0)
        b = Sentence(u"Dracarys".split(), position=1.0)
        self.score(a, b)

    def test_score_order(self):
        a = Sentence(u"Call History .".split(), position=0.0)
        b = Sentence(u"Historial de llamadas .".split(), position=0.0)
        score1 = self.score(a, b)
        a = Sentence(u"Replace the cover .".split(), position=0.0)
        b = Sentence(u"Vuelva a ingresar un nuevo código de bloqueo .".split(),
                     position=0.26)
        score2 = self.score(a, b)
        self.assertGreater(score1, score2)

    def test_score_in_bounds(self):
        for alignment in parse_training_data(self.alignments_file):
            score = self.score(*alignment)
            self.assertGreaterEqual(score, self.score.min_bound)
            self.assertLessEqual(score, self.score.max_bound)


class TestWordPairScore(unittest.TestCase):
    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        translations = os.path.join(base_path, "data", "test_word_scores.csv")
        self.score_word = WordPairScore(translations)

    def test_unicode(self):
        with self.assertRaises(ValueError):
            self.score_word("not", "unicode")
        with self.assertRaises(ValueError):
            self.score_word(u"this", "butnotthis")
        with self.assertRaises(ValueError):
            self.score_word("thisnot", u"butthis")
        score = self.score_word(u"ŧ~]ĸ\æßđæ@~½\ŋĸ}ß", u"ßøþł€ŧ€")
        self.assertEqual(score, 1.0)

    def test_not_spaces(self):
        with self.assertRaises(ValueError):
            self.score_word(u"some spaces", u"here")
        with self.assertRaises(ValueError):
            self.score_word(u"spaces", u"in here")

    def test_non_existing(self):
        score = self.score_word(u"goisdjgoi", u"ijgoihpmf3o")
        self.assertEqual(score, 1.0)

    def test_existing(self):
        score = self.score_word(u"house", u"casa")
        self.assertEqual(score, 0.0)
        score = self.score_word(u"you", u"vos")
        self.assertEqual(score, 0.25)

    def test_non_existing_equal_fields(self):
        score = self.score_word(u"Machinalis", u"Machinalis")
        self.assertEqual(score, 0.0)
        score = self.score_word(u"valar", u"valar")
        self.assertEqual(score, 0.0)
        score = self.score_word(u"morghulis", u"morghulis")
        self.assertEqual(score, 0.0)

    def test_existing_equal_fields(self):
        score = self.score_word(u"house", u"house")
        self.assertEqual(score, 1.0)
        score = self.score_word(u"you", u"you")
        self.assertEqual(score, 1.0)

    def test_case_insensitive(self):
        score1 = self.score_word(u"house", u"casa")
        score2 = self.score_word(u"HOUSE", u"CASA")
        score3 = self.score_word(u"HOusE", u"caSA")
        self.assertEqual(score1, score2)
        self.assertEqual(score2, score3)

    def test_boundaries(self):
        score1 = self.score_word(u"house", u"casa")
        score2 = self.score_word(u"Tyrion", u"Lanister")
        score3 = self.score_word(u"the", u"cake")
        min_bound = self.score_word.min_bound
        max_bound = self.score_word.max_bound
        self.assertTrue(min_bound <= score1 <= max_bound)
        self.assertTrue(min_bound <= score2 <= max_bound)
        self.assertTrue(min_bound <= score3 <= max_bound)


if __name__ == "__main__":
    unittest.main()
