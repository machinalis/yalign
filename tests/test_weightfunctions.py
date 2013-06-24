#!/usr/bin/env python
# coding: utf-8

"""
Tests for the weight functions
"""

import os
import csv
import unittest
import tempfile

from yalign.tu import TU
from yalign import tuscore
from yalign.weightfunctions import TUScore, WordScore
from helpers import default_tuscore

class TestScoreTU(unittest.TestCase):

    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        classifier_filepath = default_tuscore() 
        self.classifier = TUScore(classifier_filepath)
        self.tus = os.path.join(base_path, "data", "test_tus.csv")

    def test_does_not_raises_errors(self):
        # Since I can't test if its a good or a bad alignment at this level
        # I'll just run the code to check that no exceptions are raised
        tu = TU(u"house you", u"casa usted", 0.0)
        self.classifier(tu)
        tu = TU(u"Valar Morghulis", u"Dracarys", 1.0)
        self.classifier(tu)

    def test_empty_sentences(self):
        with self.assertRaises(ValueError):
            TU(u"", u"", 0)
        with self.assertRaises(ValueError):
            TU(u"Daenerys", u"", 0)
        with self.assertRaises(ValueError):
            TU(u"", u"Targaryen", 0)

    def test_score_order(self):
        src = u"Call History."
        tgt = u"Historial de llamadas."
        tu = TU(src, tgt, 0.0)
        score1 = self.classifier(tu)
        src = u"Replace the cover."
        tgt = u"Vuelva a ingresar un nuevo código de bloqueo."
        tu = TU(src, tgt, 0.26)
        score2 = self.classifier(tu)
        self.assertGreater(score1, score2)

    def test_score_in_bounds(self):
        data = csv.reader(open(self.tus))
        labels = None
        for elem in data:
            if labels is None:  # First line contains the labels
                labels = dict((x, elem.index(x)) for x in elem)
                continue

            src = elem[labels["src"]].decode("utf-8")
            tgt = elem[labels["tgt"]].decode("utf-8")
            src_pos = float(elem[labels["src idx"]]) / float(elem[labels["src N"]])
            tgt_pos = float(elem[labels["tgt idx"]]) / float(elem[labels["tgt N"]])
            dist = abs(src_pos - tgt_pos)
            aligned = elem[labels["aligned"]]
            tu = TU(src, tgt, dist, aligned)
            score = self.classifier(tu)

            self.assertGreaterEqual(score, self.classifier.min_bound)
            self.assertLessEqual(score, self.classifier.max_bound)


class TestWordScore(unittest.TestCase):
    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        translations = os.path.join(base_path, "data", "test_word_scores.csv")
        self.score_word = WordScore(translations)

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
