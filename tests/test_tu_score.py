#!/usr/bin/env python
# coding: utf-8

"""
Tests for Translation Units score function
"""

import os
import unittest
import tempfile
from yalign import tu_score


class TestScoreTU(unittest.TestCase):
    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        word_scores = os.path.join(base_path, "data", "test_word_scores.pickle")
        tus = os.path.join(base_path, "data", "test_tus.csv")
        _, classifier_filepath = tempfile.mkstemp()

        tu_score.train_and_save_classifier(tus, word_scores, classifier_filepath)
        self.classifier = tu_score.ScoreTU(classifier_filepath)

    def test_does_not_raises_errors(self):
        # Since I can't test if its a good or a bad alignment at this level
        # I'll just run the code to check that no exceptions are raised
        tu = tu_score.TU(u"house you", u"casa usted", 0.0)
        self.classifier(tu)
        tu = tu_score.TU(u"Valar Morghulis", u"Dracarys", 1.0)
        self.classifier(tu)

    def test_empty_sentences(self):
        with self.assertRaises(ValueError):
            tu_score.TU(u"", u"", 0)
        with self.assertRaises(ValueError):
            tu_score.TU(u"Daenerys", u"", 0)
        with self.assertRaises(ValueError):
            tu_score.TU(u"", u"Targaryen", 0)


if __name__ == "__main__":
    unittest.main()
