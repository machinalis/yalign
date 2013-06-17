#!/usr/bin/env python
# coding: utf-8

"""
Tests for Translation Units score function
"""

import os
import csv
import unittest
import tempfile
from yalign import tu_score


class TestScoreTU(unittest.TestCase):
    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        word_scores = os.path.join(base_path, "data",
                                   "test_word_scores.pickle")
        self.tus = os.path.join(base_path, "data", "test_tus.csv")
        _, classifier_filepath = tempfile.mkstemp()

        tu_score.train_and_save_classifier(self.tus, word_scores,
                                           classifier_filepath)
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

    def test_score_order(self):
        src = u"Call History."
        tgt = u"Historial de llamadas."
        tu = tu_score.TU(src, tgt, 0.0)
        score1 = self.classifier(tu)
        src = u"Replace the cover."
        tgt = u"Vuelva a ingresar un nuevo código de bloqueo."
        tu = tu_score.TU(src, tgt, 0.26)
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
            tu = tu_score.TU(src, tgt, dist, aligned)
            score = self.classifier(tu)

            self.assertGreaterEqual(score, self.classifier.min_bound)
            self.assertLessEqual(score, self.classifier.max_bound)


if __name__ == "__main__":
    unittest.main()
