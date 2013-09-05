#!/usr/bin/env python
# coding: utf-8

import os
import unittest

from yalign.datatypes import Sentence, SentencePair
from yalign.wordpairscore import WordPairScore
from yalign.sentencepairscore import SentencePairScore, CacheOfSizeOne
from yalign.input_conversion import parallel_corpus_to_documents
from yalign.train_data_generation import training_alignments_from_documents


class TestSentencePairScore(unittest.TestCase):
    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        word_scores = os.path.join(base_path, "data", "test_word_scores_big.csv")
        word_pair_score = WordPairScore(word_scores)
        fin = os.path.join(base_path, "data", "parallel-en-es.txt")
        A, B = parallel_corpus_to_documents(fin)
        self.alignments = list(training_alignments_from_documents(A, B))
        self.score = SentencePairScore()
        self.score.train(self.alignments, word_pair_score)

    def test_generates_numbers(self):
        a = Sentence(u"house you".split())
        b = Sentence(u"casa usted".split())
        x = self.score(a, b)
        self.assertIsInstance(x, (int, float))
        a = Sentence(u"Valar Morghulis".split())
        b = Sentence(u"Dracarys".split())
        x = self.score(a, b)
        self.assertIsInstance(x, (int, float))

    def test_score_order(self):
        a = Sentence(u"Call History .".split())
        b = Sentence(u"Historial de llamadas .".split())
        score1 = self.score(a, b)
        a = Sentence(u"Replace the cover .".split())
        b = Sentence(u"Vuelva a ingresar un nuevo código de bloqueo .".split())
        score2 = self.score(a, b)
        self.assertLess(score1, score2)

    def test_score_in_bounds(self):
        for alignment in self.alignments:
            score = self.score(*alignment)
            self.assertGreaterEqual(score, self.score.min_bound)
            self.assertLessEqual(score, self.score.max_bound)

    def test_linear_word_count_is_better_than_all_mismatchs(self):
        a = Sentence(u"house µa µb µc µd".split())
        b = Sentence(u"casa  µ1 µ2 µ3 µ4".split())
        s1 = self.score.problem.linear_word_count(SentencePair(a, b))

        c = Sentence(u"µx µa µb µc µd".split())
        d = Sentence(u"µ5 µ1 µ2 µ3 µ4".split())
        s2 = self.score.problem.linear_word_count(SentencePair(c, d))

        self.assertGreater(s1, s2)


class TestCacheOfSizeOne(unittest.TestCase):
    def test_calls_N_times(self):
        count = {0: 0}

        def f(x):
            count[0] += 1
            return x
        g = CacheOfSizeOne(f)

        inputs = [0] + range(3) + range(2, 6)
        for x in inputs:  # 1 0 1 1 0 1 1 1
            self.assertEqual(x, g(x))

        self.assertEqual(count[0], 6)


if __name__ == "__main__":
    unittest.main()
