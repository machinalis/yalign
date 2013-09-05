# -*- coding: utf-8 -*-

import os
import random
import tempfile
import unittest

from yalign.datatypes import Sentence
from yalign.evaluation import F_score
from yalign.wordpairscore import WordPairScore
from yalign.sequencealigner import SequenceAligner
from yalign.sentencepairscore import SentencePairScore
from yalign.input_conversion import parallel_corpus_to_documents
from yalign.train_data_generation import training_scrambling_from_documents, \
    training_alignments_from_documents
from yalign.yalignmodel import YalignModel, random_sampling_maximizer, \
    best_threshold, apply_threshold


class TestYalignModel(unittest.TestCase):
    def setUp(self):
        random.seed(hash("Y U NO?"))
        base_path = os.path.dirname(os.path.abspath(__file__))
        word_scores = os.path.join(base_path, "data", "test_word_scores_big.csv")
        parallel_corpus = os.path.join(base_path, "data", "parallel-en-es.txt")
        A, B = parallel_corpus_to_documents(parallel_corpus)
        A = A[:15]
        B = B[:15]
        self.alignments = list(training_alignments_from_documents(A, B))
        self.A, self.B, self.correct_alignments = \
                                 list(training_scrambling_from_documents(A, B))
        # Word score
        word_pair_score = WordPairScore(word_scores)
        # Sentence Score
        sentence_pair_score = SentencePairScore()
        sentence_pair_score.train(self.alignments, word_pair_score)
        # Yalign model
        self.min_ = sentence_pair_score.min_bound
        self.max_ = sentence_pair_score.max_bound
        gap_penalty = (self.min_ + self.max_) / 2.0
        document_aligner = SequenceAligner(sentence_pair_score, gap_penalty)
        self.model = YalignModel(document_aligner, 1)

    def test_save_file_created(self):
        tmp_folder = tempfile.mkdtemp()
        self.model.save(tmp_folder)
        model_path = os.path.join(tmp_folder, "aligner.pickle")
        metadata_path = os.path.join(tmp_folder, "metadata.json")
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(metadata_path))

    def test_save_load_and_align(self):
        doc1 = [Sentence([u"House"]),
                Sentence([u"asoidfhuioasgh"])]
        doc2 = [Sentence([u"Casa"])]
        result_before_save = self.model.align(doc1, doc2)

        # Save
        tmp_folder = tempfile.mkdtemp()
        self.model.save(tmp_folder)

        # Load
        new_model = YalignModel.load(tmp_folder)
        result_after_load = new_model.align(doc1, doc2)
        self.assertEqual(result_before_save, result_after_load)
        self.assertEqual(self.model.threshold, new_model.threshold)
        self.assertEqual(self.model.document_pair_aligner.penalty,
                         new_model.document_pair_aligner.penalty)

    def test_reasonable_alignment(self):
        doc1 = [Sentence([u"House"]),
                Sentence([u"asoidfhuioasgh"])]
        doc2 = [Sentence([u"Casa"])]
        result = self.model.align(doc1, doc2)
        result = [(list(x), list(y)) for x, y in result]
        self.assertIn((list(doc1[0]), list(doc2[0])), result)

    def test_optimize_gap_penalty_and_threshold_finishes(self):
        self.model.optimize_gap_penalty_and_threshold(self.A, self.B,
                                                      self.correct_alignments)

    def test_optimize_gap_penalty_and_threshold_is_best(self):
        def evaluate(penalty, threshold):
            self.model.document_pair_aligner.penalty = penalty
            self.model.threshold = threshold
            predicted = self.model.align_indexes(self.A, self.B)
            return F_score(predicted, self.correct_alignments)[0]

        random.seed(hash("12345"))
        self.model.optimize_gap_penalty_and_threshold(self.A, self.B,
                                                      self.correct_alignments)
        best_score = evaluate(self.model.document_pair_aligner.penalty,
                              self.model.threshold)
        for _ in xrange(50):
            penalty = random.uniform(self.min_, self.max_ / 2.0)
            threshold = random.uniform(self.min_, self.max_)
            score = evaluate(penalty, threshold)
            self.assertGreaterEqual(best_score, score)


class TestOptimizers(unittest.TestCase):
    def test_random_sampling_maximizer_maximizes(self):
        def F(x):
            return x * x * x + 1
        random.seed(hash("Knock knock motherfucker"))
        score, x = random_sampling_maximizer(F, -1, 1, n=100)
        self.assertGreater(x, 0.9)
        self.assertGreater(score, F(0.9))

    def test_random_sampling_maximizer_more_is_better(self):
        def F(x):
            return -(x ** 0.5)
        random.seed(hash("Want some? get some!"))
        score_20, _ = random_sampling_maximizer(F, 5, 10, n=20)
        score_100, _ = random_sampling_maximizer(F, 5, 10, n=100)
        self.assertGreater(score_100, score_20)

    def test_best_threshold1(self):
        best_threshold([], [(0, 0, 0), (1, 1, 1)])

    def test_best_threshold2(self):
        score, threshold = best_threshold([(0, 0)],
                                          [(0, 0, 0), (1, 1, 1)])
        self.assertLess(threshold, 1)
        self.assertGreater(score, 0)

    def test_best_threshold3(self):
        random.seed(hash("Son de plata y de acero, sivlerrrrhaaawks!"))
        real = [(i, i, random.random()) for i in xrange(100)]
        guess = random.sample(real, 50)
        real = [(a, b) for a, b, _ in real]
        best, _ = best_threshold(real, guess)
        for i in range(100):
            threshold = random.random()
            score = F_score(apply_threshold(guess, threshold), real)[0]
            self.assertLessEqual(score, best)


if __name__ == "__main__":
    unittest.main()
