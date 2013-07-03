# -*- coding: utf-8 -*-

"""
"""

import os
import numpy
import unittest
from yalign.svm import correlation
from yalign.wordpairscore import WordPairScore
from yalign.sentencepairscore import SentencePairScore
from yalign.input_conversion import parallel_corpus_to_documents
from yalign.train_data_generation import training_alignments_from_documents


class TestSVM(unittest.TestCase):
    def test_correlation_values(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        parallel_corpus = os.path.join(base_path, "data", "parallel-en-es.txt")
        word_scores = os.path.join(base_path, "data", "test_word_scores_big.csv")
        A, B = parallel_corpus_to_documents(parallel_corpus)
        self.alignments = [x for x in training_alignments_from_documents(A, B)]
        # Word score
        word_pair_score = WordPairScore(word_scores)
        # Sentence Score
        sentence_pair_score = SentencePairScore()
        sentence_pair_score.train(self.alignments, word_pair_score)

        cor = correlation(sentence_pair_score.classifier)
        for attr, value in cor.iteritems():
            if value is not numpy.nan:
                self.assertTrue(-1 <= value <= 1)


if __name__ == "__main__":
    unittest.main()
