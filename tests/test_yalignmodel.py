# -*- coding: utf-8 -*-

import os
import mock
import tempfile
import unittest

from yalign.datatypes import Sentence
from yalign.yalignmodel import YalignModel
from yalign.wordpairscore import WordPairScore
from yalign.sequencealigner import SequenceAligner
from yalign.input_parsing import parse_training_file
from yalign.sentencepairscore import SentencePairScore


class TestYalignModel(unittest.TestCase):
    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        word_scores = os.path.join(base_path, "data", "test_word_scores.csv")
        self.alignments_file = os.path.join(base_path, "data", "test_training.csv")
        alignments = parse_training_file(self.alignments_file)
        gap_penalty = 0.499
        threshold = 1

        # Word score
        word_pair_score = WordPairScore(word_scores)
        # Sentence Score
        sentence_pair_score = SentencePairScore()
        sentence_pair_score.train(alignments, word_pair_score)
        # Yalign model
        document_aligner = SequenceAligner(sentence_pair_score, gap_penalty)
        self.model = YalignModel(document_aligner, threshold)

    def test_save_file_created(self):
        tmp_folder = tempfile.mkdtemp()
        self.model.save(tmp_folder)
        model_path = os.path.join(tmp_folder, "aligner.pickle")
        metadata_path = os.path.join(tmp_folder, "metadata.json")
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(metadata_path))

    def test_save_load_and_align(self):
        doc1 = [Sentence([u"House"], position=0),
                Sentence([u"asoidfhuioasgh"], position=1)]
        doc2 = [Sentence(u"Casa", position=0)]
        result_before_save = self.model.align(doc1, doc2)

        # Save
        tmp_folder = tempfile.mkdtemp()
        self.model.save(tmp_folder)

        # Load
        new_model = YalignModel()
        new_model.load(tmp_folder)
        result_after_load = new_model.align(doc1, doc2)

        self.assertEqual(result_before_save, result_after_load)
        self.assertEqual(len(result_after_load), 2)
        self.assertIn((0, 0), result_after_load)

    @mock.patch("yalign.optimize.optimize")
    def test_optimize_is_called(self, mock_optimize):
        mock_optimize.return_value = 1337, 1338, 1339
        self.model.optimize_gap_penalty_and_threshold("/some/path")
        self.assertTrue(mock_optimize.called)
        self.assertEqual(self.model.document_pair_aligner.penalty, 1338)
        self.assertEqual(self.model.threshold, 1339)


if __name__ == "__main__":
    unittest.main()
