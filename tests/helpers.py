# -*- coding: utf-8 -*-

import os
import tempfile
from yalign.sentencepairscore import SentencePairScore
from yalign.wordpairscore import WordPairScore
from yalign.input_parsing import parse_training_file


def default_sentence_pair_score():
    base_path = os.path.dirname(os.path.abspath(__file__))
    word_scores = os.path.join(base_path, "data", "test_word_scores.csv")
    _, classifier_filepath = tempfile.mkstemp()
    training_file = os.path.join(base_path, "data", "test_training.csv")
    pairs = parse_training_file(training_file)
    classifier = SentencePairScore()
    classifier.train(pairs, WordPairScore(word_scores))
    classifier.save(classifier_filepath)
    return classifier, classifier_filepath
