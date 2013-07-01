# -*- coding: utf-8 -*-

import os
import random
try:
    import cPickle as pickle
except ImportError:
    import pickle

from yalign.wordpairscore import WordPairScore
from yalign.sequencealigner import SequenceAligner
from yalign.input_parsing import parse_training_file
from yalign.sentencepairscore import SentencePairScore


# FIXME: this class is untried, complete
class YalignModel(object):
    def __init__(self, document_pair_aligner=None, threshold=None):
        self.document_pair_aligner = document_pair_aligner
        self.threshold = threshold
        self.metadata = MetadataHelper()

    @property
    def sentence_pair_score(self):
        return self.document_pair_aligner.score

    @property
    def sentence_pair_aligner(self):
        return self.sentence_pair_score.sentence_pair_aligner

    @property
    def word_pair_score(self):
        return self.sentence_pair_aligner.score

    def align(self, document_a, document_b):
        """
        Try to recover aligned sentences from the comparable documents
        `document_a` and `document_b`.
        The returned alignments are expected to meet the F-measure for which
        the model was trained for.
        """
        alignments = self.document_pair_aligner(document_a, document_b)
        return [(a, b) for a, b, score in alignments if score < self.threshold]

    def load(self, model_directory, load_data=True):
        metadata = os.path.join(model_directory, "metadata.json")
        aligner = os.path.join(model_directory, "aligner.pickle")
        self.metadata.update(pickle.load(open(metadata)))
        self.threshold = self.metadata.threshold
        self.document_pair_aligner = pickle.load(open(aligner))

    def save(self, model_directory):
        metadata = os.path.join(model_directory, "metadata.json")
        aligner = os.path.join(model_directory, "aligner.pickle")
        pickle.dump(self.document_pair_aligner, open(aligner, "w"))
        self.metadata.threshold = self.threshold
        pickle.dump(dict(self.metadata), open(metadata, "w"))

    def optimize_gap_penalty_and_threshold(self, parallel_corpus):
        from yalign.optimize import optimize
        score, gap_penalty, threshold = optimize(parallel_corpus,
                                                 self.sentence_pair_aligner)
        self.document_pair_aligner.penalty = gap_penalty
        self.threshold = threshold


class MetadataHelper(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("No attribute by that name: '{}'".format(key))

    def __setattr__(self, key, value):
        self[key] = value


def basic_model(word_scores_filepath, training_filepath,
                gap_penalty=0.49, threshold=1):
    """
    Returns a model with the default score functions.
    """
    # Word score
    word_pair_score = WordPairScore(word_scores_filepath)
    # Sentence Score
    alignments = parse_training_file(training_filepath)
    sentence_pair_score = SentencePairScore()
    sentence_pair_score.train(alignments, word_pair_score)
    # Yalign model
    document_aligner = SequenceAligner(sentence_pair_score, gap_penalty)
    model = YalignModel(document_aligner, threshold)
    return model
