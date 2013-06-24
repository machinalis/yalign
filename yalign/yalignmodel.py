# -*- coding: utf-8 -*-
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import random


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
        `document1` and `document2`.
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

    # FIXME: Consider that this optimization should be trained with multiple
    #        documents, not just a single pair.
    # FIXME: This is pseudo-code. It must be moved to the optimize module.
    def optimize_gap_penalty_and_threshold(self, document_a,
                                           document_b, correct_alignments):
        N = 20
        b = self.sentence_pair_score.min_bound
        a = self.sentence_pair_score.max_bound - b
        observations = []
        while len(observations) != N:
            penalty = a * 0.5 * random.random() + b
            self.document_pair_aligner.penalty = penalty
            alignments = self.document_pair_aligner(document_a, document_b)
            # FIXME: optimal_threshold is not implemented
            score, threshold = optimize.optimal_threshold(alignments,
                                                          correct_alignments)
            observations.append((score, penalty, threshold))
        score, penalty, threshold = max(observations)
        self.document_pair_aligner.penalty = penalty
        self.threshold = threshold


class MetadataHelper(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("No attribute by that name: '{}'".format(key))

    def __setattr__(self, key, value):
        self[key] = value