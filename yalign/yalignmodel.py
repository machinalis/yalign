# -*- coding: utf-8 -*-

import os
import json
import random
try:
    import cPickle as pickle
except ImportError:
    import pickle

from yalign.evaluation import F_score
from yalign.wordpairscore import WordPairScore
from yalign.sequencealigner import SequenceAligner
from yalign.sentencepairscore import SentencePairScore
from yalign.input_conversion import parse_training_file, tmx_file_to_documents, \
    parallel_corpus_to_documents
from yalign.train_data_generation import training_alignments_from_documents, \
                                         training_scrambling_from_documents


OPTIMIZE_SAMPLE_SET_SIZE = 100
RANDOM_SAMPLING_ITERATIONS = 20


def basic_model(corpus_filepath, word_scores_filepath,
                lang_a=None, lang_b=None, optimize=False,
                gap_penalty=0.49, threshold=1):
    # Word score
    word_pair_score = WordPairScore(word_scores_filepath)

    if corpus_filepath.endswith(".csv"):
        alignments = parse_training_file(corpus_filepath)
        if optimize:
            raise ValueError("Cannot optmize using a csv corpus!")
    else:
        if corpus_filepath.endswith(".tmx"):
            A, B = tmx_file_to_documents(corpus_filepath, lang_a, lang_b)
        else:
            A, B = parallel_corpus_to_documents(corpus_filepath)
        alignments = training_alignments_from_documents(A, B)

    sentence_pair_score = SentencePairScore()
    sentence_pair_score.train(alignments, word_pair_score)
    # Yalign model
    metadata = {"lang_a": lang_a, "lang_b": lang_b}
    document_aligner = SequenceAligner(sentence_pair_score, gap_penalty)
    model = YalignModel(document_aligner, threshold, metadata=metadata)
    if optimize:
        A, B, correct = training_scrambling_from_documents(A[:OPTIMIZE_SAMPLE_SET_SIZE], B[:OPTIMIZE_SAMPLE_SET_SIZE])
        model.optimize_gap_penalty_and_threshold(A, B, correct)
    return model


class YalignModel(object):
    def __init__(self, document_pair_aligner=None,
                       threshold=None, metadata=None):
        self.document_pair_aligner = document_pair_aligner
        self.threshold = threshold
        self.metadata = MetadataHelper(metadata)

    @classmethod
    def load(cls, model_directory):
        model = cls()
        metadata = os.path.join(model_directory, "metadata.json")
        aligner = os.path.join(model_directory, "aligner.pickle")
        model.metadata.update(json.load(open(metadata)))
        model.document_pair_aligner = pickle.load(open(aligner))
        model.document_pair_aligner.penalty = model.metadata.penalty
        model.threshold = model.metadata.threshold
        return model

    @property
    def sentence_pair_score(self):
        return self.document_pair_aligner.score

    @property
    def word_pair_score(self):
        return self.sentence_pair_score.word_pair_score

    def align(self, document_a, document_b):
        alignments = self.align_indexes(document_a, document_b)
        return [(document_a[a], document_b[b]) for a, b in alignments]

    def align_indexes(self, document_a, document_b):
        """
        Try to recover aligned sentences from the comparable documents
        `document_a` and `document_b`.
        The returned alignments are expected to meet the F-measure for which
        the model was trained for.
        """
        alignments = self.document_pair_aligner(document_a, document_b)
        alignments = pre_filter_alignments(alignments)
        return apply_threshold(alignments, self.threshold)

    def save(self, model_directory):
        metadata = os.path.join(model_directory, "metadata.json")
        aligner = os.path.join(model_directory, "aligner.pickle")
        pickle.dump(self.document_pair_aligner, open(aligner, "w"))
        self.metadata.threshold = self.threshold
        self.metadata.penalty = self.document_pair_aligner.penalty
        json.dump(dict(self.metadata), open(metadata, "w"), indent=4)

    def optimize_gap_penalty_and_threshold(self, document_a, document_b,
                                                              real_alignments):
        def F(x):
            return score_with_best_threshold(self.document_pair_aligner,
                                             document_a, document_b,
                                             x,
                                             real_alignments)
        _, gap_penalty = random_sampling_maximizer(F, 0, 0.2)
        self.document_pair_aligner.penalty = gap_penalty
        alignments = self.document_pair_aligner(document_a, document_b)
        alignments = pre_filter_alignments(alignments)
        _, threshold = best_threshold(real_alignments, alignments)
        self.threshold = threshold


class MetadataHelper(dict):
    def __init__(self, metadata):
        if isinstance(metadata, dict):
            self.update(metadata)
        elif metadata is not None:
            raise ValueError("Invalid metadata initial values")

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("No attribute by that name: '{}'".format(key))

    def __setattr__(self, key, value):
        self[key] = value


def pre_filter_alignments(alignments):
    return [(a, b, c) for a, b, c in alignments if a is not None and
                                                   b is not None]


def apply_threshold(alignments, threshold):
    return [(a, b) for a, b, c in alignments if c <= threshold]


def best_threshold(real_alignments, predicted_alignments):
    """Returns the best F score and threshold value for this gap_penalty"""
    if not predicted_alignments:
        raise ValueError("predicted_alignments cannot be empty")
    best = -1, None
    for _, _, threshold in predicted_alignments:
        xs = apply_threshold(predicted_alignments, threshold)
        score = F_score(xs, real_alignments)[0]
        if score > best[0]:
            best = score, threshold
    return best


def score_with_best_threshold(aligner, xs, ys, gap_penalty, real_alignments):
    predicted_alignments = aligner(xs, ys, penalty=gap_penalty)
    predicted_alignments = pre_filter_alignments(predicted_alignments)
    if not predicted_alignments:
        return 0
    score, threshold = best_threshold(real_alignments, predicted_alignments)
    return score


def random_sampling_maximizer(F, min_, max_, n=None):
    if n is None:
        n = RANDOM_SAMPLING_ITERATIONS
    if n < 1:
        raise ValueError("n must be 1 or more")
    x = random.uniform(min_, max_)
    best = F(x), x
    for _ in xrange(n - 1):
        x = random.uniform(min_, max_)
        score = F(x)
        if score > best[0]:
            best = score, x
    return best
