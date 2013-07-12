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
from yalign.input_conversion import parse_training_file, parse_tmx_file, \
    parallel_corpus_to_documents
from yalign.train_data_generation import training_alignments_from_documents


def basic_model(corpus_filepath, word_scores_filepath,
                lang_a=None, lang_b=None, optimize=False,
                gap_penalty=0.49, threshold=1):
    # Word score
    word_pair_score = WordPairScore(word_scores_filepath)

    if corpus_filepath.endswith(".csv"):
        alignments = parse_training_file(corpus_filepath)
    else:
        if corpus_filepath.endswith(".tmx"):
            A, B = parse_tmx_file(corpus_filepath, lang_a, lang_b)
        else:
            A, B = parallel_corpus_to_documents(corpus_filepath)
        alignments = training_alignments_from_documents(A, B)

    sentence_pair_score = SentencePairScore()
    sentence_pair_score.train(alignments, word_pair_score)
    # Yalign model
    document_aligner = SequenceAligner(sentence_pair_score, gap_penalty)
    model = YalignModel(document_aligner, threshold)
    return model


class YalignModel(object):
    def __init__(self, document_pair_aligner=None, threshold=None):
        self.document_pair_aligner = document_pair_aligner
        self.threshold = threshold
        self.metadata = MetadataHelper()

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

    def load(self, model_directory, load_data=True):
        metadata = os.path.join(model_directory, "metadata.json")
        aligner = os.path.join(model_directory, "aligner.pickle")
        self.metadata.update(json.load(open(metadata)))
        self.threshold = self.metadata.threshold
        self.document_pair_aligner = pickle.load(open(aligner))

    def save(self, model_directory):
        metadata = os.path.join(model_directory, "metadata.json")
        aligner = os.path.join(model_directory, "aligner.pickle")
        pickle.dump(self.document_pair_aligner, open(aligner, "w"))
        self.metadata.threshold = self.threshold
        json.dump(dict(self.metadata), open(metadata, "w"))

    def optimize_gap_penalty_and_threshold(self, document_a, document_b,
                                                              real_alignments):
        def F(x):
            return score_with_best_threshold(self.document_pair_aligner,
                                             document_a, document_b,
                                             x,
                                             real_alignments)
        min_ = self.sentence_pair_score.min_bound
        max_ = self.sentence_pair_score.max_bound
        _, gap_penalty = random_sampling_maximizer(F, min_, max_ / 2.0, n=10)
        self.document_pair_aligner.penalty = gap_penalty
        alignments = self.document_pair_aligner(document_a, document_b)
        alignments = pre_filter_alignments(alignments)
        _, threshold = best_threshold(real_alignments, alignments)
        self.threshold = threshold


class MetadataHelper(dict):
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


def random_sampling_maximizer(F, min_, max_, n=20):
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
