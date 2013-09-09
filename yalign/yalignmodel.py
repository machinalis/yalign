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
from yalign.input_conversion import tmx_file_to_documents, \
    parallel_corpus_to_documents
from yalign.train_data_generation import training_alignments_from_documents, \
                                         training_scrambling_from_documents


OPTIMIZE_SAMPLE_SET_SIZE = 100
RANDOM_SAMPLING_ITERATIONS = 20


def basic_model(corpus_filepath, word_scores_filepath,
                lang_a=None, lang_b=None):
    """
    Creates and trains a `YalignModel` with the basic configuration and
    default values.

    `corpus_filepath` is the path to a parallel corpus used for training,
    it can be:
        - a csv file with two sentences and alignement information, or
        - a tmx file with correct alignments (a regular parallel corpus), or
        - a text file with interleaved sentences (one line in language A, the
          next in language B)

    `word_scores_filepath` is the path to a csv file (possibly gzipped) with
    word dictionary data. (for ex. "house,casa,0.91").

    `lang_a` and `lang_b` are requiered for the tokenizer in the case of a tmx
    file. In the other cases is not necesary because it's assumed that the
    words are already tokenized.
    """
    # Word score
    word_pair_score = WordPairScore(word_scores_filepath)

    if corpus_filepath.endswith(".tmx"):
        A, B = tmx_file_to_documents(corpus_filepath, lang_a, lang_b)
    else:
        A, B = parallel_corpus_to_documents(corpus_filepath)
    alignments = training_alignments_from_documents(A, B)

    sentence_pair_score = SentencePairScore()
    sentence_pair_score.train(alignments, word_pair_score)
    # Yalign model
    metadata = {"lang_a": lang_a, "lang_b": lang_b}
    gap_penalty = 0.49
    threshold = 1.0
    document_aligner = SequenceAligner(sentence_pair_score, gap_penalty)
    model = YalignModel(document_aligner, threshold, metadata=metadata)
    A, B, correct = training_scrambling_from_documents(A[:OPTIMIZE_SAMPLE_SET_SIZE], B[:OPTIMIZE_SAMPLE_SET_SIZE])
    model.optimize_gap_penalty_and_threshold(A, B, correct)
    return model


class YalignModel(object):
    """
    Main Yalign class.
    It provides methods to train a alignment model, to load a model from a
    folder and to align two documents.
    """
    def __init__(self, document_pair_aligner=None,
                       threshold=None, metadata=None):
        """
        Barebones instantiation method. If no argument is supplied the instance
        created is not suited to align documents.

        To train, you better check `basic_model` first. To use an existing
        model better check `YalignModel.load` first.

        `document_pair_aligner` is a `SequenceAligner` instance capable of
        aligning `Sentence`s.

        `threshold` is a number such that only `Sentence`s scoring lower
        than this number (lower is better) are returned by `align`.

        `metadata` is a `dict` that can be used to store any data that the
        user of `YalignModel` considers useful (for instance the languages
        that the model accepts). Metadata will be also saved as a json file
        along with the model serialization, allowing faster retrieval of
        metadata (because is no necesary to load the whole model) for any other
        user software.

        """
        self.document_pair_aligner = document_pair_aligner
        self.threshold = threshold
        self.metadata = MetadataHelper(metadata)

    @classmethod
    def load(cls, model_directory):
        """
        This method to loads an existing YalignModel from the path to the
        folder where it's contained.
        """
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
        """
        Try to detect aligned sentences from the comparable documents
        `document_a` and `document_b`.
        The returned alignments are expected to meet the F-measure for which
        the model was trained for.
        """
        alignments = self.align_indexes(document_a, document_b)
        return [(document_a[a], document_b[b]) for a, b in alignments]

    def align_indexes(self, document_a, document_b):
        """
        Same as `align` but returning indexes in documents instead of
        sentences.
        """
        alignments = self.document_pair_aligner(document_a, document_b)
        alignments = pre_filter_alignments(alignments)
        return apply_threshold(alignments, self.threshold)

    def save(self, model_directory):
        """
        Store a serialization of a YalignModel instance in a given folder.
        Metadata is stored in a separate file.
        """
        metadata = os.path.join(model_directory, "metadata.json")
        aligner = os.path.join(model_directory, "aligner.pickle")
        pickle.dump(self.document_pair_aligner, open(aligner, "w"))
        self.metadata.threshold = self.threshold
        self.metadata.penalty = self.document_pair_aligner.penalty
        json.dump(dict(self.metadata), open(metadata, "w"), indent=4)

    def optimize_gap_penalty_and_threshold(self, document_a, document_b,
                                                              real_alignments):
        """
        Given documents `document_a` and `document_b` (not necesarily aligned)
        and the `real_alignments` for that documents train the YalignModel
        instance to maximize the target F-measure (the quality measure).

        `real_alignments` is a list of indexes (i, j) of `document_a` and
        `document_b` respectively indicating that those sentences are aligned.
        Pairs not included in `real_alignments` are assumed to be wrong
        alignments.
        """
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
