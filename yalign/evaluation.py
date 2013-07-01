# -*- coding: utf-8 -*-

"""
Module to score the accuracy of alignments.
"""

import os
import numpy

from yalign.api import AlignDocuments
from yalign.input_conversion import parallel_corpus_to_documents
from yalign.train_data_generation import training_scrambling_from_documents


def evaluate(parallel_corpus, tu_scorer, gap_penalty, threshold, N=100):
    """
    Retruns statistics for N document alignment trials.
    The documents are generated from the parallel corpus.
        *parallel_corpus: A file object
        *tu_scorer: a TUScore
        *gap_penalty, threshold: parameters for squence alignments
        *N: Number of trials
    """
    results = []
    align_documents = AlignDocuments(tu_scorer, gap_penalty, threshold)
    for idx, docs in enumerate(documents(parallel_corpus)):
        A, B, alignments = training_scrambling_from_documents(*docs)
        predicted_alignments = align_documents(A, B)
        xs = [(a, b) for a, b, _ in predicted_alignments]
        scores = F_score(xs, alignments)
        results.append(scores)
        if idx >= N:
            break
    return _stats(results)


def _stats(xs):
    return dict(max=numpy.amax(xs, 0),
                mean=numpy.mean(xs, 0),
                std=numpy.std(xs, 0))


def F_score(xs, ys, beta=0.1):
    """
    Return the F score described here: http://en.wikipedia.org/wiki/F1_score
    for xs against the sample set ys.

    Change beta to give more weight to precision.
    """
    p = precision(xs, ys)
    r = recall(xs, ys)
    if (p + r) == 0:
        return 0, 0, 0
    b_2 = beta ** 2
    F = (1 + b_2) * (p * r) / (b_2 * p + r)
    return F, p, r


def precision(xs, ys):
    """Precision of xs for sample set ys."""
    return len([x for x in xs if x in ys]) / float(len(xs)) if xs else 0.


def recall(xs, ys):
    """Recall of xs for sample set ys."""
    return len([x for x in xs if x in ys]) / float(len(ys)) if ys else 0.


def documents(parallel_corpus):
    """Provides an endless stream of documents"""
    while True:
        A, B = parallel_corpus_to_documents(parallel_corpus)
        yield A, B


def alignment_percentage(document_a, document_b, model):
    """
    Returns the percentage of alignments of `document_a` and `document_b`
    using the model.
    `document_a` and `document_b` are yalign documents.
    `model` can be a YalignModel or a path to a yalign model.
    The return value it's a float between 0.0 and 100.0
    """
    # FIXME: Use in-memory model instead of path.
    from yalign.yalignmodel import YalignModel
    if isinstance(model, basestring):
        if not os.path.exists(model):
            raise ValueError(u"Invalid model path: {}".format(model))
        path = model
        model = YalignModel()
        model.load(path)
    elif not isinstance(model, YalignModel):
        raise ValueError(u"Invalid model")

    if len(document_a) == 0 and len(document_b) == 0:
        return 1.0

    align = model.align(document_a, document_b)
    align = [x for x in align if x[0] is not None and x[1] is not None]
    ratio = len(align) / float(max(len(document_a), len(document_b)))
    return round(ratio * 100, 2)
