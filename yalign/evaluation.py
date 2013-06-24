# -*- coding: utf-8 -*-

"""
Module to score the accuracy of alignments.
"""

import numpy as np

from yalign.train_data_generation import documents as train_documents
from yalign.train_data_generation import random_align
from yalign.api import AlignDocuments


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
        A, B, alignments = random_align(*docs)
        predicted_alignments = align_documents(A, B)
        xs = [(a, b) for a, b, _ in predicted_alignments]
        scores = F_score(xs, alignments)
        results.append(scores)
        if idx >= N:
            break
    return _stats(results)


def _stats(xs):
    return dict(max=np.amax(xs, 0), mean=np.mean(xs, 0), std=np.std(xs, 0))


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
        for A, B in train_documents(parallel_corpus):
            yield A, B
        parallel_corpus.seek(0)
