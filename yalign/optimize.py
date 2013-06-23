#!/usr/bin/env python
# coding: utf-8
"""
Module for optimizing parameters.
"""
import random
from functools import partial

from yalign.evaluation import F_score
from yalign.nwalign import AlignSequences
from yalign.tu import TU
from yalign.train import random_align, documents as train_documents


def optimize(parallel_corpus, tu_scorer, N=100):
    """
    Returns the best F score, gap_penalty and threshold found by random sampling.
        *parallel corpus: A file object for a file of alternaing lines in the languages
                          of the model to be optimized.
        *tu_scorer: A TUScore to be used when scoring tu's.
        *N: Number of random sampling iterations.
    """
    best = 0, 0, 0
    weight = partial(_weight, tu_scorer)
    for idx, docs in enumerate(_documents(parallel_corpus)):
        A, B, alignments = random_align(*docs) 
        gap_penalty = random.uniform(0,1)
        predicted_alignments = AlignSequences(_items(A), _items(B), weight, gap_penalty)
        score, threshold = _optimize_threshold(gap_penalty, 
                                               alignments, 
                                               predicted_alignments)
        if score > best[0]:
            best = score, gap_penalty, threshold
        if idx >= N: break
    return best 


def _optimize_threshold(gap_penalty, real_alignments, predicted_alignments):
    """Returns the best F score and threshold value for this gap_penalty"""
    best = 0, 1
    for threshold in _costs(predicted_alignments):
        xs = [(a, b) for a, b, c in predicted_alignments if c <= threshold]
        score = F_score(xs, real_alignments)[0]
        if score > best[0]:
            best = score, threshold
    return best 


def _costs(alignments):
    """Costs from alignments sorted by highest cost"""
    costs = [c for _, _, c in alignments]
    costs.sort(reverse=True)
    for cost in costs:
        yield cost


def _weight(tu_scorer, a, b):
    """Retruns the tu_score for items a and b"""
    distance = abs(a[1] - b[1])
    tu = TU(a[0], b[0], distance)
    return tu_scorer(tu)[0][0]


def _items(xs):
    pos = lambda idx: float(idx) / len(xs) 
    return [(val, pos(idx)) for idx, val in enumerate(xs)]


def _documents(parallel_corpus):
    """Provides an endless stream of documents"""
    while True:
        for A,B in train_documents(parallel_corpus):
            yield A,B
        parallel_corpus.seek(0)
