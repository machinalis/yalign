# -*- coding: utf-8 -*-

"""
Module for optimizing parameters.
"""

import random
from yalign.evaluation import F_score, documents
from yalign.train_data_generation import training_scrambling_from_documents
from yalign.api import AlignDocuments


def optimize(parallel_corpus, tu_scorer, N=100):
    """
    Returns the best F score, gap_penalty and threshold found by random sampling.
        *parallel corpus: A file object for a file of alternaing lines in the languages
                          of the model to be optimized.
        *tu_scorer: A TUScore to be used when scoring tu's.
        *N: Number of random sampling iterations.
    """
    best = 0, 0, 0
    align_documents = AlignDocuments(tu_scorer)
    for idx, docs in enumerate(documents(parallel_corpus)):
        A, B, alignments = training_scrambling_from_documents(*docs)
        gap_penalty = random.uniform(0,1)
        predicted_alignments = align_documents(A, B, gap_penalty=gap_penalty, threshold=1)
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
