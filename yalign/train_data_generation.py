#!/usr/bin/env python
# coding: utf-8
"""
Module to generate training data.
"""
import random
from datatypes import SentencePair


def training_alignments_from_documents(document_a, document_b):
    """
    Returns an iterable of SentencePairs to be used for training.
    The inputs `document_a` and `document_b` are both lists of
    Sentences made from a parallel corpus.
    """
    if not len(document_a) == len(document_b):
        raise ValueError("Documents must be the same size")
    document_a, document_b, alignments = \
                    training_scrambling_from_documents(document_a, document_b)
    for sample in _aligned_samples(document_a, document_b, alignments):
        yield sample
    for sample in _misaligned_samples(document_a, document_b, alignments):
        yield sample


def training_scrambling_from_documents(document_a, document_b):
    """
    Returns a tuple `(scrambled_a, scrambled_b, correct_alignments)` where:
        * `scrambled_a` is a scrambled version of document_a.
        * `scrambled_b` is a scrambled version of document_b.
        * `correct_alignments` are all the correct sentence alignments that exist
           between `scrambled_a` and `scrambled_b`.
    """
    xs = list(enumerate(document_a))
    ys = list(enumerate(document_b))
    xs = _reorder(xs, _random_range(len(xs)))
    ys = _reorder(ys, _random_range(len(ys)))
    alignments = _extract_alignments(xs, ys)
    A = [x[1] for x in xs]
    B = [y[1] for y in ys]
    return A, B, alignments


def _extract_alignments(xs, ys):
    """
    Returns alignments for lists xs and ys.

    The items in the lists are tuples where each tuple consists of
    a key and value. The alignments are formed by matching the keys.
    If there is no matching key then the item is aligned with None.
    """
    x_dict = dict(xs)
    y_dict = dict(ys)
    alignments = []
    n = max(len(xs), len(ys))
    for idx in xrange(n):
        x = x_dict.get(idx, None)
        y = y_dict.get(idx, None)
        i, j = None, None
        if not x is None:
            i = xs.index((idx, x))
        if not y is None:
            j = ys.index((idx, y))
        if not (i, j) == (None, None):
            alignments.append((i, j))
    alignments.sort()
    return alignments


def _aligned_samples(A, B, alignments):
    for i, j in alignments:
        yield SentencePair(A[i], B[j], aligned=True)


def _misaligned_samples(A, B, alignments):
    misalignments = []
    if len(alignments) > 1:
        while len(misalignments) < len(alignments):
            i = random.randint(0, len(A) - 1)
            j = random.randint(0, len(B) - 1)
            if not (i, j) in alignments and not (i, j) in misalignments:
                misalignments.append((i, j))
                yield SentencePair(A[i], B[j], aligned=False)


def _reorder(xs, indexes):
    """Reorder list xs by indexes"""
    if not len(indexes) == len(xs):
        raise ValueError("xs and indexes must be the same size")
    ys = [None] * len(xs)
    for i, j in enumerate(indexes):
        ys[j] = xs[i]
    return ys


def _random_range(N, span=10):
    """
    Returns a list of N integers.
    The span determines the length of the sections
    that are shuffled in the list.

    Eg.. If the span is 10 then every group of
         10 items will be shuffled.
    """
    span = span if span > 1 else 1
    xs = []
    n = 0
    while n < N:
        r = random.randint(1, span)
        n = min(n + r, N)
        ys = range(len(xs), n)
        random.shuffle(ys)
        xs += ys
    return xs
