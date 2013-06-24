#!/usr/bin/env python
# coding: utf-8
"""
Module to generate training data.
"""
import re
import random
from itertools import islice

from bs4 import BeautifulSoup
from nltk import sent_tokenize


MIN_LINES = 1
MAX_LINES = 50


def training_samples(parallel_corpus, m=MIN_LINES, n=MAX_LINES):
    """
    Training sample generater. Creates samples from the provided parallel corpus.
        *parallel_corpus: A file object of a file consists of alternating sentences
                          in the languages.

    A sample is a tuple containing:
        {aligned: 0 or 1}, {doc A size}, {index a}, {a}, {doc B size}, {index b}, {b}
    """
    for A, B in documents(parallel_corpus, m, n):
        for sample in generate_samples(A, B):
            yield sample


def documents(parallel_corpus, m=MIN_LINES, n=MAX_LINES):
    """
    Document generator. Documents are created from the parallel corpus and
    will be between m and n lines long.
    """
    m = m if m > 0 else 1
    N = random.randint(m, n)
    A, B = _next_documents(parallel_corpus, N)
    while len(A) >= m and len(B) >= m:
        yield A, B
        N = random.randint(m, n)
        A, B = _next_documents(parallel_corpus, N)


def _next_documents(reader, N):
    """Read the next documents. Each docment will be N lines long."""
    n = N * 2
    lines = [x.decode('utf-8') for x in islice(reader, n)]
    return lines[0:n:2], lines[1:n:2]


def random_align(A, B, perc_to_remove=0):
    """Realign A and B and return the documents along with the alignments """
    if not 0 <= perc_to_remove <= 1:
        raise ValueError("perc_to_remove must be between 0 and 1")
    xs = list(enumerate(A))
    ys = list(enumerate(B))
    xs = _reorder(xs, random_range(len(xs)))
    ys = _reorder(ys, random_range(len(ys)))
    _random_remove(xs, limit=perc_to_remove)
    _random_remove(ys, limit=perc_to_remove)
    alignments = _extract_alignments(xs, ys)
    A = list([x[1] for x in xs])
    B = list([y[1] for y in ys])
    return A, B, alignments


def _random_remove(xs, limit=0.2):
    n = int(len(xs) * random.uniform(0,limit))
    for _ in xrange(n):
        r = random.randint(0, len(xs) - 1)
        xs.pop(r)


def _extract_alignments(xs, ys):
    """
    Returns alignments for lists xs and ys.

    The items in the lists are tuples where each tuple consists of
    a key and value. The alignments are formed by matching the keys.
    If there is no matching key then the item is alignd with None.
    """
    x_dict = dict(xs)
    y_dict = dict(ys)
    alignments = []
    n = max(len(xs), len(ys))
    for idx in xrange(n):
        x = x_dict.get(idx, None)
        y = y_dict.get(idx, None)
        i,j = None, None
        if not x is None:
            i = xs.index((idx, x))
        if not y is None:
            j = ys.index((idx, y))
        if not (i, j) == (None, None):
            alignments.append((i,j))
    alignments.sort()
    return alignments


def generate_samples(A, B):
    """Generates aligned and misaligned samples for documents A and B"""
    if not len(A) == len(B):
        raise ValueError("Documents must be the same size")
    A, B, alignments = random_align(A, B)
    for sample in _aligned_samples(A, B, alignments):
        yield sample
    for sample in _misaligned_samples(A, B, alignments):
        yield sample


def _sample(A, B, alignment, aligned=True):
    """Helper function to build sample"""
    i, j = alignment
    sample = [int(aligned)]
    sample += [len(A), i, A[i].strip().encode('utf-8')]
    sample += [len(B), j, B[j].strip().encode('utf-8')]
    return tuple(sample)


def _aligned_samples(A, B, alignments):
    for alignment in alignments:
        yield _sample(A, B, alignment)


def _misaligned_samples(A, B, alignments):
    misalignments = []
    n = len(alignments)
    if n > 1:
        while len(misalignments) < n:
            i = random.randint(0, len(A) - 1)
            j = random.randint(0, len(B) - 1)
            if not (i, j) in alignments and not (i,j) in misalignments:
                misalignments.append((i,j))
                yield _sample(A, B, (i,j), aligned=False)


def _reorder(xs, indexes):
    """Reorder list xs by indexes"""
    if not len(indexes) == len(xs):
        raise ValueError("xs and indexes must be the same size")
    ys = [None] * len(xs)
    for i, j in enumerate(indexes):
        ys[j] = xs[i]
    return ys


def random_range(N, span=10):
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


BAD_CHARS_PATTERN = re.compile('(\n|\t)+')


def text_to_corpus(text):
    """Extract sentences split by newlines from plain text."""
    return [re.sub(BAD_CHARS_PATTERN, ' ', x.strip()) for x in sent_tokenize(text)]


def html_to_corpus(html_text):
    """Extract sentences split by newlines from html."""
    soup = BeautifulSoup(html_text)
    text = soup.body.get_text()
    return text_to_corpus(text)

