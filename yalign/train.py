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


def training_samples(parallel_corpus):
    """
    Training sample generater. Creates samples from the provided parallel corpus.
        *parallel_corpus: A file object of a file consists of alternating sentences
                          in the languages.

    A sample is a tuple containing:
        {aligned: 0 or 1}, {doc A size}, {index a}, {a}, {doc B size}, {index b}, {b}
    """
    for A, B in documents(parallel_corpus):
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


def random_align(A, B):
    """Realign A and B and return the documents along with the alignments """
    alignments = _random_alignments(len(A))
    A, B = _realign(A, B, alignments)
    return A, B, alignments


def generate_samples(A, B):
    """Generates aligned and non aligned samples for documents A and B"""
    assert len(A) == len(B), "Documents must be the same size"
    A, B, alignments = random_align(A, B)
    for sample in _aligned_samples(A, B, alignments):
        yield sample
    for sample in _non_aligned_samples(A, B, alignments):
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

    
def _non_aligned_samples(A, B, alignments):
    """Generate non aligned samples"""
    non_alignments = []
    n = len(alignments)
    if n > 1:  
        while len(non_alignments) < n:
            i = random.randint(0, len(A) - 1) 
            j = random.randint(0, len(B) - 1)
            if not (i, j) in alignments and not (i,j) in non_alignments: 
                non_alignments.append((i,j))
                yield _sample(A, B, (i,j), aligned=False)


def _realign(xs, ys, alignments):
    """Reorders lists xs,ys according to the alignments"""
    xs = _reorder(xs, [i for i, _ in alignments])
    ys = _reorder(ys, [j for _, j in alignments])
    return xs, ys


def _reorder(xs, indexes):
    """Reorder list xs by indexes"""
    assert len(indexes) == len(xs), "xs and indexes must be the same size"
    ys = [None] * len(xs)
    for i, j in enumerate(indexes):
        ys[j] = xs[i]
    return ys 


def _random_alignments(n):
    """Create n random alignments. Highest index will be n -1."""
    xs = random_range(n)
    ys = random_range(n)
    return zip(xs,ys)


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

