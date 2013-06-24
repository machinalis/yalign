# -*- coding: utf-8 -*-

import re
from random import choice, randint
from itertools import islice
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from scramble import shuffle, remove


###
### FIXME: Module-wise, separate stuff between train_data_generation and
###        input_conversion. Adapt to new datatypes.
###


def training_alignments_from_documents(document_a, document_b, mix=0.5):
    """
    Returns an iterable of `Alignment`s to be used for training.
    `document_a` and `document_b` are aligned documents.
    `mix` is the proportion of correct alignments generated.
    For example, a ratio of 1 means all alignments generated are correct
    alignments, a ratio of 0 means that all alignments generated are incorrect.
    """
    # FIXME: implement
    pass


def training_scrambling_from_documents(document_a, document_b):
    """
    Returns a tuple `(scrambled_a, scrambled_b, correct_alignments)`
    `scrambled_a` is a scrambled version of document_a.
    `scrambled_b` is a scrambled version of document_b.
    `correct_alignments` are all the correct sentence alignments that exist
    between `scrambled_a` and `scrambled_b`.
    """
    # FIXME: implement
    pass


def read_lines(parallel_corpus, n):
    """
    Read n lines of source and target in parallel corpus.
    Returns tuple of A and B lines.
    Each line consists of an index and sentence.
    The index can be used to match the sentences as
    matching sentences have the same index number.
    Eg.. returns [(1, 'hello'),(2, 'goodbye')],[(1, 'hola'), (2, 'adios')]
    """
    N = n * 2
    xs = list(x.decode('utf-8').strip()
              for x in islice(parallel_corpus, N))
    A = enumerate(xs[0:N:2])
    B = enumerate(xs[1:N:2])
    return list(A), list(B)


def generate_documents(parallel_corpus):
    """
    Returns two scrambled documents derived from the parallel_corpus.
    Each line is a tuple of an index and sentence. The index can be used
    to match pairs of sentences together.
    """
    n = randint(10, 30)
    A, B = read_lines(parallel_corpus, n)
    while A:
        A = scramble(A)
        B = scramble(B)
        yield A, B
        n = randint(10, 30)
        A, B = read_lines(parallel_corpus, n)


def samples(source):
    """
    Generate aligned and non-aligned training samples.
    Sample output for target t and source s is:
    {aligned: 0 or 1}, {s doc length}, {s index}, s, {t doc length}, {t index}, t
    """
    for A, B in generate_documents(source):
        for sample in aligned_samples(A, B):
            yield sample
        for sample in non_aligned_samples(A, B):
            yield sample


def aligned_samples(A, B):
    for idx, pair in enumerate(aligned_sentences(A, B)):
        a, b = pair
        yield 1, len(A), A.index(a), a[1], len(B), B.index(b), b[1]


def non_aligned_samples(A, B):
    N = max(len(A), len(B))
    for idx in xrange(N):
        a, b = choice(A), choice(B)
        if not a[0] == b[0]:
            yield 0, len(A), A.index(a), a[1], len(B), B.index(b), b[1]


def alignments(A, B):
    """
    Returns list of A and B alignments.
    Eg.. [(0, 0), (1, 2), (2, 1)]
    """
    for a, b in aligned_sentences(A, B):
        yield A.index(a), B.index(b)


def aligned_sentences(A, B):
    A_dict = dict(A)
    B_dict = dict(B)
    indexes = list(A_dict.keys())
    indexes.sort()
    for idx in indexes:
        a = A_dict.get(idx, None)
        b = B_dict.get(idx, None)
        if a and b:  # FIXME: 0 is a valid index and gets confused with None
            yield (idx, a), (idx, b)


def scramble(xs):
    ys = []
    n = 0
    r = randint(5, 10)
    x = xs[0:r]
    while x:
        n = n + r
        #remove(x, randint(0, 2))
        shuffle(x, randint(1, 3))
        ys += x
        r = randint(5, 10)
        x = xs[n:n + r]
    return ys

BAD_CHARS_PATTERN = re.compile('(\n|\t)+')


def text_to_corpus(text):
    """
    Extract sentences split by newlines from plain text.
    """
    return [re.sub(BAD_CHARS_PATTERN, ' ', x.strip()) for x in sent_tokenize(text)]


def html_to_corpus(html_text):
    """
    Extract sentences split by newlines from html.
    """
    soup = BeautifulSoup(html_text)
    text = soup.body.get_text()
    return text_to_corpus(text)
