# -*- coding: utf-8 -*-

from yalign.datatypes import Sentence, Document
import nltk
import random
from itertools import islice

def tokenize(text, language="en"):
    """
    Returns a Sentence with Words (ie, a list of unicode objects)
    """
    # FIXME: Implement, using a proper tokenizer
    return Sentence(text.split())


def text_to_document(text, language="en"):
    # FIXME: Implement
    # FIXME: Add multi-language options. See:
    #    en = nltk.data.load('tokenizers/punkt/english.pickle')
    #    sp = nltk.data.load('tokenizers/punkt/spanish.pickle')
    #    pt = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
    return Document(tokenize(sentence, language)
                    for sentence in sentence_splitter(text))


def html_to_document(html, language="en"):
    # FIXME: implement
    pass


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
        for sample in training_alignments_from_documents(A, B):
            yield sample

def documents_from_parallel_corpus(parallel_corpus, m=MIN_LINES, n=MAX_LINES):
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


def _sentences(xs):
    return [tokenize(x) for x in xs]


def _next_documents(reader, N):
    """Read the next documents. Each document will be N lines long."""
    n = N * 2
    lines = [x.decode('utf-8') for x in islice(reader, n)]
    return _sentences(lines[0:n:2]), _sentences(lines[1:n:2])
