# -*- coding: utf-8 -*-

from nltk.data import load as nltkload
import random
from bs4 import BeautifulSoup
from itertools import islice
from yalign.datatypes import Sentence, Document
from yalign.tokenizers import get_tokenizer
from collections import defaultdict


class Memoized(defaultdict):
    def __missing__(self, key):
        x = self.default_factory(key)
        self[key] = x
        return x


_tokenizers = Memoized(lambda lang: get_tokenizer(lang))
_punkt = {"en": "tokenizers/punkt/english.pickle",
                  "es": "tokenizers/punkt/spanish.pickle",
                  "pt": "tokenizers/punkt/portuguese.pickle"}
_sentence_splitters = Memoized(lambda lang: nltkload(_punkt[lang]))


def tokenize(text, language="en"):
    """
    Returns a Sentence with Words (ie, a list of unicode objects)
    """
    if not isinstance(text, unicode):
        raise ValueError("Can only tokenize unicode strings")
    return Sentence(_tokenizers[language].tokenize(text))


def text_to_document(text, language="en"):
    sentence_splitter = _sentence_splitters[language]
    return Document(tokenize(sentence, language)
                    for sentence in sentence_splitter.tokenize(text))


def html_to_document(html, language="en"):
    soup = BeautifulSoup(html)
    text = soup.body.get_text()
    return text_to_document(text, language)


MIN_LINES = 1
MAX_LINES = 50


def training_samples(parallel_corpus, m=MIN_LINES, n=MAX_LINES):
    """
    Training sample generater. Creates samples from the provided
    parallel corpus.
        * parallel_corpus: A file object of a file consists of
                           alternating sentences in the languages.

    A sample is a tuple containing:
        {aligned: 0 or 1}, {doc A size}, {index a}, {a},
        {doc B size}, {index b}, {b}
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
