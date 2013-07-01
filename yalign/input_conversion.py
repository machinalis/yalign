# -*- coding: utf-8 -*-

import codecs
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.data import load as nltkload

from yalign.datatypes import Sentence
from yalign.tokenizers import get_tokenizer


class Memoized(defaultdict):
    def __missing__(self, key):
        x = self.default_factory(key)
        self[key] = x
        return x


_punkt = {
    "en": "tokenizers/punkt/english.pickle",
    "es": "tokenizers/punkt/spanish.pickle",
    "pt": "tokenizers/punkt/portuguese.pickle"
}
_tokenizers = Memoized(lambda lang: get_tokenizer(lang))
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
    return [tokenize(sentence, language)
            for sentence in sentence_splitter.tokenize(text)]


def html_to_document(html, language="en"):
    soup = BeautifulSoup(html)
    text = soup.body.get_text()
    return text_to_document(text, language)


def parallel_corpus_to_documents(filepath):
    """
    Transforms a parallel corpus file format into two
    documents.
    The Parallel corpus has:

        * One sentences per line.
        * One line of each language.
        * Sentences are tokenized and tokens are space separated.
        * The file encoding is UTF-8

    For example:

        This is a sentence .
        Esto es una oración .
        And this , my friend , is another .
        Y esta , mi amigo , es otra .

    """

    handler = iter(codecs.open(filepath, encoding="utf-8"))
    document_a = []
    document_b = []
    total_lines = 0

    while True:
        try:
            line_a = next(handler)
            line_b = next(handler)
        except StopIteration:
            break

        document_a.append(Sentence(line_a.split()))
        document_b.append(Sentence(line_b.split()))
        total_lines += 1

    i = 0.0
    for a, b in zip(document_a, document_b):
        position = i / total_lines
        a.position = position
        b.position = position
        i += 1.0

        for word_a, word_b in zip(a, b):
            message = u"Word {!r} is not tokenized"
            if not _is_tokenized(word_a):
                raise ValueError(message.format(word_a))
            if not _is_tokenized(word_b):
                raise ValueError(message.format(word_b))

    return document_a, document_b


def _is_tokenized(word):
    """
    Runs some checks to see if the word is tokenized.
    Note: if this functions returns True doesn't mean is really tokenized, but
    if returns False you know it's not tokenized propperly.
    """

    # FIXME: add harder checks
    return not ((word.endswith(".") or word.endswith(",")) and \
                word[:-1].isalpha())
