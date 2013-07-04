# -*- coding: utf-8 -*-

import re
import csv
import codecs
from lxml import etree
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.data import load as nltkload

from yalign.tokenizers import get_tokenizer
from yalign.datatypes import Sentence, SentencePair

XMLNS = "{http://www.w3.org/XML/1998/namespace}"
STRIP_TAGS_REGEXP = re.compile("(>)(.*)(<)", re.DOTALL)


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
    return not ((word.endswith(".") or word.endswith(",")) and
                word[:-1].isalpha())


def parse_training_file(training_file):
    """
    Parses the file and yields SentencePair objects.
    """
    labels = None
    data = csv.reader(open(training_file))
    for elem in data:
        if labels is None:  # First line contains the labels
            labels = dict((x, elem.index(x)) for x in elem)
            continue

        sentence_a = sentence_from_csv_elem(elem, "a", labels)
        sentence_b = sentence_from_csv_elem(elem, "b", labels)
        aligned = elem[labels["aligned"]] == "1"
        # FIXME: Consider moving this to a test
        assert aligned is True or aligned is False
        yield SentencePair(sentence_a, sentence_b, aligned=aligned)


def sentence_from_csv_elem(elem, label, labels):
    words = elem[labels[label]].decode("utf-8").split()
    position = float(elem[labels["pos " + label]])
    for word in words:
        # FIXME: add harder checks
        if (word.endswith(".") or word.endswith(",")) and word[:-1].isalpha():
            raise ValueError("Word {!r} is not tokenized".format(word))
    return Sentence(words, position=position)


def _language_from_node(node):
    return node.attrib.get(XMLNS + "lang")


def _node_to_sentence(node):
    text = etree.tostring(node)
    match = re.search(STRIP_TAGS_REGEXP, text)
    text = match.group(2) if match else u""
    text = text.replace("\n", " ")
    return text.decode("utf-8")  # Fixme: pick up encoding from file


def _iterparse(input_file, tag=None, events=("end",),
               encoding=None, remove_blank_text=False):
    parser = etree.iterparse(input_file, events=events,
                             tag=tag, encoding=encoding,
                             remove_blank_text=remove_blank_text)
    for _, node in parser:
        yield node
        node.clear()
        while node.getprevious() is not None:
            del node.getparent()[0]


def parse_tmx_file(filepath, lang_a=None, lang_b=None):
    inputfile = open(filepath)

    tu = _iterparse(inputfile, "tu").next()
    languages = tuple(_language_from_node(tuv) for tuv in tu.findall("tuv"))
    source, target = languages
    lang_a = source if lang_a is None else lang_a
    lang_b = target if lang_b is None else lang_b
    inputfile.seek(0)
    document_a = []
    document_b = []
    for tu in _iterparse(inputfile, "tu"):
        sentences = {}
        for tuv in tu.findall("tuv"):
            seg = tuv.find("seg")
            lang = _language_from_node(tuv)
            if lang in languages:
                sentences[lang] = tokenize(_node_to_sentence(seg), lang)
        document_a.append(sentences[lang_a])
        document_b.append(sentences[lang_b])

    return document_a, document_b
