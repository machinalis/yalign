# -*- coding: utf-8 -*-
"""
A module of helper functions for dealing with various inputs.
"""

import re
import csv
import codecs
import random
from itertools import islice
from lxml import etree
from lxml.etree import XMLSyntaxError
from bs4 import BeautifulSoup, UnicodeDammit
from nltk.data import load as nltkload

from yalign.tokenizers import get_tokenizer
from yalign.datatypes import Sentence, SentencePair
from yalign.utils import Memoized

SRT_REGEX = "\d+\n[\d:,]+?\s*-->\s*[\d:,]+?\n(.+?)(:?\n\n|$)"
SRT_REGEX = re.compile(SRT_REGEX.replace("\n", "(?:\n|\r\n)"), re.DOTALL)
SRT_PRE_IGNORE = re.compile("<i>|</i>")
SRT_POST_IGNORE = set(["-"])


MIN_LINES = 20
MAX_LINES = 20
XMLNS = "{http://www.w3.org/XML/1998/namespace}"
STRIP_TAGS_REGEXP = re.compile("(>)(.*)(<)", re.DOTALL)

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
    return Sentence(_tokenizers[language].tokenize(text), text=text)


def text_to_document(text, language="en"):
    """ Returns string text as list of Sentences """
    splitter = _sentence_splitters[language]
    utext = unicode(text, 'utf-8') if isinstance(text, str) else text
    sentences = splitter.tokenize(utext)
    return [tokenize(text, language) for text in sentences]


def html_to_document(html, language="en"):
    """ Returns html text as list of Sentences """
    soup = BeautifulSoup(html, "html5lib")
    text = '\n'.join([tag.get_text() for tag in soup.body.find_all('p')])
    return text_to_document(text, language)


def generate_documents(filepath, m=MIN_LINES, n=MAX_LINES):
    """
    Document generator. Documents are created from the parallel corpus and
    will be between m and n lines long.
    """
    parallel_corpus = iter(codecs.open(filepath, encoding="utf-8"))
    m = m if m > 0 else 1
    N = random.randint(m, n)
    A, B = _next_documents(parallel_corpus, N)
    while len(A) >= m and len(B) >= m:
        yield A, B
        N = random.randint(m, n)
        A, B = _next_documents(parallel_corpus, N)


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
    handler = codecs.open(filepath, encoding="utf-8")
    return _next_documents(handler)


def _next_documents(parallel_corpus, N=None):
    lines_a, lines_b = _split_parallel_corpus(parallel_corpus, N)
    return _document(lines_a), _document(lines_b)


def _document(lines):
    doc = list([Sentence(line.split()) for line in lines])
    for sentence in doc:
        sentence.check_is_tokenized()
    return doc


def _split_parallel_corpus(parallel_corpus, N=None):
    if N is None:
        parallel_corpus = parallel_corpus.readlines()
        n = len(parallel_corpus)
    else:
        n = N * 2
    lines = [x for x in islice(parallel_corpus, n)]
    return list(lines[0:n:2]), list(lines[1:n:2])


def parse_training_file(training_file):
    """
    Reads SentencePairs from a training file.
    """
    labels = None
    data = csv.reader(open(training_file))
    result = []
    for elem in data:
        if labels is None:  # First line contains the labels
            labels = dict((x, elem.index(x)) for x in elem)
            continue

        sentence_a = _sentence_from_csv_elem(elem, "a", labels)
        sentence_b = _sentence_from_csv_elem(elem, "b", labels)
        aligned = elem[labels["aligned"]] == "1"
        result.append(SentencePair(sentence_a, sentence_b, aligned=aligned))

    return result


def _sentence_from_csv_elem(elem, label, labels):
    words = elem[labels[label]].decode("utf-8").split()
    sentence = Sentence(words)
    sentence.check_is_tokenized()
    return sentence


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


def tmx_file_to_documents(filepath, lang_a=None, lang_b=None):
    """
    Converts a tmx file into two lists of Sentences.
    The first for language lang_a and the second for language lang_b.
    """
    inputfile = open(filepath)

    tu = _iterparse(inputfile, "tu").next()
    languages = tuple(_language_from_node(tuv) for tuv in tu.findall("tuv"))
    source, target = languages
    lang_a = source if lang_a is None else lang_a
    lang_b = target if lang_b is None else lang_b
    inputfile.seek(0)
    document_a = []
    document_b = []
    try:
        for tu in _iterparse(inputfile, "tu"):
            sentences = {}
            for tuv in tu.findall("tuv"):
                seg = tuv.find("seg")
                lang = _language_from_node(tuv)
                if lang in languages:
                    sentences[lang] = tokenize(_node_to_sentence(seg), lang)
            document_a.append(sentences[lang_a])
            document_b.append(sentences[lang_b])
    except XMLSyntaxError as error:
    #bug in lxml (see https://bugs.launchpad.net/lxml/+bug/1185701)
        if error.text is not None:
            raise

    return document_a, document_b


def srt_to_document(text, lang="en"):
    """ Convert a string of srt into a list of Sentences. """
    text = UnicodeDammit(text).markup
    d = []
    for m in SRT_REGEX.finditer(text):
        sent = m.group(1)
        sent = SRT_PRE_IGNORE.sub("", sent)
        sent = Sentence(x for x in tokenize(sent, lang)
                        if x not in SRT_POST_IGNORE)
        d.append(sent)
    return d
