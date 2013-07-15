# -*- coding: utf-8 -*-

import re
import csv
import codecs
import random
from itertools import islice
from lxml import etree
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.data import load as nltkload

from yalign.tokenizers import get_tokenizer
from yalign.datatypes import Sentence, SentencePair

MIN_LINES = 20
MAX_LINES = 20
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
    document = []
    sentence_splitter = _sentence_splitters[language]
    sentences = sentence_splitter.tokenize(text)
    total_sentences = float(len(sentences))
    for i, sentence_text in enumerate(sentences):
        sentence = tokenize(sentence_text, language)
        sentence.position = i / total_sentences
        document.append(sentence)
    return document


def html_to_document(html, language="en"):
    soup = BeautifulSoup(html, "html5lib")
    text = '\n'.join([tag.get_text() for tag in soup.find_all('p')])
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
    handler = iter(codecs.open(filepath, encoding="utf-8"))
    return _next_documents(handler)


def _next_documents(parallel_corpus, N=None):
    lines_a, lines_b = _split_parallel_corpus(parallel_corpus, N)
    return _document(lines_a), _document(lines_b)


def _document(lines):
    doc = list([Sentence(line.split()) for line in lines])
    for idx, sentence in enumerate(doc):
        sentence.position = float(idx) / len(doc)
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
    sentence = Sentence(words, position=position)
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


def host_and_page(url):
    url = url.split('//')[1]
    parts = url.split('/')
    host = parts[0]
    page = "/".join(parts[1:])
    return host, '/' + page


def read_from_url(url):
    import httplib
    host, page = host_and_page(url)
    conn = httplib.HTTPConnection(host)
    conn.request("GET", page)
    response = conn.getresponse()
    return response.read()
