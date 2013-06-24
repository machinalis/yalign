# -*- coding: utf-8 -*-

from yalign.datatypes import Sentence, Document
import nltk


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
