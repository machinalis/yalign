# -*- coding: utf-8 -*-

import csv
from yalign.datatypes import Sentence, Alignment


def parse_training_file(training_file):
    """
    Parses the file and yields Alignment objects.
    """
    labels = None
    data = csv.reader(open(training_file))
    for elem in data:
        if labels is None:  # First line contains the labels
            labels = dict((x, elem.index(x)) for x in elem)
            continue

        sentence_a = sentence_from_csv_elem(elem, "src", labels)
        sentence_b = sentence_from_csv_elem(elem, "tgt", labels)
        aligned = elem[labels["aligned"]] == "1"
        # FIXME: Consider moving this to a test
        assert aligned is True or aligned is False
        yield Alignment(sentence_a, sentence_b, are_really_aligned=aligned)


def sentence_from_csv_elem(elem, label, labels):
    words = elem[labels[label]].decode("utf-8").split()
    idx = float(elem[labels[label + " idx"]])
    N = float(elem[labels[label + " N"]])
    position = idx / N
    for word in words:
        # FIXME: add harder checks
        if (word.endswith(".") or word.endswith(",")) and word[:-1].isalpha():
            raise ValueError("Word {!r} is not tokenized".format(word))
    return Sentence(words, position=position)
