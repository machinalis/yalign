#!/usr/bin/env python
# coding: utf-8

import csv
from simpleai.machine_learning import is_attribute
from simpleai.machine_learning import ClassificationProblem

from yalign.tu import TU
from yalign.svm import SVMClassifier
from yalign.align_sequences import align_sequences
from yalign.weightfunctions import WordScore


class SentenceProblem(ClassificationProblem):
    def __init__(self, word_score_filepath):
        super(SentenceProblem, self).__init__()
        self.score_word = WordScore(word_score_filepath)

    @is_attribute
    def word_score(self, tu):
        src_words = [x.lower() for x in tu.source_words]
        tgt_words = [x.lower() for x in tu.target_words]
        gap = 0.5  # If gap > 0.5 then the returned value could be > 1.
        alignment = align_sequences(src_words, tgt_words, self.score_word, gap)
        word_score = [x[2] for x in alignment]
        return abs(sum(word_score) / max(len(tu.src), len(tu.tgt)))

    @is_attribute
    def position_difference(self, tu):
        return tu.distance

    @is_attribute
    def word_length_difference(self, tu):
        a = len(tu.source_words)
        b = len(tu.target_words)
        return ratio(a, b)

    @is_attribute
    def uppercase_words_difference(self, tu):
        a = len([x for x in tu.source_words if x.isupper()])
        b = len([x for x in tu.target_words if x.isupper()])
        return ratio(a, b)

    @is_attribute
    def capitalized_words_difference(self, tu):
        a = len([x for x in tu.source_words if x.istitle()])
        b = len([x for x in tu.target_words if x.istitle()])
        return ratio(a, b)

    def target(self, tu):
        return tu.aligned


def ratio(a, b):
    if max(a, b) == 0:
        return 1.0
    return min(a, b) / float(max(a, b))


def parse_training_data(dataset_filepath):
    """
    Parses the file and yields a TU object.
    """

    labels = None
    data = csv.reader(open(dataset_filepath))
    for elem in data:
        if labels is None:  # First line contains the labels
            labels = dict((x, elem.index(x)) for x in elem)
            continue

        src = elem[labels["src"]].decode("utf-8")
        tgt = elem[labels["tgt"]].decode("utf-8")
        src_pos = float(elem[labels["src idx"]]) / float(elem[labels["src N"]])
        tgt_pos = float(elem[labels["tgt idx"]]) / float(elem[labels["tgt N"]])
        dist = abs(src_pos - tgt_pos)
        aligned = elem[labels["aligned"]]
        yield TU(src, tgt, dist, aligned)


def train_and_save_classifier(dataset_filepath, word_scores, out_filepath):
    """
    Trains the classifier using the information of `dataset_filepath`
    and saves it to `out_filepath`.

    The information in `dataset_filepath` must be in YAML format
    and must contain the following data:
        * src: source sentence
        * tgt: target sentence
        * dist: distance between src and tgt relative to the document
        * aligned: if the source and target sentences are aligned
    """

    training_data = parse_training_data(dataset_filepath)
    classifier = SVMClassifier(training_data, SentenceProblem(word_scores))
    classifier.save(out_filepath)
