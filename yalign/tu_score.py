#!/usr/bin/env python
# coding: utf-8

"""
Trains the TU Classifier to score the alignment of two sentences

Usage:
    tu_score [options] score <source> <target> <distance>
    tu_score [options] train <dataset>

Options:
  -e --eval        Evaluates a training using 10-fold
  -h --help        Show this screen.
"""

import sys
import csv
import math
from docopt import docopt

from svm import SVMClassifier
from word_score import score_word
from nwalign import AlignSequences
from simpleai.machine_learning import is_attribute, Attribute
from simpleai.machine_learning import ClassificationProblem


CLASSIFIER_FILEPATH = "sentence_classifier.pickle"
__classifier = None


class TU(object):
    def __init__(self, src, tgt, distance, aligned=None):
        """
        Creates a Translation Unit with source, target, the
        distance between this two and if it's aligned or not.
        """
        self.src = src
        self.tgt = tgt
        self.distance = distance
        self.aligned = aligned


class SentenceProblem(ClassificationProblem):
    @is_attribute
    def word_score(self, tu):
        alignment = AlignSequences(tu.tgt.split(), tu.src.split(), score_word)
        word_score = [x[2] for x in alignment]
        return abs(sum(word_score) / max(len(tu.src), len(tu.tgt)))

    @is_attribute
    def position_difference(self, tu):
        return tu.distance

    @is_attribute
    def word_length_difference(self, tu):
        src_words = len(tu.src.split())
        tgt_words = len(tu.tgt.split())
        return normalization(abs(src_words - tgt_words))

    def target(self, tu):
        return tu.aligned

    def __getstate__(self):
        # A quick and dirty fix to allow pickle-ability
        attributes = [a for a in self.attributes
                      if not hasattr(a.function, "is_attribute")]
        return {"attributes": attributes}

    def __setstate__(self, d):
        # A quick and dirty fix to allow pickle-ability
        attrs = []
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, "is_attribute"):
                attr = Attribute(method, method.name)
                attrs.append(attr)
        self.attributes = attrs + d["attributes"]
        self.attributes.sort(key=lambda attr: attr.name)


def normalization(x, unit=2):
    """
    Maps positive real numbers to [0, 1).
    Strictly increasing function, continuous function.
    """
    x = math.log(x + 1.0, unit)
    x = 1.0 / (x + 1.0)
    return 1.0 - x


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

        src = elem[labels["src"]]
        tgt = elem[labels["tgt"]]
        src_pos = float(elem[labels["src idx"]]) / float(elem[labels["src N"]])
        tgt_pos = float(elem[labels["tgt idx"]]) / float(elem[labels["tgt N"]])
        dist = abs(src_pos - tgt_pos)
        aligned = elem[labels["aligned"]]
        yield TU(src, tgt, dist, aligned)


def train_and_save_classifier(dataset_filepath, out_filepath):
    """
    Trains the classifier using the information of `dataset_filepath`
    and saves it to `out_filepath`

    The information in `dataset_filepath` must be in YAML format
    and must contain the following data:
        * src: source sentence
        * tgt: target sentence
        * dist: distance between src and tgt relative to the document
        * aligned: if the source and target sentences are aligned
    """

    training_data = parse_training_data(dataset_filepath)
    classifier = SVMClassifier(training_data, SentenceProblem())
    classifier.save(out_filepath)


class ScoreSentence(object):
    def __init__(self):
        self.classifier = SVMClassifier.load(CLASSIFIER_FILEPATH)
        self.dimention = len(self.classifier.problem.attributes)
        self.min_bound = 0
        self.max_bound = 2 * self.dimention

    def __call__(self, ut):
        """
        Returns the score of a sentence.
        The result will always be a in (self.min_bound, self.max_bound)
        """
        score = self.classifier.score(tu)
        constant = math.sqrt(self.dimention)
        # We add sqrt(n) to avoid negative numbers.
        result = score + constant

        assert self.min_bound <= result <= self.max_bound
        return result


if __name__ == "__main__":
    args = docopt(__doc__)

    if args["score"]:
        tu = TU(args["<source>"], args["<target>"], float(args["<distance>"]))
        print ScoreSentence(tu)
    elif args["train"]:
        dataset_filepath = args["<dataset>"]
        out_filepath = CLASSIFIER_FILEPATH

        if args["--eval"]:
            from simpleai.machine_learning import kfold
            training_data = parse_training_data(dataset_filepath)
            score = kfold(training_data, SentenceProblem(), SVMClassifier)
            message = "Classifier precision {:.3f}% (10-fold crossvalidation)"
            print >> sys.stderr, message.format(score * 100)
            exit(0)

        print >> sys.stderr, "Starting training"
        train_and_save_classifier(dataset_filepath, out_filepath)
        print >> sys.stderr, "Training done"
