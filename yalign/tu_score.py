#!/usr/bin/env python
# coding: utf-8

"""
Trains the TU Classifier to score the alignment of two sentences

Usage:
    tu_score [options] score <filepath> <source> <target> <distance>
    tu_score [options] train <dataset> <word_scores> <outfile>

Options:
  -e --eval        Evaluates a training using 10-fold
  -h --help        Show this screen.
"""

import sys
import csv
import math
from docopt import docopt

from svm import SVMClassifier
from word_score import ScoreWord
from nwalign import AlignSequences
from simpleai.machine_learning import is_attribute
from simpleai.machine_learning import ClassificationProblem


class TU(object):
    def __init__(self, src, tgt, distance, aligned=None):
        """
        Creates a Translation Unit with source, target, the
        distance between this two and if it's aligned or not.
        """

        if not isinstance(src, unicode) or not isinstance(tgt, unicode):
            raise ValueError("Source and target must be unicode")
        if not src or not tgt:
            raise ValueError("Source or target empty")
        if not isinstance(distance, float) or not 0.0 <= distance <= 1.0:
            raise ValueError("Invalid distance: {} ({})".format(distance))

        self.src = src
        self.tgt = tgt
        self.distance = distance
        self.aligned = aligned


class SentenceProblem(ClassificationProblem):
    def __init__(self, word_score_filepath):
        super(SentenceProblem, self).__init__()
        self.score_word = ScoreWord(word_score_filepath)

    @is_attribute
    def word_score(self, tu):
        src_words = tu.tgt.lower().split()
        tgt_words = tu.src.lower().split()
        alignment = AlignSequences(src_words, tgt_words, self.score_word, 0.49)
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

    @is_attribute
    def uppercase_words_difference(self, tu):
        up_source_words = len([x for x in tu.src.split() if x.isupper()])
        up_target_words = len([x for x in tu.src.split() if x.isupper()])
        return normalization(abs(up_source_words - up_target_words))

    @is_attribute
    def capitalized_words_difference(self, tu):
        cap_source_words = len([x for x in tu.src.split() if x.istitle()])
        cap_target_words = len([x for x in tu.src.split() if x.istitle()])
        return normalization(abs(cap_source_words - cap_target_words))

    def target(self, tu):
        return tu.aligned


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


class ScoreTU(object):
    def __init__(self, filepath):
        self.classifier = SVMClassifier.load(filepath)
        self.dimention = len(self.classifier.problem.attributes)
        self.min_bound = 0
        self.max_bound = 2 * math.sqrt(self.dimention)

    def __call__(self, tu):
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
        filepath = args["<filepath>"]
        classifier = ScoreTU(filepath)
        print classifier(tu)
    elif args["train"]:
        dataset_filepath = args["<dataset>"]
        out_filepath = args["<outfile>"]
        word_scores = args["<word_scores>"]

        if args["--eval"]:
            from simpleai.machine_learning import kfold
            training_data = parse_training_data(dataset_filepath)
            problem = SentenceProblem(word_scores)
            score = kfold(training_data, problem, SVMClassifier)
            message = "Classifier precision {:.3f}% (10-fold crossvalidation)"
            print >> sys.stderr, message.format(score * 100)
            exit(0)

        print >> sys.stderr, "Starting training"
        train_and_save_classifier(dataset_filepath, word_scores, out_filepath)
        print >> sys.stderr, "Training done"
