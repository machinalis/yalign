# -*- coding: utf-8 -*-

"""
Module to evaluate the accuracy.
"""

import numpy
from simpleai.machine_learning import kfold

from yalign.svm import SVMClassifier
from yalign.input_conversion import generate_documents
from yalign.train_data_generation import training_scrambling_from_documents
from yalign.train_data_generation import training_alignments_from_documents
from collections import defaultdict

def evaluate(parallel_corpus, model, N=100):
    """
    Returns statistics for N document alignment trials.
    The documents are generated from the parallel corpus.
        * parallel_corpus: A file object
        * sentence_pair_score: A function that scores sentences alignment
        * gap_penalty, threshold: parameters for squence alignments
        * N: Number of trials
    """

    results = []
    for idx, docs in enumerate(generate_documents(parallel_corpus)):
        A, B, alignments = training_scrambling_from_documents(*docs)
        predicted_alignments = model.align_indexes(A, B)
        scores = F_score(predicted_alignments, alignments)
        results.append(scores)
        if idx >= N - 1:
            break
    return _stats(results)


def _stats(xs):
    return dict(max=numpy.amax(xs, 0),
                mean=numpy.mean(xs, 0),
                std=numpy.std(xs, 0))


def F_score(xs, ys, beta=0.01):
    """
    Return the F score described here: http://en.wikipedia.org/wiki/F1_score
    for xs against the sample set ys.

    Change beta to give more weight to precision.
    """
    p = precision(xs, ys)
    r = recall(xs, ys)
    if (p + r) == 0:
        return 0, 0, 0
    b_2 = beta ** 2
    F = (1 + b_2) * (p * r) / (b_2 * p + r)
    return F, p, r


def precision(xs, ys):
    """Precision of xs for sample set ys."""
    return len([x for x in xs if x in ys]) / float(len(xs)) if xs else 0.


def recall(xs, ys):
    """Recall of xs for sample set ys."""
    return len([x for x in xs if x in ys]) / float(len(ys)) if ys else 0.


def alignment_percentage(document_a, document_b, model):
    """
    Returns the percentage of alignments of `document_a` and `document_b`
    using the model.
    `document_a` and `document_b` are yalign documents.
    `model` can be a YalignModel or a path to a yalign model.
    The return value it's a float between 0.0 and 100.0
    """

    if len(document_a) == 0 and len(document_b) == 0:
        return 100.0

    align = model.align(document_a, document_b)
    align = [x for x in align if x[0] is not None and x[1] is not None]
    ratio = len(align) / float(max(len(document_a), len(document_b)))
    return round(ratio * 100, 2)


def classifier_precision(document_a, document_b, model):
    """
    Runs a ten-fold validation on the classifier and returns
    a value between 0 and 100 meaning how good it is.
    """
    if len(document_a) == 0 and len(document_b) == 0:
        return 0.0

    training = training_alignments_from_documents(document_a, document_b)
    problem = model.sentence_pair_score.problem
    score = kfold(training, problem, SVMClassifier)
    return round(score * 100, 2)


def correlation(classifier, dataset=None):
    """
    Calculates the correlation of the attributes on a classifier.
    For more information see:
        - http://en.wikipedia.org/wiki/Correlation_and_dependence
    """
    if dataset is None:
        assert hasattr(classifier, "dataset")
        dataset = classifier.dataset

    result = {}
    answers = []
    attributes = defaultdict(list)

    for data in dataset:
        answers.append(int(classifier.problem.target(data)))
        for i, attr in enumerate(classifier.attributes):
            attributes[i].append(attr(data))

    answers_std = numpy.std(answers)
    for i in xrange(len(attributes)):
        cov = numpy.cov(attributes[i], answers)[0][1]
        std = numpy.std(attributes[i]) * answers_std
        if std == 0:
            corr = numpy.nan
        else:
            corr = cov / std
        result[classifier.attributes[i]] = corr
    return result
