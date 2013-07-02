# -*- coding: utf-8 -*-

"""
Module to score the accuracy of alignments.
"""

import numpy

from yalign.svm import SVMClassifier
from yalign.api import AlignDocuments
from simpleai.machine_learning import kfold
from yalign.sequencealigner import SequenceAligner
from yalign.input_conversion import parallel_corpus_to_documents
from yalign.train_data_generation import training_scrambling_from_documents, \
                                         training_alignments_from_documents


def evaluate(parallel_corpus, tu_scorer, gap_penalty, threshold, N=100):
    """
    Retruns statistics for N document alignment trials.
    The documents are generated from the parallel corpus.
        *parallel_corpus: A file object
        *tu_scorer: a TUScore
        *gap_penalty, threshold: parameters for squence alignments
        *N: Number of trials
    """
    results = []
    align_documents = AlignDocuments(tu_scorer, gap_penalty, threshold)
    for idx, docs in enumerate(documents(parallel_corpus)):
        A, B, alignments = training_scrambling_from_documents(*docs)
        predicted_alignments = align_documents(A, B)
        xs = [(a, b) for a, b, _ in predicted_alignments]
        scores = F_score(xs, alignments)
        results.append(scores)
        if idx >= N:
            break
    return _stats(results)


def _stats(xs):
    return dict(max=numpy.amax(xs, 0),
                mean=numpy.mean(xs, 0),
                std=numpy.std(xs, 0))


def F_score(xs, ys, beta=0.1):
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


def documents(parallel_corpus):
    """Provides an endless stream of documents"""
    while True:
        A, B = parallel_corpus_to_documents(parallel_corpus)
        yield A, B


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


def word_alignment_percentage(document_a, document_b, model):
    """
    Returns the percentage of word alignments of `document_a` and `document_b`
    `document_a` and `document_b` are yalign documents.
    `model` can be a YalignModel or a path to a yalign model.
    The return value it's a float between 0.0 and 100.0
    """

    wordcount_a = sum([len(sentence) for sentence in document_a])
    wordcount_b = sum([len(sentence) for sentence in document_b])
    wordcount_aligned = 0.0

    if wordcount_a == 0 or wordcount_b == 0:
        return 100.0 if (wordcount_a == 0 and wordcount_b == 0) else 0.0

    sentence_align = model.document_pair_aligner(document_a, document_b)
    sentence_align = [x for x in sentence_align if
                      x[0] is not None and x[1] is not None]

    word_aligner = SequenceAligner(model.word_pair_score, 4.999)
    for pair in sentence_align:
        sentence_a = document_a[pair[0]]
        sentence_b = document_b[pair[1]]
        word_align = word_aligner(sentence_a, sentence_b)
        word_align = [x for x in word_align if
                      x[0] is not None and x[1] is not None]
        wordcount_aligned += len(word_align)

    ratio = wordcount_aligned / min(wordcount_a, wordcount_b)
    return round(ratio * 100.0, 2)


def word_translations_percentage(document_a, document_b, model):
    """
    Returns the percentage of word that are contained in the model's
    dictionary.
    """

    if len(document_a) == 0 or len(document_b) == 0:
        if len(document_a) == 0 and len(document_b) == 0:
            return 100.0
        else:
            return 0.0

    sentence_align = model.document_pair_aligner(document_a, document_b)
    sentence_align = [x for x in sentence_align if
                      x[0] is not None and x[1] is not None]

    word_aligner = SequenceAligner(model.word_pair_score, 4.999)
    count = 0.0
    total = 0.0
    for pair in sentence_align:
        sentence_a = document_a[pair[0]]
        sentence_b = document_b[pair[1]]
        word_align = word_aligner(sentence_a, sentence_b)
        word_align = [x for x in word_align if
                      x[0] is not None and x[1] is not None]

        for word_pair in word_align:
            word_a = sentence_a[word_pair[0]]
            word_b = sentence_b[word_pair[1]]
            if word_a in model.word_pair_score.translations:
                if word_b in model.word_pair_score.translations[word_a]:
                    count += 1.0
            total += 1.0

    ratio = count / total
    return round(ratio * 100.0, 2)


def classifier_precision(document_a, document_b, model):
    if len(document_a) == 0 and len(document_b) == 0:
        return 0.0

    training = training_alignments_from_documents(document_a, document_b)
    problem = model.sentence_pair_score.problem
    score = kfold(training, problem, SVMClassifier)
    return round(score * 100, 2)
