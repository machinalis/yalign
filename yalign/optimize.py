from yalign.train import generate_documents, alignments
from yalign.evaluation import F_score
from yalign.nwalign import AlignSequences
from functools import partial
from yalign.tu import TU
from random import uniform


def run_random_sample(parallel_corpus, scorer, n):
    """
    Returns the best value obtained by random sampling.
    Inputs
        -parallel_corpus: A parallel corpus to be used as source of documents.
        -scorer: A trained TUScore
        -n: Number of random samples to be generated
    """
    weight_fn = partial(weight, scorer)
    score_fn = partial(align, documents(parallel_corpus), weight_fn)
    return random_sample(n, score_fn)


def random_sample(n, fn):
    """
    Returns the best gap_penalty and threshold found by random sampling.
    Inputs:
        -n: The number of random samples to generate.
        -fn: Function that returns a score and takes a gap_penalty and threshold.
    Returns:
        (score, gap_penalty, threshold)
    """
    best_result = (0, 0, 0)
    for i in xrange(n):
        gap_penalty = uniform(0, 0.5)
        threshold = uniform(0, 1)
        score = fn(gap_penalty, threshold)
        if score > best_result[0]:
            best_result = score, gap_penalty, threshold
        print best_result
    return best_result


def weight(scorer, a, b):
    """
    A weight function to be used when aligning.
    Inputs:
        -scorer: A trained TUScore
        -a, b: Items to be compared. Each item ia a tuple of three values.
               {relative pos in document}, {sentence}
    """
    distance = abs(a[0] - b[0])
    tu = TU(a[1], b[1], distance)
    score = scorer(tu)[0][0]
    return score


def items(values):
    """
    Takes a list of alignment_index and sentence pairs and converts it to
    a tuple of:
        {relative pos in document}, {sentence}
    """
    return list([((idx + 1) / float(len(values)), x[1])
                for idx, x in enumerate(values)])


def best_alignments(threshold, gap_penalty, A, B, w):
    """
    The best alignment of documents A and B using this threshold, gap_penalty and
    weight function.
    """
    align = AlignSequences(items(A), items(B), w, gap_penalty=gap_penalty)
    return list([(a, b) for a, b, c in align if c < threshold])


def align(reader, weight_fn, gap_penalty, threshold):
    """
    Function to be used when optimizing. Returns the score for the alignment
    of two documents provided by the reader. The weight_fn, gap_penalty and threshold
    are used as inputs to get the best alignment.
    """
    A, B = reader.next()
    guessed_alignments = best_alignments(threshold, gap_penalty, A, B, weight_fn)
    actual_alignments = list(alignments(A, B))
    score = F_score(guessed_alignments, actual_alignments)[0]
    return score


def documents(parallel_corpus):
    """
    Generate an endless sequence of documents from a parallel_corpus.
    """
    while True:
        for A, B in generate_documents(parallel_corpus):
            yield A, B
        parallel_corpus.seek(0)
