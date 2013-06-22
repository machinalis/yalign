from yalign.train import generate_documents, alignments
from yalign.evaluation import F_score
from yalign.sequencealigner import SequenceAligner
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
        score, threshold = fn(gap_penalty)
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


def align(reader, w, gap_penalty):
    """
    Function to be used when optimizing.
    Returns the highest score obtained by the lowest threshold value on an
    alignment performed with the weight function w and the gap_penalty.
    The documents for this alignment are provided by the reader.
    """
    A, B = reader.next()
    actual_alignments = list(alignments(A, B))
    aligner = SequenceAligner(w, gap_penalty)
    xs = aligner(items(A), items(B))
    costs = [c for a, b, c in xs]
    costs.sort(reverse=True)

    best_threshold = 1
    best_score = 0
    for threshold in costs:
        guessed_alignments = [(a, b) for a, b, c in xs if c < threshold]
        score = F_score(guessed_alignments, actual_alignments)[0]
        if score > best_score:
            best_score = score
            best_threshold = threshold
        if score == 0:
            break
    return best_score, best_threshold


def documents(parallel_corpus):
    """
    Generate an endless sequence of documents from a parallel_corpus.
    """
    while True:
        for A, B in generate_documents(parallel_corpus):
            yield A, B
        parallel_corpus.seek(0)
