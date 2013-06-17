from yalign.train import generate_documents, alignments
from yalign.evaluation import F_score
from yalign.nwalign import AlignSequences
import numpy as np
from scipy.optimize import anneal
from functools import partial
from yalign.tu import TU


def optimize(parallel_corpus, scorer):
    x = np.array([0.8])
    weight_fn = partial(weight, scorer)
    score_fn = partial(align, documents(parallel_corpus), weight_fn)
    print anneal(score_fn, x, full_output=True)


def weight(scorer, a, b):
    distance = abs(a[0] - b[0])
    tu = TU(a[2], b[2], distance)
    return scorer(tu)[0][0]


def best_alignments(threshold, gap_penalty, src, tgt, w):
    src = list([((idx + 1) / float(len(src)), idx, x[1])
                for idx, x in enumerate(src)])
    tgt = list([((idx + 1) / float(len(tgt)), idx, x[1])
                for idx, x in enumerate(tgt)])
    print len(src), len(tgt)
    align = AlignSequences(src, tgt, w, gap_penalty, minimize=True)
    return list([(a, b) for a, b, c in align if threshold > 0])


def align(reader, weight_fn, gap_penalty):
    src, tgt = reader.next()
    xs = best_alignments(0.5, gap_penalty, src, tgt, weight_fn)
    ys = list(alignments(src, tgt))
    score = F_score(xs, ys)[0]
    print gap_penalty, len(src), len(tgt), score
    return score


def documents(source):
    while True:
        for src, tgt in generate_documents(source):
            yield src, tgt
        source.seek(0)

if __name__ == "__main__":
    from yalign.weightfunctions import TUScore
    parallel_corpus = open('../cc/motorola.en-es')
    scorer = TUScore('data/svm.pickle')
    weight_fn = partial(weight, scorer)
    docs = documents(parallel_corpus)
    src, tgt = docs.next()
    xs = best_alignments(0.5, 1, src, tgt, weight_fn)
    ys = list(alignments(src,tgt))
    score = F_score(xs, ys)
    print len(xs), xs
    print len(ys), ys
    print score
