from yalign.train import generate_documents, alignments
from yalign.evaluation import F_score
from yalign.nwalign import AlignSequences
from functools import partial
from yalign.tu import TU
from random import uniform


def run_random_sample(parallel_corpus, scorer, n):
    weight_fn = partial(weight, scorer)
    score_fn = partial(align, documents(parallel_corpus), weight_fn)
    return random_sampler(n, score_fn)


def random_sampler(n, fn):
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
    distance = abs(a[0] - b[0])
    tu = TU(a[2], b[2], distance)
    score = scorer(tu)[0][0]
    return score


def items(values):
    return list([((idx + 1) / float(len(values)), idx, x[1])
                for idx, x in enumerate(values)])


def best_alignments(threshold, gap_penalty, src, tgt, w):
    align = AlignSequences(items(src), items(tgt), w, gap_penalty=gap_penalty)
    return list([(a, b) for a, b, c in align if c < threshold])


def align(reader, weight_fn, gap_penalty, threshold):
    src, tgt = reader.next()
    xs = best_alignments(threshold, gap_penalty, src, tgt, weight_fn)
    ys = list(alignments(src, tgt))
    score = F_score(xs, ys)[0]
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
    run_random_sample(parallel_corpus, scorer, 1000)
