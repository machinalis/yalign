# -*- coding: utf-8 -*-


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
    F = (1 + beta ** 2) * (p * r) / ((beta * p) + r)
    return F, p, r


def precision(xs, ys):
    """
    Precision of xs for sample set ys.
    """
    return len([x for x in xs if x in ys]) / float(len(xs)) if xs else 0.


def recall(xs, ys):
    """
    Recall of xs for sample set ys.
    """
    return len([x for x in xs if x in ys]) / float(len(ys)) if ys else 0.
