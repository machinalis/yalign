#!/usr/bin/env python
# coding: utf-8
"""
Generate alignment and non alignment samples from a parallel corpus.

Usage:
    optimize <parallel-corpus> <scorer>
"""

from docopt import docopt
from yalign.optimize import optimize
from yalign.weightfunctions import TUScore

if __name__ == "__main__":
    args = docopt(__doc__)
    parallel_corpus = open(args["<parallel-corpus>"])
    scorer = TUScore(args["<scorer>"])
    #optimize(parallel_corpus, scorer)
