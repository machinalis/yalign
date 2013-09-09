Implementation
==============

Yalign is implemented using:

 - A sentence similarity metric. Given two sentences it produces a rough
   estimate (a number between 0 and 1) of how likely are those two sentences
   to be a translation of each other.
 - A sequece aligner, such that given two documents (a list of sentences) it
   produces an alignment which maximizes the sum of the individual
   (per sentence pair) similarities.

So Yalign's main algorithm is actually a pretty wrapper to a standard sequence
alignment algorithm.

For the sequence alignment Yalign uses a variation of the
`Needleman-Wunch algorithm <http://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm>`_
to find an optimal alignment between the sentences in two given documents.
On the good side, the algorithm has polynomial time worst case complexity and it
produces an optimal alignment.
On the bad side it can't handle alignments that
cross each other or alignments from two sentences into a single one (even tough
is possible to modify the current implementation to handle those cases).

Since the sentence similarity is a computationally expensive operation,
the mentioned "variation" on the Needleman-Wunch algorithm consists in using
the `A* <http://en.wikipedia.org/wiki/A*_search_algorithm>`_ to explore the
search space instead of using the classical dynamic programming aproach (which
would always requiere `N * M` calls to the sentence similarity metric).

After the alignment, only sentences that have a high probability of being
translations are included in the final alignment. Ie, the result is filtered
in order to deliver high quality alignments. To do this, a threshold value is
used such that if the sentence similarity metric is bad enough that pair is
excluded.
 

For the sentence similarity metric the algorithm uses a statistical
classifier's likelihood output and adapts it into the 0-1 range.

The classifier is trained to determine if a pair of sentences are translations
of each other or not (a binary value). The particular classifier used for this
project is a
`Support Vector Machine <http://en.wikipedia.org/wiki/Support_vector_machine>`_.
Besides being excelent classifiers, SVMs can provide a distance to the
separation hyperplane during classification, and this distance can be easily
modified using a
`Sigmoid Function <http://en.wikipedia.org/wiki/Sigmoid_function>`_ to return
a likelihood between 0 and 1.

The use of a classifier means that the quality of the alignment is dependent
not only on the input but also on the quality of the trained classifier.
