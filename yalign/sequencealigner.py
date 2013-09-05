# -*- coding: utf-8 -*-
"""
Module for handling sequence alignment.
"""
from simpleai.search import SearchProblem, astar


class SequenceAligner(object):
    """
    Aligns two sequences.
    """
    def __init__(self, score, gap_penalty):
        self.score = score
        self.penalty = gap_penalty

    def __call__(self, xs, ys, score=None, penalty=None):
        """
        Returns an alignment of sequences `xs` and `ys` such that it maximizes
        the sum of weights as given by the `score` function and the
        `gap_penalty`.
        The aligment format is a list of tuples `(i, j, cost)` such that:
            `i` and `j` are indexes of elements in `xs` and `ys` respectively.
            The alignment weight is sum(cost for i, j, cost in alignment).
            if `i == None` then `j` is not aligned to anything (is a gap).
            if `j == None` then `i` is not aligned to anything (is a gap).
        If `minimize` is `True` this function minimizes the sum of the weights
        instead.
        """
        if score is None:
            score = self.score
        if penalty is None:
            penalty = self.penalty
        problem = SequenceAlignmentSearchProblem(xs, ys, score, penalty)
        node = astar(problem, graph_search=True)
        path = [action for action, node in node.path()[1:]]
        return path


class SequenceAlignmentSearchProblem(SearchProblem):
    """
    Represents and manipulates the search space for a sequence
    alignment problem. Used by simpleai's graph search algorithm.
    """
    def __init__(self, xs, ys, score, gap_penalty):
        super(SequenceAlignmentSearchProblem, self).__init__((-1, -1))
        self.xs = xs
        self.ys = ys
        self.W = score
        if gap_penalty < 0.0:
            raise ValueError("gap penalty cannot be negative")
        self.D = gap_penalty
        self.N = len(xs)
        self.M = len(ys)
        self.goal = (self.N - 1, self.M - 1)

    def actions(self, state):
        """
        Returns the next actions from a given state.
        A state is an alignment (tuple of indexes from either sequence).
        An action is a the next alignment to consider with a score for
        that alignment.
        """
        i, j = state
        i += 1
        j += 1
        if i != self.N and j != self.M:
            a, b = self.xs[i], self.ys[j]
            w = self.W(a, b)
            if w < 0.0:
                raise ValueError("cannot have negative weights")
            yield (i, j, w)
        if i != self.N:
            yield (i, None, self.D)
        if j != self.M:
            yield (None, j, self.D)

    def result(self, state, action):
        """ Returns the next state for this state, action pair."""
        x, y = state
        i, j, cost = action
        if i is None:
            i = x
        if j is None:
            j = y
        return i, j

    def cost(self, state1, action, state2):
        """ Cost of this action. """
        i, j, cost = action
        return cost

    def is_goal(self, state):
        """
        Are we finished aligning? True when our when state is the
        alignment (N, M) where N and M are the lengths of the
        two sequences.
        """
        return state == self.goal

    def heuristic(self, state):
        """
        A heuristic for A* type searches. Currently we return
        The distance of this state from the diagonal in a N*M
        lattice where N and M are the lengths of the two sequences.
        """
        i, j = state
        x, y = self.N - i, self.M - j
        n = max(x, y) - min(x, y)
        # To test that this bound does not overestimates the cost try
        # uncommenting the multiplication and re-running the tests.
        return n * self.D  # * 1.001
