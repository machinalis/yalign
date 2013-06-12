# -*- coding: utf-8 -*-

from simpleai.search import SearchProblem, astar


class NWSequenceAlignmentSearchProblem(SearchProblem):
    def __init__(self, xs, ys, weight, gap_penalty):
        super(NWSequenceAlignmentSearchProblem, self).__init__((-1, -1))
        self.xs = xs
        self.ys = ys
        self.W = weight
        self.D = gap_penalty
        self.N = len(xs)
        self.M = len(ys)
        self.goal = (self.N - 1, self.M - 1)

    def actions(self, state):
        i, j = state
        i += 1
        j += 1
        if i != self.N and j != self.M:
            a, b = self.xs[i], self.ys[j]
            yield (i, j, self.W(a, b))
        if i != self.N:
            yield (i, None, self.D)
        if j != self.M:
            yield (None, j, self.D)

    def result(self, state, action):
        x, y = state
        i, j, cost = action
        if i is None:
            i = x
        if j is None:
            j = y
        return (i, j)

    def cost(self, state1, action, state2):
        i, j, cost = action
        return cost

    def is_goal(self, state):
        return state == self.goal


def AlignSequences(xs, ys, weight, gap_penalty=-1):
    """
    Returns an alignment of sequences `xs` and `ys` such that it maximizes the
    sum of weights as given by the `weight` function and the `gap_penalty`.
    The aligment format is a list of tuples `(i, j, cost)` such that:
        `i` and `j` are indexes of elements in `xs` and `ys` respectively.
        The alignment weight is sum(cost for i, j, cost in alignment).
        if `i == None` then `j` is not aligned to anything (is a gap).
        if `j == None` then `i` is not aligned to anything (is a gap).
    """
    W = lambda a, b: -weight(a, b)
    problem = NWSequenceAlignmentSearchProblem(xs, ys, W, -gap_penalty)
    node = astar(problem, graph_search=True)
    path = [action for action, node in node.path()[1:]]
    return [(i, j, -cost) for i, j, cost in path]
