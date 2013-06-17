#!/usr/bin/env python
# coding: utf-8

"""
Support Vector Machine Classifier
"""

import numpy
from sklearn import svm
from simpleai.machine_learning import Classifier


class SVMClassifier(Classifier):
    def learn(self):
        vectors = []
        answers = []
        for data in self.dataset:
            vector = self.vectorize(data)
            vectors.append(vector)
            answer = self.problem.target(data)
            answers.append(answer)
        self.svm = svm.SVC()
        self.svm.fit(vectors, answers)

    def classify(self, data):
        vector = self.vectorize(data)
        return self.svm.predict(vector)[0]

    def score(self, data):
        vector = self.vectorize(data)
        return self.svm.decision_function(vector)

    def vectorize(self, data):
        vector = [attr(data) for attr in self.attributes]
        vector = numpy.array(vector)
        assert(vector.all() >= 0)
        assert(vector.all() <= 1)
        return vector


if __name__ == "__main__":
    pass
