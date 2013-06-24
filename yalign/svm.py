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
        # FIXME: Move the sign trick into sentencepairscore
        self.sign = 1
        vector = self.svm.support_vectors_[0]
        class_ = self.svm.predict(vector)
        score = self.svm.decision_function(vector)
        assert score != 0  # Because it's a support vector
        if (class_ is True and score < 0) or \
           (class_ is False and score > 0):
            self.sign = -1

    def classify(self, data):
        vector = self.vectorize(data)
        return self.svm.predict(vector)[0]

    def score(self, data):
        """
        True class is positive, False class is negative.
        """
        vector = self.vectorize(data)
        return self.svm.decision_function(vector) * self.sign

    def vectorize(self, data):
        vector = [attr(data) for attr in self.attributes]
        vector = numpy.array(vector)
        # FIXME: This is OUR convention (values in 0..1) not in general.
        #        Consider moving.
        assert(vector.all() >= 0)
        assert(vector.all() <= 1)
        return vector


if __name__ == "__main__":
    pass
