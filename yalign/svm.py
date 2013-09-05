# -*- coding: utf-8 -*-

"""
Module for code dealing with the classifier.
"""

import numpy

from sklearn import svm
from simpleai.machine_learning import Classifier


class SVMClassifier(Classifier):
    """
    A Support Vector Machine classifier to classify if a sentence is a
    translation of another sentence.
    """

    def learn(self):
        """
        Train the classifier.
        """
        vectors = []
        answers = []
        for data in self.dataset:
            vector = self._vectorize(data)
            vectors.append(vector)
            answer = self.problem.target(data)
            answers.append(answer)
        if not vectors:
            raise ValueError("Cannot train on empty set")
        self.svm = svm.SVC()
        self._SVC_hack()
        self.svm.fit(vectors, answers)

    def classify(self, sentence_pair):
        """
        Classify if this SentencePair `sentence_pair` has sentences
        that are translations of each other.
        """
        self._SVC_hack()
        vector = self._vectorize(sentence_pair)
        return self.svm.predict(vector)[0], 1

    def score(self, data):
        """
        The score is positive for an alignment.
        """
        self._SVC_hack()
        vector = self._vectorize(data)
        return float(self.svm.decision_function(vector))

    def _vectorize(self, data):
        vector = [attr(data) for attr in self.attributes]
        vector = numpy.array(vector)
        return vector

    def __getstate__(self):
        result = self.__dict__.copy()
        if "dataset" in result:
            del result["dataset"]
        return result

    def _SVC_hack(self):
        """
        This is a dirty hack to deal with SVC's that so that a pickled classifier
        works across scikit-learn versions.
        """
        if not hasattr(self.svm, '_impl'):
            self.svm._impl = 'c_svc'
        if not hasattr(self.svm, 'impl'):
            self.svm.impl = 'c_svc'

