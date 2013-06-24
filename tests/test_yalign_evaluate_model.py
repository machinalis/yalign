# -*- coding: utf-8 -*-

import unittest
import os
from subprocess import Popen, PIPE
from helpers import default_tuscore

class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.parallel_corpus = os.path.join(base_path, "data", "canterville.txt")
        self.metadata_filename = os.path.join(base_path, "data", "metadata.json")
        self.classifier_filename = default_tuscore()

    def test_output(self):
        cmd = 'yalign-evaluate-model %s %s %s' % \
                (self.parallel_corpus, self.classifier_filename, self.metadata_filename)
        p = Popen(cmd, shell=True,stdout=PIPE)
        output, _ = p.communicate() 
        for x in ('max', 'mean', 'std'):
            self.assertTrue(output.find(x) > -1)

if __name__ == "__main__":
    unittest.main()
