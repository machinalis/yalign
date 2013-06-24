import os
import unittest
import tempfile
import subprocess as sub 
import json

from yalign.tu import TU
from yalign import tuscore
from yalign.weightfunctions import TUScore, WordScore
from helpers import default_tuscore

class TestMakeMetadata(unittest.TestCase):

    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.parallel_corpus = os.path.join(base_path, "data", "canterville.txt")
        self.classifier_filename = default_tuscore()
        _, self.metadata_filename = tempfile.mkstemp()

    def test_make_metadata(self):
        cmd = 'yalign-make-metadata -o %s %s %s' % \
                (self.metadata_filename, self.parallel_corpus, self.classifier_filename)
        sub.call(cmd, shell=True)
        self.assertTrue(os.path.exists(self.metadata_filename))
        metadata = json.load(open(self.metadata_filename))
        self.assertEquals(metadata['src_lang'], 'en')
        self.assertEquals(metadata['tgt_lang'], 'es')
        self.assertTrue(metadata['gap_penalty'] > 0)
        self.assertTrue(metadata['threshold'] > 0)

    def tearDown(self):
        os.remove(self.metadata_filename)
        os.remove(self.classifier_filename)

if __name__ == "__main__":
    unittest.main()
