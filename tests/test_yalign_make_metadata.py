import os
import unittest
import tempfile
import subprocess as sub
import json

from helpers import default_sentence_pair_score

class TestMakeMetadata(unittest.TestCase):

    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.parallel_corpus = os.path.join(base_path, "data", "canterville.txt")
        _, self.classifier_filename = default_sentence_pair_score()
        _, self.metadata_filename = tempfile.mkstemp()

    def test_make_metadata(self):
        cmd = 'yalign-make-metadata -o %s %s %s' % \
                (self.metadata_filename, self.parallel_corpus, self.classifier_filename)
        sub.call(cmd, shell=True)
        self.assertTrue(os.path.exists(self.metadata_filename))
        metadata = json.load(open(self.metadata_filename))
        self.assertEquals(metadata['lang_a'], 'en')
        self.assertEquals(metadata['lang_b'], 'es')
        self.assertTrue(metadata['gap_penalty'] > 0)
        self.assertTrue(metadata['threshold'] > 0)

    def tearDown(self):
        os.remove(self.metadata_filename)
        os.remove(self.classifier_filename)

if __name__ == "__main__":
    unittest.main()
