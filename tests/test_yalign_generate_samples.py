# -*- coding: utf-8 -*-

import unittest
import os
import subprocess as sub
import csv
from tempfile import mkstemp

class TestGenerateSamples(unittest.TestCase):

    headings = ["aligned", "pos a", "a", "pos b", "b"]

    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.infile_name = os.path.join(base_path, "data", "canterville.txt")
        _, self.outfile_name = mkstemp()

    def test_output(self):
        cmd = 'yalign-generate-samples %s %s' % (self.infile_name, self.outfile_name)
        sub.call(cmd, shell=True)
        self.assertTrue(os.path.exists(self.outfile_name))
        reader = csv.reader(open(self.outfile_name))
        self.assertEquals(self.headings, reader.next())
        samples = list([sample for sample in reader])
        self.assertEquals(4, len(samples))

    def tearDown(self):
        os.remove(self.outfile_name)

if __name__ == "__main__":
    unittest.main()
