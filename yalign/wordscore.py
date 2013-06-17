#!/usr/bin/env python
# coding: utf-8

import os
import gzip
import codecs
import tempfile
import subprocess
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle


def _open_phrasetable(filepath):
    """
    Opens a phrasetable file using the
    necesary method, either gzip or plain open.
    """

    if filepath.endswith(".gz"):
        return gzip.open(filepath)
    else:
        return codecs.open(filepath, "r", encoding="utf-8")


def _pre_filter(filepath):
    """
    Uses egrep command to pre-filter things
    """

    _, output_filepath = tempfile.mkstemp()
    _, error_filepath = tempfile.mkstemp()
    outfile = open(output_filepath, "w")
    errfile = open(error_filepath, "w")

    if filepath.endswith(".gz"):
        read_cmdline = "gzip -cd {filepath}"
    else:
        read_cmdline = "cat {filepath}"
    read_cmdline = read_cmdline.format(filepath=os.path.abspath(filepath))
    filter_cmdline = "grep -E '^\S+\s\|\|\|\s\S+\s\|\|\|'"
    cmdline = "{read} | {filter}".format(read=read_cmdline,
                                         filter=filter_cmdline)
    status = subprocess.call(cmdline,
                             stdout=outfile,
                             stderr=errfile,
                             shell=True)

    if status != 0:
        message = "Error precompiling: program returned {}.\n" \
                  "Check error output at: {}"
        raise Exception(message.format(status, error_filepath))
    return output_filepath


def filter_phrasetable(in_filepath):
    """
    Given a phrasetable file returns an iterator over the
    important entries on the table.

    The unnecessary entries are the ones with:
        * more than a 1-gram
        * low word count
    """

    in_filepath = _pre_filter(in_filepath)

    with _open_phrasetable(in_filepath) as filehandler:
        for line in filehandler:
            fields = line.split("|||")
            src = fields[0].strip().lower()
            tgt = fields[1].strip().lower()

            if len(src.split()) > 1 or len(tgt.split()) > 1:
                continue

            counts = fields[4].split()
            tgt_count = int(float(counts[0]))
            src_count = int(float(counts[1]))

            count_limit = 1
            if src_count < count_limit or tgt_count < count_limit:
                continue

            probs = fields[2].split()
            inverse_prob = float(probs[0])
            direct_prob = float(probs[2])
            # The resulting prob its a combination of both direct
            # and inverse probability
            prob = 0.5 * inverse_prob + 0.5 * direct_prob

            yield src, tgt, prob


def save_translation_dictionary(translations, outfile):
    """
    Given a iterator that returns (src, tgt, prob) it creates a dictionary
    and saves it in a pickle format into `outfile`
    """

    result = defaultdict(lambda: defaultdict(float))
    for src, tgt, prob in translations:
        result[src][tgt] = prob

    pickle.dump(dict(result), open(outfile, "w"))
