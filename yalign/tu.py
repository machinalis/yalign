#!/usr/bin/env python
# coding: utf-8

"""
Abstraction of a Translation Unit
"""


class TU(object):
    def __init__(self, src, tgt, distance, aligned=None):
        """
        Creates a Translation Unit with source, target, the
        distance between this two and if it's aligned or not.
        """

        if not isinstance(src, unicode) or not isinstance(tgt, unicode):
            raise ValueError("Source and target must be unicode")
        if not src or not tgt:
            raise ValueError("Source or target empty")
        if not isinstance(distance, float) or not 0.0 <= distance <= 1.0:
            raise ValueError("Invalid distance: {} ({})".format(distance))

        self.src = src
        self.tgt = tgt
        self.distance = distance
        self.aligned = aligned


