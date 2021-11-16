# -*- coding:utf-8 -*-
"""
Classes for handling geological formations in wells.
"""
from dataclasses import dataclass
import glob
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils

logging.basicConfig(level=logging.WARNING)


@dataclass
class Top:
    """
    One top, or point in our pointset.
    """
    name: float
    top: float
    base: float

    @property
    def thickness(self):
        return self.base - self.top


class Well:
    """
    A well class to hold well data.
    """
    def __init__(self, uwi, xy, list_of_tops=None):
        x, y = xy
        self.uwi = uwi
        self.xy = (float(x), float(y))
        self.list_of_tops = list_of_tops or []

    def __repr__(self):
        return f"Well(uwi={self.uwi}, xy={self.xy}, {len(self.list_of_tops)} tops)"

    @classmethod
    def from_file(cls, fname):
        uwi, x, y, tops = utils.well_reader(fname)
        try:
            True
        except:
            uwi = None
            x, y = None, None
            tops = []

        list_of_tops = []
        for top, base, name in tops:
            this_top = Top(name, top, base)
            list_of_tops.append(this_top)

        return cls(uwi, (x, y), list_of_tops)

    def get_top(self, name):
        try:
            top, = [t for t in self.list_of_tops if t.name == name]
        except ValueError:
            top = None
        return top

    @property
    def x(self):
        if self.xy[0] is None:
            return np.nan
        return self.xy[0]

    @property
    def y(self):
        if self.xy[1] is None:
            return np.nan
        return self.xy[1]