# -*- coding:utf-8 -*-
"""
Utilities for tops parsing.
"""
import re

import numpy as np


def file_reader(name, fname):
    """
    Parse a file and retieve the data for a single top.
    """

    with open(fname) as f:
        text = f.read()

    string = r'# UWI:([ A-Z0-9]+).+'
    string += r'# Latitude \(NAD83\):([-.0-9]+).+'
    string += r'# Longitude \(NAD83\):([-.0-9]+).+'
    string += fr'MD,(.*),(.*),M,{name}'
    pattern = re.compile(string, flags=re.DOTALL)

    uwi, lat, lon, top, base = pattern.search(text).groups()

    return uwi.strip(), float(lon), float(lat), float(top or 0), float(base)


def well_reader(fname):
    """
    Parse a file and retieve the data for a single top.
    """

    with open(fname) as f:
        text = f.read()

    string = r'# UWI:([ A-Z0-9]+).+'
    string += r'# Latitude \(NAD83\):([-.0-9]+).+'
    string += r'# Longitude \(NAD83\):([-.0-9]+).+'
    pattern = re.compile(string, flags=re.DOTALL)
    uwi, lat, lon = pattern.search(text).groups()

    pattern = re.compile(r'MD,(.*),(.*),M,([ A-Z0-9]+)')
    tops = pattern.findall(text)
    tops = [[float(a or np.nan), float(b or np.nan), c] for a,b,c in tops]

    return uwi.strip(), float(lon), float(lat), tops
