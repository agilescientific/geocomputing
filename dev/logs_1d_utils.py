import math

import numpy as np

from welly import Well


def get_GR(w, dropna=False):
    """Convenience function to return GR as list from a well.

    Parameters
    ----------
        w, (welly.Well): a well object to retrieve a GR from
        dropna, (bool):  whether to drop nan values
    Returns
    -------
        gr, (list): curve values in a `list` object.
    """
    # Read the curve data and save them as a list
    gr = list(w.data['GR'].values)

    # Create empty list
    gr_clean = []
    # Flow control: clean nan values
    if dropna:
        # Loop: only add values that are NOT nan to clean list
        for val in gr:
            if not math.isnan(val):
                gr_clean.append(val)
        return gr_clean
    # Flow control: do not clean nan values
    else:
        return gr


def get_GR_np(w, dropna=False):
    """Convenience function to return GR as list from a well.

    Parameters
    ----------
        w, (welly.Well): a well object to retrieve a GR from
        dropna, (bool):  whether to drop nan values
    Returns
    -------
        gr, (list): curve values in a `list` object.

    Note
    ----
        requires numpy
    """
    gr = w.data['GR'].values
    if dropna:
        gr = gr[~np.isnan(gr)]
    return gr