#!/usr/bin/env python
import sys


def impedance(vp, rho):
    """
    Given Vp and rho, compute impedance. Convert units if necessary.
    Test this module with:

        python -m doctest -v impedance.py

    Args:
        vp (float): P-wave velocity.
        rho (float): bulk density.

    Returns:
        float. The impedance.

    Examples:
    >>> impedance(2100, 2350)
    4935000
    >>> impedance(2.1, 2.35)
    4935000.0
    >>> impedance(-2000, 2000)
    Traceback (most recent call last):
        ...
    ValueError: vp and rho must be positive

    """
    if vp < 10:
        vp = vp * 1000
    if rho < 10:
        rho = rho * 1000
    if not vp * rho >= 0:
        raise ValueError("vp and rho must be positive")

    return vp * rho


if __name__ == "__main__":

    vp = float(sys.argv[1])
    rho = float(sys.argv[2])
    print(impedance(vp, rho))
