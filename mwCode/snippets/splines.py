import sys
sys.path.append(r'C:\Users\woottenm\Documents\Code\zebrafish-analysis')

import numpy as np
from scipy.interpolate import interp1d
from snippets.assorted import map_np
from snippets.dbasisP import dbasisP
from numpy.linalg import pinv


def oknotP(n, k, length, etamax):
    """
    Generate a B-spline uniform periodic knot vector.

    Parameters
    ----------
    n : int
        the number of defining polygon vertices
    k : int
        order of the basis function (degree + 1, i.e. cubic => k=4)
    length : int
        length of the organism
    etamax : int
        The maximum of eta, I guess. I have no idea what this does.
    Returns
    -------
    x : ndarray, one dimension
        knot vector

    Author
    ------
    This is a straight Python port of oknotP.m by Ebraheem Fontaine
    Code in repo FlyRanch/flytracker in oknotP.m
    """
    epsilon = 0
    mid = np.linspace(-etamax - length/2, etamax + length/2, n - k + 2)
    delta = length / (n - k + 2 - 1)
    beg = mid[0] + np.arange(-1*(k-1), 0) * delta
    end = mid[-1] + np.arange(1, k) * delta
    return np.concatenate([beg, mid, end])

def fit_b_spline_to_curve(curve, **kwargs):
    """
    Given a curve and some input parameters, returns a smoothened curve after
    fitting with B-Spline periodic functions

    Parameters
    ----------
    curve : UNKNOWN_DTYPE
        The curve to fit
    order : int (optional, default = 4)
        The order of the B-splines
    smoothness : int (optional, default = 6)
        Smoothness. Larger values lead to smoother curves.
    length : int (optional, default = len(curve))
        Desired length of the output curve
    Returns
    -------
    curve_fit : UNKNOWN_DTYPE
        Curve obtained from fitting.
    N : UNKNOWN_DTYPE
        B-spline basis used to fit
    BB : UNKNOWN_DTYPE
        Coefficients, such that curve_fit = N * BB
    D1 : UNKNOWN_DTYPE
        1st derivative of the basis to get tangents
    D2 : UNKNOWN_DTYPE
        2nd derivative of the basis

    Author
    ------

    Avinash Pujala, Koyama lab/JRC, 2018, based on scripts from Fontaine et al
    (see auto_init.m)

    """
    order = kwargs.get('order', 4)
    smoothness = kwargs.get('smoothness', 6)
    length = kwargs.get('length', len(curve))

    mLen = len(curve)
    npts = max([round(mLen / smoothness), order])
    t = np.linspace(0, 1, mLen)
    tt = np.linspace(0, 1, length)
    spline_function = interp1d(t, curve)
    curve_spline = map_np(spline_function, tt)

    knotv = oknotP(npts, order, 1, 0)
    T = np.linspace(-1, 0, length)
    (N,D1,D2) = dbasisP(order,T,npts,knotv)
    BB = pinv(N) @ curve_spline
    curve_fit = N @ BB
    return (curve_fit, N, BB, D1, D2)
