"""Assorted utilities for generic problems."""
import numpy as np
from scipy.signal import convolve2d

def map_np(function, images) -> np.ndarray:
    """Like `map`, but it converts its result into a numpy array."""
    return np.array(list(map(function, images)))

def zscore(x: np.ndarray) -> np.ndarray:
    """Replace all array values with their z-scores."""
    return (x - x.mean()) / x.std(ddof=1)

def sparse_index(M, rInds, cInds):
    """
    Given a matrix and row and col indices, returns entries from the matrix
    that correspond to one-to-one matched row and col indices.
    """
    assert len(rInds) == len(cInds)
    elems = []
    for (r, c) in zip(rInds, cInds):
        elems.append(M[r, c])
    return np.array(elems)

def calcDist(manyPts, singlePt):
    """
    Calculate the distance between a point and others.

    Parameters
    ----------
    manyPts : ndarray (N × 2)
        The points to compare
    singlePt : ndarray (2)
        The reference point against which the other points are measured

    Returns
    -------
    distances : ndarray (N)
        The distances between the points and the reference

    """
    repeated = np.tile(singlePt, (len(manyPts), 1))
    return np.sqrt(((repeated - manyPts) ** 2).sum(axis=1))

def find_closest_points(set1, set2):
    """
    Given two sets of points (coordinates; currently only in 2D), returns two
    points, one from each set such that the distance between these two points
    is the shortest between any two such points.

    Parameters
    ----------
    set1 : ndarray (dimensions N × 2, where N is the point count)
        2D coordinates of the first set of points
    set2 : ndarray (dimensions M × 2, where M is the point count)
        2D coordinates of the second set of points

    Returns
    -------
    closestPts : (n × 2 ndarray, m × 2 ndarray)
        The closest points. The first part of the tuple holds the points from
        the first set, and the second part holds the points from the second
        set.
    minDist : float
        The distance between the closest points
    D : ndarray (N × M) (TODO: is this right?)
        The matrix of distances between each point pair

    Author
    ------
    Avinash Pujala, Koyama lab/JRC, 2018

    """
    assert set1.shape[1] == 2
    assert set2.shape[1] == 2

    X_one = np.tile(set1[:, 0], (1, len(set2)))
    Y_one = np.tile(set1[:, 1], (1, len(set2)))

    X_two = np.tile(set2[:, 0].T, (len(set1), 1))
    Y_two = np.tile(set2[:, 1].T, (len(set1), 1))

    D = np.sqrt((X_one - X_two) ** 2 + (Y_one - Y_two) ** 2)
    (r, c) = (D == D.min()).nonzero()
    minDist = sparse_index(D, r, c)
    closestPts = (set1[r], set2[c])
    return (closestPts, minDist, D)

def gausswin(L, alpha=2.5):
    """
    An N-point Gaussian window with alpha proportional to the reciprocal of the
    standard deviation.  The width of the window is inversely related to the
    value of alpha.  A larger value of alpha produces a more narrow window.

    Parameters
    ----------
    L : int
    alpha : float
      Defaults to 2.5
    Returns
    -------
    The gaussian window

    Author
    ------
    Taken from openworm/open-worm-analysis-toolbox (GitHub) utils.py
    """

    N = L - 1
    n = np.arange(0, N + 1) - N / 2
    w = np.exp(-(1 / 2) * (alpha * n / (N / 2)) ** 2)

    return w

def gaussblur(stack, size):
    def blur_one(image):
        ker = gausswin(size)
        ker = ker.reshape((len(ker), 1))
        ker = ker @ ker.T
        ker = ker / sum(ker)
        return convolve2d(image, ker, mode='same')
    return map_np(blur_one, stack)
