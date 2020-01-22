import numpy as np
from numpy import ndarray as npa

def dbasisP(c: int, T: npa, npts: int, x: npa) -> (npa, npa, npa):
    """
    Generate B-spline basis functions and their derivatives for uniform open knot vectors.

    Parameters
    ----------

    c : int
        order of the B-spline basis function
    T : 1D ndarray of float
        parameter values?
    npts : int
        number of defining polygon vertices
    x : ndarray of float
        knot vector

    Returns
    -------

    N : ndarray of floats with dimensions len(T) × npts
        array containing the basis functions
    D1 : ndarray of floats with dimensions len(T) × npts
        array containing the derivatives of the basis functions
    D2 : ndarray of floats with dimensions len(T) × npts
        array containing the second derivatives of the basis functions
    """
    N = np.empty((len(T), npts), dtype=float)
    D1 = np.empty((len(T), npts), dtype=float)
    D2 = np.empty((len(T), npts), dtype=float)

    for i in range(len(T)):
        t = T[i]

        # allows for number of defining polygon vertices = npts
        temp  = np.zeros(npts + c, dtype=float)
        temp1 = np.zeros(npts + c, dtype=float)
        temp2 = np.zeros(npts + c, dtype=float)

        # calculate the first order basis functions n[j]
        for j in range(npts + c - 1):
            if x[j] <= t < x[j+1]:
                temp[j] = 1
            else:
                temp[j] = 0

        # pick up last point
        if t == x[npts - 1]:
              temp[npts - 1] = 1
              temp[npts] = 0

        # calculate the higher order basis functions
        for k in range(2, c + 1):
            for j in range(npts + c - k):
                b1 = ((t-x[j]) * (temp[j])) / (x[j+k-1] - x[j]) if temp[j] != 0 else 0
                b2 = ((x[j+k]-t) * (temp[j+1])) / (x[j+k] - x[j+1]) if temp[j+1] != 0 else 0
                # calculate first derivative
                f1 = temp[j]/(x[j+k-1] - x[j]) if temp[j] != 0 else 0
                f2 = -temp[j+1]/(x[j+k] - x[j+1]) if temp[j+1] != 0 else 0
                f3 = ((t - x[j]) * (temp1[j])) / (x[j+k-1] - x[j]) if temp1[j] != 0 else 0
                f4 = ((x[j+k] - t) * (temp1[j+1])) / (x[j+k] - x[j+1]) if temp1[j+1] != 0 else 0
                # calculate second derivative
                s1 = (2 * (temp1[j])) / (x[j+k-1] - x[j]) if temp1[j] != 0 else 0
                s2 = (-2 * (temp1[j+1])) / (x[j+k] - x[j+1]) if temp1[j+1] != 0 else 0
                s3 = ((t - x[j]) * (temp2[j])) / (x[j+k-1] - x[j]) if temp2[j] != 0 else 0
                s4 = ((x[j+k] - t) * (temp2[j+1])) / (x[j+k] - x[j+1]) if temp2[j+1] != 0 else 0

                temp[j] = b1 + b2
                temp1[j] = f1 + f2 + f3 + f4
                temp2[j] = s1 + s2 + s3 + s4

                for j in range(npts):
                    N[i, j] = temp[j]
                    D1[i, j] = temp1[j]
                    D2[i, j] = temp2[j]
    return (N, D1, D2)
