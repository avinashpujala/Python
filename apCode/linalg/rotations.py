# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 03:31:28 2017

@author: pujalaa
"""

def rotateInto(a,b):
    """
    Returns the rotation matrix for rotating the vector b into a.
    Parameters
    ----------
    a: array, (N,) or column vector (N, 1)
        Reference vector in N dimensions.
    b: array, (N,) or column vector (N, 1)
        Vector to rotate
    Returns
    -------
    R: array, (N,N)
        Rotation matrix, the multiplication with with rotates b into a. 
        Note: Will not work if the vectors are completely facing in the opposite directions
    
    Stolen from here:
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/2672702#2672702
    """
    from scipy.linalg import norm
    import numpy as np
    if np.ndim(a) == 1:
        a = a.reshape(-1,1)
    singleton = False
    if np.ndim(b) == 1:
        singleton = True
        b = b.reshape(-1,1)
    n1, n2 = norm(a), norm(b)
    if (n1 ==0) | (n2 ==0):
        raise IOError('Vector cannot be null. Check input!')
    a, b = a/n1, b/n2
    c = a+b
    R = 2*(c@c.T)/(c.T@ c)-np.eye(len(a))
    b_rot = (R@b)*n2
    if singleton:
        b_rot = np.squeeze(b_rot)
    return R, b_rot

def rotateAboutAxisIn3D(axis, theta):
    """
    Returns the rotation matrix for rotation about an axis by the specified angle
    using the "Euler-Rodrigues" formula.
    Stolen from here:
    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    Parameters
    ----------
    axis: array (N,)
        Axis to rotate about. For example rotate in 3D about the z-axis or within
        the x-y plane, set axis = (0,0,1)
    theta: scalar
        Angle in degrees to rotate by
    Returns
    -------
    R: array,(N,N)
        Rotation matrix, which when an N-dimensional vector is multiplied by returns
        the rotated vector.
    """
    from numpy import cross, eye, deg2rad
    from scipy.linalg import expm, norm
    return expm(cross(eye(len(axis)), axis/norm(axis)*deg2rad(theta)))
