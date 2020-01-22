# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 18:09:33 2018

Computer Vision related scripts


@author: pujalaa
"""

def fitCircle(coords,nIter = 50, tol = 1e-4):
    """
    Given a set of points that roughly describe a circle and some noise,
    returns the coordinates of a fitted circle
    Parameters
    ----------
    coords: array, shape = (2,N)
        Coordinates of a set of points, 1st and 2nd row are x- and y-
        coordinates respectively
    nIter: scalar
        Number of iterations to cycle through in determining circle with best
        fit. If the difference between the previous and the current error is 
        below specified tolerance, then breaks out of the loop
    tol: scalar
        Tolerance, see above
        
    Returns
    -------
    coords_fit: array, same shape as coords
        Coordinates of the fit circle
    E: array
        Array of length equal to the # of iterations looped through. This
        gives the difference in error between previous and current iteration
    """
    import numpy as np
    import apCode.volTools as volt
    
    getErr = lambda a,b: np.sqrt(np.sum((a-b)**2))/len(a)    
    x,y  = coords    
          
    N = len(x)
    #th_fit = np.linspace(0,2*np.pi,N)   
    
    n = 0
    E =[]
    for n in np.arange(nIter):
        th,rho = volt.cart2pol(x,y)
        rho_fit = rho.mean()*np.ones((N,)) 
        x_fit,y_fit = volt.pol2cart(th,rho_fit)
        err_now = getErr(rho,rho_fit)
        if n ==0:
            err_diff = tol + 1
            err_prev = err_now
        else:
            err_diff = err_prev-err_now
            err_prev = err_now
        if err_diff < tol:
#            print('Reached tolerance: epsilon = {0}, iter = {1}'.format(err_diff,n))
            break
        offset = np.mean(x)-np.mean(x_fit), np.mean(y)-np.mean(y_fit)
        x, y = x - offset[0], y - offset[1]
        E.append(err_diff)
    coords_new = np.array([x_fit,y_fit])
    mu1 = np.mean(coords,axis = 1)
    mu2 = np.mean(coords_new,axis = 1)
    shift = np.array([mu2-mu1]).reshape((-1,1))    
    coords_new = coords_new -np.array(shift)     
    return coords_new, np.array(E)
        
        