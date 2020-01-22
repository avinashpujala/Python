# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 03:42:15 2018

@author: pujalaa
"""
__codeDir = r'V:\Code\Python\code'
import sys as __sys
__sys.path.append(__codeDir)
#from apCode.volTools import pol2cart, cart2pol
#import apCode.volTools as __volt
#pol2cart = __volt.pol2cart
#cart2pol = __volt.cart2pol
#neighborhood = __volt.morphology.neighborhood
   

def blockAroundPoint(p, r):
    """
    Get coordinates of a block of width and height r around a point p
    Parameters
    ----------
    p: array or tuple, shape = (2,)
        Coords of a point in 2D
    r: scalar or array-like, (2,)
        radius of block or rather width and height (w,h)
    
    """
    import numpy as np
    if np.size(r)==1:
        r = np.tile(r,(len(p),))
    coords = [np.arange(p_-r_, p_+r_) for p_, r_ in zip(p,r)]
    [x,y] = np.meshgrid(coords[0],coords[1])    
    return np.array([x.flatten(),y.flatten()])


def baryCenterAlongCircleInImg(img, ctr, r, ctr_prev = None, avoid_angle = 100, plot_bool = False):
    """
    Returns the barycenter along a circle with a specified center in an image
    Parameters:
    ----------
    img: array, (M,N)
        Image.
    ctr: (2,)
        (x,y) coordinates of circle center
    r: scalar
        Radius of circle
    ctr_prev: (2,)
        Center of the circle used to determined the previous barycenter. Useful for 
        iterative estimation of the skeleton of a blob in an image. In my case, the
        midline of the fish.
    avoid_angle: scalar
        To avoid finding the center of a previously used circle as the current barycenter
        the algorithm avoids points in the circle that have angular distance smaller than
        this angle w.r.t the current barycenter
    plot_bool: boolean
        If True, then plots the circles.
    Returns
    -------
    bc: tuple, (2,)
        Barycenter estimated along the circle using the square of the pixel values along the 
        circle as weights.
    C: array, (k,)
        A set of complex numbers, wherein the angle of each number represents the vector
        drawn from the center of the circle used to a point along the circle. The magnitude
        of the number is the relative weight of this vector (proportional to square of the pixel
        intensity).
    """
    import numpy as np
    import apCode.geom as geom
    
    c = geom.circlesAroundPoint(ctr,r)
    if plot_bool:
        import matplotlib.pyplot as plt
        plt.plot(c[0], c[1])
    v = c-ctr.reshape(-1,1)
    v = v[0] + v[1]*1j    
    if np.all(ctr_prev != None):        
        v_ref = ctr_prev - ctr
        v_ref = v_ref[0] + v_ref[1]*1j        
        angles_with_ref = np.abs(np.angle(np.conjugate(v)*v_ref, deg = True))
        inds_ignore = np.where(angles_with_ref < avoid_angle)[0] 
    else:
        inds_ignore = []
    wts = img[c[1].astype(int),c[0].astype(int)]**2
    wts[inds_ignore] = 0
    if np.sum(wts)==0:
        return np.ones((2,))*np.nan
    else:
        wts = wts/wts.sum()
        v_mean  = np.mean(wts*v)
        ind_bc = np.argmin(np.abs(np.angle(v_mean*np.conj(v))))
        return c[:,ind_bc], wts*v
    
def circularAreaAroundPoint(p,r):
    """
    Returns the coordinates of a circular region around a point
    Parameters
    ----------
    p: array or tuple, (2,)
        Point around which to get a circular region. Works in 2D,
        but not in D > 2
    r: scalar, array, or tuple, (2,)
        Radius of circular region
    """
    import numpy as np
    vol = [[] for p_ in p]
    for r_ in range(r):
        t = np.linspace(0, 2*np.pi,8*r_)
        for d in range(len(p)):
            vol[d].extend(r_*np.cos(t + 0.5*np.pi*d))
            
    return np.array(vol)+np.array(p).reshape(-1,1)        

def circlesAroundPoint(pt, radii, density = 1, constant_density = False):
    """
    Returns the coordinates of a series of concentric circles of varying radii.
    Parameters:
    pt: array or tuple, (2,)
        x,y coordinates of the center of the circles.
    radii: array, list or tuple, (n,)
        Radii of each of the circles.
    density: integer
        Determines the number of points on the circle. The number of points on
        a circle are computed as follows np.arange(0,2*npi+dTh,dTh), 
        where dTh = np.pi/(8*density).
    Returns
    -------
    c: list
        A list of (x,y) coordinates for concentric circles of varying radii.    
    """
    import numpy as np
    radii = np.array(radii)
    if np.ndim(radii)==0:
        radii = np.array([radii])
    radii = np.delete(radii, np.where(radii==0))
    if len(radii)==0:
        print('No valid radii')
    c = []    
    for r in radii:
#        t = np.linspace(0,2*np.pi,int(8*r*density))
        if (constant_density) & (r*density!=0):
            dTh = np.pi/(8*r*density)
        else:
            dTh = np.pi/(8*density)
        t = np.arange(0,2*np.pi+dTh,dTh)
        c_now = np.array([r*np.cos(t) + pt[0], r*np.sin(t) + pt[1]])
        c.append(c_now)
    if len(c)==1:
        c = np.squeeze(np.array(c))
    return c

def clipCurvesWithBSpline(curves):
    from dask import delayed, compute
    import numpy as np
#    from apCode.geom import fitBSpline
    from scipy.interpolate import make_interp_spline
    n = curves.shape[1]
    if np.ndim(curves) == 2:
        curves = curves[np.newaxis,...]
    cumLens, cLens = curveLens(curves)
    cLens_norm = cLens/np.min(cLens)
    tt = np.linspace(0,1,n)
    curves_clip = [delayed(make_interp_spline)(np.linspace(0,cln,n), c, k = 3)(tt) for c, cln in zip(curves,cLens_norm)]
    curves_clip = np.array(compute(*curves_clip))
    return np.squeeze(curves_clip)

def clipInterpCurves(curves, q = 60, interp_kind:str = 'cubic'):
    """
    Clip a collection of curves of varying lengths to the specified
    percentile of the lengths and then interpolate to have same number
    of points in each curve
    Parameters
    ----------
    curves: array, ([T,], K, D), where T = number of curves, K = number of points
        making up each curve, and D is the dimensionality of the curve.
        Curves to clip and interpolate.
    q: int
        Percentile of the lengths of the curves to clip all curves to at first.
        Curves less than this value are obviously not clipped.
    interp_kind: str
        Kind of interpolation to use
    """
    from dask import delayed, compute
    from apCode.geom import curveLens, interpolate_curve
    import numpy as np
    nPts = curves.shape[1]
    cumLens, cLens  = curveLens(curves)
    minLen = np.percentile(cLens,q)
    clipInds = np.array([np.argmin(np.abs(c-minLen)) for c in cumLens])
    curves_clip = np.array([c[:ind,:] for c, ind in zip(curves, clipInds)])
    curves_clip = [delayed(interpolate_curve)(c,kind = interp_kind, n = nPts) for c in curves_clip]
    return np.array(compute(*curves_clip))

def closestPointsBetweenIslands(points1, points2):
    """
    When given two sets of points (their coordinates), representing
    say islands in some space, then returns the indices of the closest
    points, one point from each set of points. Uses KDTree algorithm
    from sklearn
    
    Parameters
    ----------
    points1: array, (M, D)
        First set of M points in D dimensional space
    points2: array, (N, D)
        2nd set of N points in D dimensional space
    Returns
    -------
    inds: tuple, (2,)
        Indices, where inds[0], and inds[1] are the indices in the 1st and 2nd
        set of points such that these points are the closest pair of points from
        the 2 sets of points
    minDist: scalar
        The distance betweeen the two closest points
    """
    import numpy as np
    from sklearn.neighbors import KDTree # Using KDtree to compute all combinations of distance intead of brute force approach
    d,inds = KDTree(points1).query(points2)    
    d, inds = d.ravel(),inds.ravel()
    minIdx = np.argmin(d)    
    ind1 = inds[minIdx]
    ind2 = minIdx
    minDist = d[minIdx]
    return (ind1,ind2), minDist

def connectIslands(img, mult = 1):
    """
    Given an image with islands(non-zero pixels surrounded on all sides by zero pixels), 
    returns image where the islands are connected by lines extending between the nearest pair
    of points for a pair of islands.
    
    Parameters
    ----------
    img: array, (M,N)
        Image in which to connect islands.
    mult: scalar
        Value by which to multiply pixel values of bridges connecting islands.
    Returns
    -------
    img_new: array, (M,N)
        Image with lines connecting islands
    coords_line: list, (K,)
        Each element in the list has shape (k,2) and are the row and column coordinates of the bridges
        connecting a pair of islands. K is the number of islands, and k is the number of coordinates
        for a given bridge.    
    """
    import numpy as np
    from skimage.measure import label    
    from itertools import combinations
    img_new = img.copy()
    img_bool = (img_new >0).astype(int) # Binarize image
    img_lbl = label(img_bool) # Get labeled image
    lbls = np.unique(img_lbl)[1:] # Get labels, but ignore background
    if len(lbls) <2:
        print('No islands')
        return img_new, np.fliplr(np.array(np.nonzero(img_new)).T)
    combs = list(combinations(lbls,2))
    coords_line = []
    for comb in combs:        
        p1 = np.fliplr(np.array(np.where(img_lbl == comb[0])).T)
        p2 = np.fliplr(np.array(np.where(img_lbl == comb[1])).T)
        inds,d = closestPointsBetweenIslands(p1,p2)        
        xy = np.array((p1[inds[0],:], p2[inds[1],:]))
        if xy[0,0] > xy[1,0]:
            c = np.arange(xy[0,0]+1,xy[1,0],-1)
        else:
            c = np.arange(xy[0,0],xy[1,0]+1)
        if xy[0,1] > xy[1,1]:
            r = np.arange(xy[0,1]+1,xy[1,1],-1)
        else:
            r = np.arange(xy[0,1],xy[1,1]+1)
        if len(r)>len(c):
            c = np.linspace(c[0],c[-1],len(r)).astype(int)
#             r = np.linspace(r[0], r[-1], len(r)).astype(int)
        elif len(c)>len(r):
            r = np.linspace(r[0],r[-1], len(c)).astype(int)
#             c = np.linspace(c[0], c[-1], len(c)).astype(int)
        keep_row = np.where((r>=0) & (r < img.shape[0]))[0]
        keep_col = np.where((c>=0) & (c < img.shape[1]))[0]
        keepInds = np.intersect1d(keep_row, keep_col)
        r,c  = r[keepInds], c[keepInds]
        img_new[r,c] = 1*mult
        coords_line.append(np.array([r,c]).T)
    return img_new, coords_line

def dCurve_old(curve, kind = 'cubic', resolution  = 1):
    """
    Returns the curvatures (kappa) along a curve
    Parameters
    ----------
    curve: 2D array, (N,2)
        Curve in 2D
    kind: string
        Specifies the type of interpolation to use so as to return the same number of curvatures as the
        length of the curve
    resolution - Scalar, 0 to 1.
        Determines how smooth the curvatures along the curve will be. Lower 
        values lead to smoother curves
    Returns
    -------
    kappa: array, (N,)
        Curvatures along the curve in degrees
    """
    import numpy as np
    from scipy.interpolate import interp1d
    N = curve.shape[0]
    n = np.max((N*resolution, 3)).astype(int)
    if n < N:
        t = np.arange(N)
        tt = np.linspace(t[0], t[-1], n)
        xx = interp1d(t,curve[:,0],kind = kind)(tt)
        yy = interp1d(t,curve[:,1],kind = kind)(tt)
        curve = np.array([xx,yy]).T    
    T = np.gradient(curve)[0]
    Tj = T[:,0] + T[:,1]*1j
    kappa = np.angle(Tj[1:] * np.conj(Tj[:-1]), deg = True)
    t = np.linspace(0,1, len(kappa))
    tt = np.linspace(t[0],t[-1],N)
    kappa = interp1d(t,kappa,kind = kind)(tt)
    return kappa

    """
    Returns the curvatures (kappa) along a curve
    Parameters
    ----------
    curve: 2D array, (N,2)
        Curve in 2D
    kind: string
        Specifies the type of interpolation to use so as to return the same number of curvatures as the
        length of the curve
    resolution - Scalar, 0 to 1.
        Determines how smooth the curvatures along the curve will be. Lower 
        values lead to smoother curves
    Returns
    -------
    kappa: array, (N,)
        Curvatures along the curve in degrees
    """
    import numpy as np
    from scipy.interpolate import interp1d
    N = curve.shape[0]
    n = np.max((N*resolution, 3)).astype(int)
    if n < N:
        t = np.arange(N)
        tt = np.linspace(t[0], t[-1], n)
        xx = interp1d(t,curve[:,0],kind = kind)(tt)
        yy = interp1d(t,curve[:,1],kind = kind)(tt)
        curve = np.array([xx,yy]).T    
    T = np.gradient(curve)[0]
    Tj = T[:,0] + T[:,1]*1j
    kappa = np.angle(Tj[1:] * np.conj(Tj[:-1]), deg = True)
    t = np.linspace(0,1, len(kappa))
    tt = np.linspace(t[0],t[-1],N)
    kappa = interp1d(t,kappa,kind = kind)(tt)
    return kappa

def dCurve(curve, kind = 'cubic'):
    """
    Returns the curvatures (kappa) along a curve
    Parameters
    ----------
    curve: 2D array, (N,2)
        Curve in 2D
    kind: string
        Specifies the type of interpolation to use so as to return the same number of curvatures as the
        length of the curve
    resolution - Scalar, 0 to 1.
        Determines how smooth the curvatures along the curve will be. Lower 
        values lead to smoother curves
    Returns
    -------
    kappa: array, (N,)
        Curvatures along the curve in degrees
    """
    import numpy as np
    from scipy.interpolate import interp1d    
    t = np.linspace(0,1,len(curve))
    tt = np.linspace(0,1,len(curve)+2)
    x_new = interp1d(t,curve[:,0], kind = 'cubic')(tt)
    y_new = interp1d(t,curve[:,1], kind = 'cubic')(tt)
    curve = np.array([x_new, y_new]).T
    T = np.diff(curve,axis = 0)
    Tj = T[:,0] + T[:,1]*1j
    kappa = np.angle(np.conjugate(Tj[:-1])*Tj[1:], deg = True)    
    return kappa

def equalizeCurveLens(curves, kind:str = 'linear'):
    """
    Given a set of curves in ND, returns them after
    interpolating to make them all the same length
    Parameters
    ----------
    curves: array, (nCurves,nPtsInCurve,nDims)
        Curves to uniformize.
    kind: str
        Kind of interpolation. See scipy.interpolate.interp1d
    Returns
    -------
    curves_unif: array, same shape as curves
        Curves after length uniformization.
    """
    from dask import delayed, compute
    import numpy as np
    import os
    from scipy.interpolate import interp2d
    n_workers = np.min((os.cpu_count(),32))
    getLenVec = lambda c: np.cumsum((np.gradient(c,axis = 0)**2).sum(axis = 1)**0.5)
    lenVecs = compute(*[delayed(getLenVec)(_) for _ in curves])
    lenVecs = np.array([np.linspace(0, lv[-1], len(lv)) for lv in lenVecs])    
    cLens = np.array([lv[-1] for lv in lenVecs])
    lenVecs_norm = lenVecs/np.min(cLens)
    yy = np.linspace(0,1, curves.shape[1])    
#    curves_unif = compute(*[delayed(interp1d)(lv,c, kind = kind, axis =0)(yy) for\
#                            lv, c in zip(lenVecs_norm,curves)], scheduler = 'processes', num_workers = n_workers)
    x = np.arange(curves.shape[2])    
    curves_unif = compute(*[delayed(interp2d)(x,y,c, kind = kind)(x, yy) for\
                            y, c in zip(lenVecs_norm, curves)], scheduler = 'processes', num_workers = n_workers)
    return np.asarray(curves_unif)

def endpoints_curve_2d(curve, nhood = 1):
    """
    Returns the indices of the endpoints of a curve in the space S
    Parameters
    ---------
    curve: array, (K,2)
        Row-column coordinates of the curve, such that curve[0,0], and 
        curve[0,1] are the row and column coordinates of the first point of the 
        curve.
    space: array, (M,N)
        The 2D space in which the curve exists, such as an image. To correctly 
        determine the endpoints, the pixel values of the image must be zero everywhere 
        except on the curve, where the pixel values must be 1.
    """
    import numpy as np
    import apCode.volTools as volt
    curve = curve-np.min(curve,axis = 0).astype(int) + 10  # Adding 10 simply to make sure the curve is
                    # not at the edge of the space in which it is embedded
    space = np.zeros(np.array(np.max(curve,axis = 0)).astype(int) + 11)
    space[curve[:,0].astype(int),curve[:,1].astype(int)] =1
    P = volt.morphology.neighborhood(space, curve, n = nhood)[0]
    n = np.array([np.sum(p) for p in P])
    inds = np.where(n==2)[0]
    return inds

def fitLine(coords):
    """
    Given a set of x-y coords, returns the coordinates of a line fit
    as well as the slope and y-intercept to that line
    Parameters
    ----------
    coords: array, (N,2)
        Coordinates (2D) of points to fit line to.
    Returns
    -------
    y_pred: array, (N,)
        y coordinates of line fit to points. x-coordinates are the same
    params: 2-tuple
        params = (m,b), where m is slope and b y-intercept
    pcov: array, (2,2)
        Covariance matrix
    """
    from scipy.optimize import curve_fit
    import numpy as np
    def f(x,m,b):
        return m*x+b
    
    try:
        popts,pcov = curve_fit(f,coords[:,0],coords[:,1])
        m,b = popts
    except:
        popts,pcov = curve_fit(f,coords[:,0],10*coords[:,1])
        m,b = popts
        m, b = m/10, b/10
    if np.sum(pcov <1e3):        
        y_pred = m*coords[:,0] + b
        coords_line = np.c_[coords[:,0],y_pred]
    else:
#        print('Flipping axes')
        T_rot_90 = np.array([[0,-1],[1,0]]) # Counter clockise rotation by 90 deg
        T_ref_y = np.array([[-1,0],[0,1]]) # Reflection in y-axis
        T = np.dot(T_ref_y, T_rot_90) # Switching x-y axes
        coords_new = np.dot(T,coords.T).T
        popts, pcov = curve_fit(f,coords_new[:,0], coords_new[:,1])
        m,b = popts
        y_pred = m*coords_new[:,0] + b
        coords_line = np.c_[coords_new[:,1], coords_new[:,0]]
#        coords_line = np.dot(T,coords_new.T).T
#        m = 1/m
#        b = b/m    
    
    return coords_line, (m,b), pcov

def fitLines(coords, n_angles = 8):
    """
    Given the coordinates of a set of point such as defining a curve in 2D, returns
    the coordinates of a series of fit lines, the angles between successive lines,
    as well the complex notations of vectors representing the lines
    Parameters
    ---------
    coords: array, (N,2)
        Coordinates of points
    n_angles: integer
        Number of angles to return. Number of lines will exceed this number by 1.
    Returns
    -------
    thetas: array, (n_angles,)
        Angles between successive lines fit to the points.
    lines: list, (n_angles+1,)
        List of lines fit to the points. Each element in the list is the x-y
        coordinates of the cooresponding line
    z: array, (n_angles+1,)
        Complex numbers vectorially representing the lines
    """
    import numpy as np 
    inds = np.linspace(0, len(coords),n_angles+1).astype(int)
    ind_list = [np.arange(inds[i-1],inds[i]) for i in range(1,len(inds))]
    lines, z = [],[]
    for i in ind_list:
        c = coords[i,:]
        try:
            c_fit = fitLine(c)[0]
        except:
            c_fit = fitLine(c)[0]
        lines.append(c_fit)
        z_ = c_fit[-1]-c_fit[0]
        z.append(z_[0] + z_[1]*1j)
    thetas = []
    for i in np.arange(1,len(z)):
        thetas.append(np.angle(z[i]*np.conj(z[i-1]),deg = True))    
    return np.array(thetas), lines, np.array(z)

def curveLens(curves):
    from dask import delayed, compute
    import numpy as np
    getCumLen = lambda c: np.cumsum((np.diff(c,axis = 0)**2).sum(axis = 1)**0.5)
    if np.ndim(curves)==2:
        curves = [curves]
    cumLens = np.array(compute(*[delayed(getCumLen)(c) for c in curves]))
    lens = np.array([c[-1] for c in cumLens])
    if isinstance(cumLens, np.ndarray):
        cumLens= np.squeeze(cumLens)
    return cumLens, lens

def fitBSpline(curve, k:int = 3, smoothness = 0.5, n =None, tRange = (0,1), bc_type:str = 'clamped'):
    """
    Smooth, subsample or truncate a curve using B-splines
    Parameters
    ----------
    Curve: array, (N, 2)
        Curve in 2D (may work for ND, but did not test)
    k: int
        B-spline polynomial degree
    smoothness: scalar, interval (0,1)
        Smoothing factor. Larger values lead to more smoothing. At 0, the
        curve passes through every point.
    n: int or None
        Number of points in the final curve.
    tRange: 2-tuple
        Specifies truncation of the curve. tRange = (0,1) results in the full
        curve. Smaller values lead o truncation at one or both ends. For e.g.,
        tRange = (0.1,0.9) leads to 10% truncation at both ends.
    bcType: str
        Type of boundary conditions. See, scipy.inteerpolate.make_interp_spline.
        The default value of 'clamped' results in the first derivatives at the curve
        ends to be zero.
    Returns
    -------
    curve_fit: array, (n,2)
        The fitted curve.
    """
    import numpy as np
    from scipy.interpolate import make_interp_spline
    from apCode.geom import smoothen_curve
    cLen = curve.shape[0]
    if np.any(n == None):
        n = cLen
    t = np.linspace(0,1,cLen)
    tt = np.linspace(*tRange,n)
    curve_s = smoothen_curve(curve, order = k, smooth = smoothness)
    curve_b = make_interp_spline(t,curve_s,k=3,bc_type = bc_type)(tt)
    return curve_b

def interpolateCurvesND(curves, q = 60, kind:str = 'nearest', N:int = 50, mode:str = '2D'):
    """
    When given midlines returned by FreeSwimBehavior.track.midlinesFromImages, returns an array of 
    midlines adjusted for the most common length and interpolated to fill NaNs
    Parameters
    ----------
    midlines: list (T,)
        T midlines of varying length
    kind: string
        Kind of interpolation to use; see scipy.interpolate.interp1d
        fill_value: float, None (default), or "extrapolate".  
    Returns
    -------
    midlines_interp,  (T, N, 2)
    Interpolated array of midlines (midlines_interp) and length-adjusted (by extrapolation)
            array of midlines respectively
            
    """
    import numpy as np
    from scipy.interpolate import griddata
    from apCode.geom import interpExtrapCurves
    def interp2D(C,kind):
        coords = np.where(np.isnan(C)==False)
        gx, gy = np.meshgrid(np.arange(C.shape[1]), np.arange(C.shape[0]))
        C_interp = griddata(coords, C[coords],(gy, gx), method = kind)
        return C_interp
    def interp_1D(C,kind):
        from scipy.interpolate import interp1d
        from dask import delayed, compute                
        def interp_fit(x,y,xx,kind):
            return interp1d(x,y,kind = kind)(xx)
        C_interp = []
        x = np.arange(C.shape[1])
        for c_ in C:
            nnInds = np.where(np.isnan(c_)==False)[0]            
            C_interp.append(delayed(interp_fit)(nnInds,c_[nnInds],x,kind))
        return np.array(compute(*C_interp, scheduler = 'processes'))        
    C = interpExtrapCurves(curves,q = q, kind = kind, N = N)[0]
    C_interp = []
    for i in range(C.shape[-1]):
        if mode == '2D':
            C_interp.append(interp2D(C[:,:,i].T,kind).T)
        else:                    
            C_interp.append(interp_1D(C[:,:,i].T,kind).T)    
    return np.transpose(np.array(C_interp),(1,2,0))


def interpExtrapCurves(curves, q = 60, kind:str = 'cubic', N:int = 50):
    """
    Interpolates a set of curves to equate the spacing between successive points within each curve and
    across curves. Then, extrapolate curves shorter than the specified (q) percentile by padding with NaNs.
    Parameters
    ----------
    curves: array, (T,)
        Collection of T curves, where each curve, say curve[n] has dimensions (M,k) where M is the variable number
        of points constituting the curves and k is the dimensionality of the curves. For instance, an 
        M-point curve in 2D would have shape (M, 2)
    q: scalar
        The percentile of the lengths of the curves to use for eliminating interpolating longer curves and 
        extrapolating shorter curves using NaNs
    kind: string
        Kind of interpolation. See scipy.interp1d
    N: int
        The number of points making up the final curves.
    Returns
    -------
    curves_interp: array, (T, N, k)
        Interpolated-extrapolated curves
    lenVec: array, (N,)
        The length vector giving the cumulative sum of distance between successive points in the new curves.
    """
    from scipy.interpolate import interp1d
    from dask import delayed, compute
    import numpy as np
    getLenVec = lambda c: np.insert(np.cumsum((np.sum(np.diff(c,axis = 0)**2,axis = 1)**0.5),axis = 0),0,0)
    lenVecs = compute(*[delayed(getLenVec)(c) for c in curves])
    cLens = np.array([lv[-1] for lv in lenVecs])
    lenVec_unif = np.linspace(0, np.percentile(cLens,q),N)
    curves_interp = np.array(compute(*[delayed(interp1d)(lv, c, axis = 0, kind= kind, bounds_error = False)(lenVec_unif) for lv, c in zip(lenVecs, curves)],\
                                       scheduler = 'processes'))
    return curves_interp, lenVec_unif

def interpolate_curve(points, kind:str = 'cubic', n = None, axis:int = 0):
    """
    Interpolates a curve
    ** From the good sam xdze2 at stackoverflow.com
    Parameters
    ----------
    points: array, (N,k)
        A k-dimensional curve made up of N points
    kind: string
        Kind of interpolation to use (see scipy.interpolate.interp1d). 
        Options are ('linear','nearest','zero','slinear','quadratic','cubic')
    n: scalar
        Number of points in the interpolated curve, If None, then same number of 
        points as the original curve
    Returns
    -------
    points_interp: array, (n,k)
        Smoother curve with n points.
    """
    from scipy.interpolate import interp1d
    import numpy as np
    
    dist = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)/dist[-1]
    _, inds = np.unique(dist,return_index = True)
    inds = np.sort(inds)
    dist, points = dist[inds], points[inds]    
    if n == None:
        n = len(dist)
    alpha = np.linspace(0,1,n)
    points_fit = interp1d(dist,points,kind = kind,axis = axis)(alpha)
    return np.array(points_fit)

def smoothen_curve(points, order = 3, smooth = 0.1, wts = None):
    """
    Smoothen's a curve using interpolation.
    ** From the good sam xdze2 at stackoverflow.com
    Parameters
    ----------
    points: array, (nPoints, nDimensions)
        The curve to smooth
    order: scalar
        Order of the spline function (UnivariateSpline from scipy.interpolate).
        k = 3, results in cubic spline
    smooth: scalar
        Smoothing factor with values in [0,1]. Smaller values lead to less
        smoothing.
    Returns
    -------
    points_smooth: array, (N,k)
        Smoothed curve.
    """
    from scipy.interpolate import UnivariateSpline
    import numpy as np
    nPts = len(points)
    smooth = int(smooth*nPts)  
    dist = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1 )))
    dist = np.insert(dist, 0, 0)/dist[-1]
    _, inds = np.unique(dist, return_index = True)
    dist, points = dist[inds], points[inds]
    if not np.any(wts==None):
        print('Using weights...')
        wts = wts[inds]
        splines = [UnivariateSpline(dist,coords, k = order, w = wts) for coords in points.T]
    else:
        splines = [UnivariateSpline(dist,coords, k = order, s = smooth) for coords in points.T]
    alpha = np.linspace(0,1,nPts) 
    points_fit = np.vstack([spl(alpha) for spl in splines]).T
    return points_fit

def smoothen_curves(curves, order:int = 3, smooth = 0.5, wts = None):
    """
    Smoothen's a curve using interpolation.
    ** From the good sam xdze2 at stackoverflow.com
    Parameters
    ----------
    curves: array, (nCurves, nPointsInCurve, nDimensions)
        A curve with N points in k dimensions.
    order: scalar
        Order of the spline function (UnivariateSpline from scipy.interpolate).
        k = 3, results in cubic spline
    smooth: scalar
        Smoothing factor with values in [0,1]. Smaller values lead to less
        smoothing.
    Returns
    -------
    curves_smooth: array, (nCurves, nPointsInCurve, nDimensions)
        Smoothed curve.
    """
    import numpy as np
    import dask
    curves_smooth = np.asarray(dask.compute(*[dask.delayed(smoothen_curve)(_, smooth = 20) for _ in curves],\
                                              scheduler = 'processes'))
    return curves_smooth

def smoothenCurve(curve, nInt = 7, kind = 'cubic', N = None):
    """
    Given a curve, smoothens and returns it.
    Parameters
    ---------
    curve: array, (M,2)
        Curve in 2D.
    nInt: integer scalar
        Number of intervals to break the curve into while smoothing. 
        Larger values result in more smoothing.
    kind: string
        Kind of interpolation to use for smoothing. See scipy.interpolate.interp1d
    N: integer scalar or None
        Length of final smoothened curve. If None, then has same length as input curve.
    Returns
    -------
    curve_smooth: array, (N,2)
        Smoothened curve
    """
    import numpy as np
    from scipy.interpolate import interp1d, UnivariateSpline
    if N is None:
        N = len(curve)
    curveLen = len(curve)
    t = np.linspace(0,1,curveLen)
    tt = np.linspace(0,1,nInt+1)
    if (len(tt)<=3) & (curveLen > 3):
        tt = np.linspace(0,1,curveLen)
    if np.any(np.isnan(curve)):
        kinds = np.array(['linear','nearest','zero','slinear','quadratic','cubic'])
        k = np.where(kinds == kind)[0][0]
        w = np.ones((len(t),))
        nanInds = np.where(np.isnan(np.sum(curve,axis = 1)))[0]
        curve_new = curve.copy()
        curve_new[nanInds,:] = 0
        w[nanInds] = 0
        curve_interp = np.array([UnivariateSpline(t,c, k = k, w = w)(tt) for c in curve_new.T]).T
    else:
        curve_interp = np.array([interp1d(t,c, kind = kind)(tt) for c in curve.T]).T
    t = np.linspace(0,1,np.shape(curve_interp)[0])
    tt = np.linspace(0,1,N)
    curve_interp = np.array([interp1d(t,c, kind = kind)(tt) for c in curve_interp.T]).T        
    return curve_interp

def sortCurvePts(curve):
    """
    Given the coordinates of a curve in 2D, returns them after sorting
    such that each point along the curve is followed by the next closest one
    Parameters
    ----------
    curve: array, (N,2)
    
    Returns
    -------
    curve_sorted: (N,2)
        Sorted curve
    inds_sort: array, (N,)
        Indices in sorted order
    D: array, (N,)
        Distance between each point and the next one in sorted order
    
    """
    import numpy as np
    ec2d = endpoints_curve_2d
    dist = lambda p1, p2: np.sqrt(np.sum((np.array(p2)-np.array(p1))**2))
    inds_new = [ec2d(curve)[0]]
    inds_old = np.delete(np.arange(len(curve)),inds_new)
    D = [0] 
    while len(inds_old)>1:
        d = np.array([dist(curve[inds_new[-1],:],curve[ind,:]) for ind in inds_old])
        ind_min = np.argmin(d)
        D.append(np.min(d))
        inds_new.append(inds_old[ind_min])
        inds_old = np.delete(inds_old,ind_min)
    inds_new.append(inds_old[0])
    return curve[inds_new,:], inds_new, D

def sortPointsByWalking(pts, ref = None):
    """
    Returns the indices for sorting a set of points on a curve. Sorts by walking
    from one point to the next nearest and then correcting for the start of the
    journey from a non-endpoint
    Parameters
    ----------
    pts: array, (N,2)
        Points to sort
    ref: array, (2 [,1])
        Reference point from where to start walking. If None, then starts walking
        from the current first point
    """
    import numpy  as np
    if ref is None:
        ref = pts[0,:]+1
    inds_ord = []
    S = lambda pt, pts: np.sum((pt-pts)**2, axis = 1)**0.5
    dS = lambda pts : np.sum(np.diff(pts,axis = 0)**2,axis = 1)**0.5
    pts_new = pts.copy()
    inds_ord, dists_ord = [],[]
    for ind in range(len(pts)):
        dists = S(ref,pts_new)    
        ind_nearest = np.argsort(dists)[0]
        dists_ord.append(dists[ind_nearest])
        inds_ord.append(ind_nearest)
        ref = pts[ind_nearest]
        pts_new[ind_nearest] = np.inf
    inds_ord, dists_ord = np.array(inds_ord), np.array(dists_ord)    
    pts_ord = pts[inds_ord,:]
    dists = dS(pts_ord)
    jumpInd = np.argmax(dists)
    inds_ord_flip = inds_ord.copy()
    inds_ord_flip[:jumpInd+1] = np.flipud(inds_ord_flip[:jumpInd+1])
    sum_dists = np.sum(dists)
    dists_flip = dS(pts[inds_ord_flip,:])
    if np.sum(dists_flip) < np.sum(sum_dists):
        inds_ord = inds_ord_flip
    else:
        dists_ord = dists_flip       
    return inds_ord

def sortPointsByKDTree(points, n_neighbors = 4,src = 0):
    """
    Given a set of points (only tested for points in 2D, but 
    could work for higher dimensions), returns indices of
    points such that total distance traveled is minimal 
    (traveling salesman problem).
    This code is possiible thanks to a good Samaritan on 
    stackoverflow.com
    https://stackoverflow.com/users/764322/imanol-luengo
    
    Parameters
    ----------
    points: array, (N,D)
        A set of N points in D dimensions (typically D = 2)
    n_neighbors: int
        Number of neighbors to consider at a time (see NearestNeighbors in
        sklearn.neighbors)
    src: int
        The index of the input points to use as the first node
        
    Returns
    -------
    opt_order: array, (M,)
        Optimal order for the points. Typically, M = N, but if the spacing
        between two closest points is large then M < N
    """
    from sklearn.neighbors import NearestNeighbors
    import networkx as nx
    import numpy as np
    clf = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)
    #order = list(nx.dfs_postorder_nodes(T,source = src))
    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
    mindist = np.inf
    minidx = 0
    for i in range(len(points)):
        p = paths[i]           # order of nodes
        ordered = points[p]    # ordered nodes
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = i
    opt_order = paths[minidx]
    return opt_order

