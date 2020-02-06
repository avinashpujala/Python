# -*- coding: utf-8 -*-
"""
Created on Tue Aug 04 14:06:13 2015

@author: pujalaa
"""


class AlignTimeseries(object):
    """
    Class for aligning two timeseries of variable length.
    Parameters
    ----------
    x: array, (N,)
    Reference timeseries used as a template to align other timeseries
    """

    def __init__(self, x, preserve_polarity: bool = False):
        self.ref = x
        self.preserve_polarity = preserve_polarity

    def fit(self, y):
        """
        Computes the translational shifts and the sign of the multiplier to
        produce the best correlations between a collection of timeseries
        and the reference signal specified at class initialization
        Parameters
        ----------
        y: array of shape (N,T) or list of len (N)
            Collection of signals to compute best correlation parameters for.
            If array, N = number of timeseries, and T = lengths of the
            timeseries. If list, then the N timeseries can be of varying
            length.
        preserve_polarity: bool
            If true, does not flip the timeseries in obtaining the best shift
            parameters.
        Returns
        -------
        self: class object
            Class object With the following added attributes that can be used
            for alignment of signals to reference using the method
            self.transform
            self.shifts: Translational shift for best correlations.
            self.correlogram: array-like
                Correlograms of each of the timeseries
            self.lag: array-like
                Lags for each of the timeseries.
            self.polarity: array-like
                The signs (i.e., +1 or -1) by which the signal were multiplied
                to produce the best correlations.
            self.shift: array-like
                Translation shifts for producing best correlations.
        """
        import numpy as np
        import dask
        from scipy.signal import correlate

        def corr(a, b):
            y = correlate(a, b, mode='full')
            y = y/(np.linalg.norm(a)*np.linalg.norm(b))
            return y

        def get_corr_params(a, b, preserve_polarity):
            c = corr(a, b)
            lags = np.arange(-len(b)+1, len(a))
            ind_max = np.argmax(np.abs(c))
            pol = np.sign(c[ind_max])
            shift = lags[ind_max]
            return c, lags, shift, pol

        if isinstance(y, np.ndarray):
            if np.ndim(y) == 1:
                y = y[np.newaxis, :]
        foo = [dask.delayed(get_corr_params)(self.ref, y_,
                                             self.preserve_polarity)
               for y_ in y]
        foo = dask.compute(*foo)
        correlogram, lag, shift, polarity = zip(*foo)
        if isinstance(y, np.ndarray):
            correlogram = np.squeeze(np.array(correlogram))
            lag = np.squeeze(np.array(lag))
        self.correlogram = correlogram
        self.lag = lag
        self.shift = np.squeeze(np.array(shift))
        self.polarity = np.squeeze(np.array(polarity))
        return self

    def get_correlations(self, y, ref=None):
        """
        Align signals and get correlations to provided reference signal or mean
        """
        from dask import delayed, compute
        import numpy as np
        if hasattr(self, 'signals_aligned_'):
            y_aligned = self.signals_aligned_
        else:
            y_aligned = self.transform(y)

        if ref is None:
            ref = y_aligned.mean(axis=0)

        c = compute(*[delayed(np.corrcoef)(y_, ref)[0, 1] for y_ in y])
        c = np.array(c)
        self.correlations_ = c
        return c

    def transform(self, y):
        """
        Applies the best correlation parameters computed by self.fit to
        transform the timeseries in y.
        """
        import numpy as np
        import dask
        if np.ndim(y) == 1:
            y = y[np.newaxis, :]
        if self.preserve_polarity:
            polarity = np.ones_like(self.polarity,)
        else:
            polarity = self.polarity
        shift = self.shift
        y_fit = []
        for i, y_ in enumerate(y):
            y_fit.append(polarity[i]*dask.delayed(np.roll)(y_, shift[i]))
        y = np.squeeze(np.array(dask.compute(*y_fit)))
        self.signals_aligned_ = y
        return y


def broadcastBack(X, collapsedAxis, initialShape):
    """
    Given an array, broadcasts it back to the specified intial shape by
    assuming it was collapsed along a specified dimension. It broadcasts it back
    by tiling (np.tile).
        This can be useful for certain operations, such as for
    instance taking the mean along a certain axis of X and subtracting this
    from all the values in X.
    Parameters
    -----------
    X- Array to broadcast back.
    collapsedAxis - Axis along which an array was collapsed to get X
    initialShape - The initial shape of the array, prior to collapse
    Returns
    -------
    Y - Array of shape = intialDims, which consists of tiled X.
    """
    import numpy as np
    initialShape = np.array(initialShape)
    X = np.expand_dims(X, axis=collapsedAxis)
    vec = np.arange(len(initialShape))
    sd = np.setdiff1d(vec, collapsedAxis)
    initialShape[sd] = 1
    return np.tile(X, initialShape)


def causalConvWithSemiGauss1d(y, n):
    """
    Causal convolution of a timeseries with a semi-gaussian kernel
    Parameters
    ----------
    y: array, (N,)
        Timseries to convolve
    n: scalar
        Kernel length in points
    Returns
    -------
    y_conv: array, (N,)
        Convolved signal
    """
    from scipy.signal import convolve
    from apCode.SignalProcessingTools import gausswin
    ker = gausswin(int(2*n))
    ker = ker[int(len(ker)/2+1):]
    ker = ker/ker.sum()
    return convolve(y, ker, mode='full')[:len(y)]


class emd(object):
    """
    Functions and methods related to Empirical Mode Decomposition
    """

    def envelopesAndImf(x, n_comps=1, interp_kind='slinear', triplicate: bool = False):
        """
        Given an timseries iteratively performs the
        steps involved in Empirical Mode Decomposition, upto the specified number
        of components and returns info such as the IMF, and different envelopes
        Parameters
        -----------
        x - array, (N,)
        Returns
        -------
        out: List of dictionaries, (nComponents,)
            Each element of the list corresponds to information obtained from EMD
            at the level of a particular component.
            Each dictionary has to following key-value pairs
            'imf': Intrinsic mode function
            'pks/crests': Crest peaks
            'pks/troughs': Trough peaks
            'envelopes/crests': Crests envelope
            'envelopes/troughs': Troughs envelope
            'envelopes/mean': Mean of crests and troughs envelopes
            'envelopes/diff': Difference of the above envelopes
            'envelopes/max': Max of the envelopes
        """
        import numpy as np
        import scipy as sp
        from apCode.SignalProcessingTools import timeseries as ts

        def getEnvelopesAndImf(x, interp_kind: str = 'cubic', triplicate: bool = False):
            if triplicate:
                x = ts.triplicate(x)
                indVec = np.arange(len(x))
            pks_min = findPeaks(x, pol=-1)[0]
            pks_max = findPeaks(x, pol=1)[0]
            xp = np.hstack((0, pks_min.ravel(), len(x)))
            fp = np.hstack((x[0], x[pks_min], x[-1]))
            xx = np.arange(0, len(x))
            f = sp.interpolate.interp1d(xp, fp, kind=interp_kind)
            env_troughs = f(xx)
            xp = np.hstack((0, pks_max.ravel(), len(x)))
            fp = np.hstack((x[0], x[pks_max], x[-1]))
            f = sp.interpolate.interp1d(xp, fp, kind=interp_kind)
            env_crests = f(xx)
            env_mean = 0.5*(env_troughs + env_crests)
            imf = x-env_mean
            env_diff = 0.5*(env_crests-env_troughs)
            env_max = np.max(np.abs(np.vstack((env_troughs, env_crests))), axis=0)
            if triplicate:
                imf, env_troughs, env_crests, env_mean, env_diff, env_max = \
                    [ts.middleThird(_) for _ in (imf, env_troughs, env_crests,
                                                 env_mean, env_diff, env_max)]
                ivm = ts.middleThird(indVec)
                pks_min = np.take(pks_min, np.where((pks_min >= ivm[0]) & (pks_min <= ivm[-1]))[0])
                pks_min = pks_min - ivm[0]
                pks_max = np.take(pks_max, np.where((pks_max >= ivm[0]) & (pks_max <= ivm[-1]))[0])
                pks_max = pks_max - ivm[0]
            return imf, pks_min, pks_max, env_troughs, env_crests, env_mean, env_diff, env_max
        y = []
        for cmp in np.arange(n_comps):
            foo = getEnvelopesAndImf(x, interp_kind=interp_kind, triplicate=triplicate)
            pks = dict(troughs=foo[1], crests=foo[2])
            env = dict(troughs=foo[3], crests=foo[4], mean=foo[5],
                       diff=foo[6], max=foo[7])
            dic = dict(imf=foo[0], pks=pks, env=env)
            y.append(dic)
            x = foo[5]
        if len(y) == 1:
            y = y[0]
        return y

    def emd(x, nIMF=3, stoplim=.001):
        """Perform empirical mode decomposition to extract 'niMF' components
        out of the signal 'x'.

        Author: Scott Cole
        https://github.com/srcole/binder_emd/blob/master/emd.ipynb
        """
        import numpy as np
        import scipy as sp

        def _emd_comperror(h, mean, pks, trs):
            """Calculate the normalized error of the current component"""
            samp_start = np.max((np.min(pks), np.min(trs)))
            samp_end = np.min((np.max(pks), np.max(trs))) + 1
            return np.sum(np.abs(mean[samp_start:samp_end]**2)) / np.sum(np.abs(h[samp_start:samp_end]**2))

        def _emd_complim(mean_t, pks, trs):
            samp_start = np.max((np.min(pks), np.min(trs)))
            samp_end = np.min((np.max(pks), np.max(trs))) + 1
            mean_t[:samp_start] = mean_t[samp_start]
            mean_t[samp_end:] = mean_t[samp_end]
            return mean_t

        r = x
        t = np.arange(len(r))
        imfs = np.zeros(nIMF, dtype=object)
        for i in range(nIMF):
            r_t = r
            is_imf = False
            while is_imf == False:
                # Identify peaks and troughs
                pks = sp.signal.argrelmax(r_t)[0]
                trs = sp.signal.argrelmin(r_t)[0]

                # Interpolate extrema
                pks_r = r_t[pks]
                fip = sp.interpolate.InterpolatedUnivariateSpline(pks, pks_r, k=3)
                pks_t = fip(t)

                trs_r = r_t[trs]
                fitr = sp.interpolate.InterpolatedUnivariateSpline(trs, trs_r, k=3)
                trs_t = fitr(t)

                # Calculate mean
                mean_t = (pks_t + trs_t) / 2
                mean_t = _emd_complim(mean_t, pks, trs)

                # Assess if this is an IMF (only look in time between peaks and troughs)
                sdk = _emd_comperror(r_t, mean_t, pks, trs)

                # if not imf, update r_t and is_imf
                if sdk < stoplim:
                    is_imf = True
                else:
                    r_t = r_t - mean_t

            imfs[i] = r_t
            r = r - imfs[i]
        return imfs


def chebFilt(X, dt, Wn, order=2, ripple=0.2, btype='bandpass'):
    """ Filter a signal using a non-phase shifting Chebyshev filter
    Inputs:
    dt = Sampling interval
    Wn - Filter range, upto a 2-element array-type
    btype = ['bandpass'] | 'lowpass' | 'highpass' |'bandstop'
    """
    from scipy import signal
    import numpy as np
    nyqFreq = (1/dt)*0.5
    Wn = np.array(Wn)/nyqFreq
    if np.size(Wn) > 1:
        Wn[Wn == 0] = 1e-4
    b, a = signal.cheby1(order, ripple, Wn, btype=btype.lower())
    if np.ndim(X) > 1:
        y = signal.filtfilt(b, a, X, axis=0)
    else:
        y = signal.filtfilt(b, a, X)
    return y


def findOnAndOffs(x, thr):
    """
    Given a signal and a threshold value, estimates and returns the onset and offset indices of events
        that cross the threshold. The algorithm assumes the signal has been detrended beforehand to that
        it has a stable baseline. If not, detrend before passing as input.
    Parameters:
    x - Time series vector
    thr - Threshold for defining events, i.e. regions of the signal above the threshold are considered
        events
    Returns:
    onOffInds - Array-like of shape (2,N), where N is the number of events. The 1st and 2nd rows are the
        presumed onsets and offsets of the events
    """
    import numpy as np
    inds_lvl = levelCrossings(x, thr)
    x_chopped = x.copy()
    x_chopped[np.where(x_chopped >= thr)[0]] = thr
    baseline = np.median(x_chopped)
    inds_baseline = levelCrossings(x, baseline)
    #inds_onOff = np.zeros(np.shape(inds_lvl))

    def getBefore(ind, inds):
        inds = np.delete(inds, np.where(inds >= ind))
        return inds[np.argmax(inds-ind)]

    def getAfter(ind, inds):
        inds = np.delete(inds, np.where(inds <= ind))
        return inds[np.argmin(inds-ind)]

    inds_on = [getBefore(ind, inds_baseline[0]) for ind in inds_lvl[0]]
    inds_off = [getAfter(ind, inds_baseline[1]) for ind in inds_lvl[1]]
    inds_onOff = np.array([inds_on, inds_off])
    return inds_onOff


def findPeaks(y, thr=0, minPkDist=0, thrType='abs', pol=0):
    '''
    Find peaks in a timeseries signal

    Parameters
    ----------
    y = signal
    thr = amplitude threshold below which to ignore peaks
    minPkDist = min distance between peaks
    thrType - Type of amplitude estimation; 'abs' or 'rel'.
        'abs' - Absolute peak amplitude
        'rel' - Relative peak amplitude obtained by detecting all peaks and
            subtracting spline interpolation
    pol = 0, 1, or -1; The polarity of the peaks; if pol==0, the returns
        both maxima and minima, if 1 then only returns maxima, and
        if -1 then returns only minima

    Returns
    -------
    pks - 2 element list with peak indices as well as the signal values at
        peaks.
    '''
    import numpy as np
    from scipy import interpolate as interp

    def findAllPeaks(y):
        y = np.array(y)
        dy = np.diff(y)
        pkInds = np.where((dy[0:-1] > 0) & (dy[1::] <= 0))[0]
        pkInds = np.array(pkInds) + 1
        return pkInds

    # ---If thrType is 'rel' then subtract the mean of the maximal and minimal
    #  envelopes from the signal
    if thrType.lower() == 'rel':
        pkInds_max = findAllPeaks(y)
        pkInds_min = findAllPeaks(-y)
        inds = np.concatenate(([0], pkInds_min, [len(y)-1])).astype(int)
        f = interp.interp1d(inds, y[inds])
        y_minEnv = f(np.arange(len(y)))
        inds = np.concatenate(([0], pkInds_max, [len(y)-1])).astype(int)
        f = interp.interp1d(inds, y[inds])
        y_maxEnv = f(np.arange(len(y)))
        y_spline = (y_minEnv + y_maxEnv)/2
        y = y-y_spline

    # ---Find either the maxima, the minima or both, depending on specified
    #   polarity
    if pol == 0:
        pkInds_max = findAllPeaks(y)
        pkInds_min = findAllPeaks(-y)
        pkInds = np.union1d(pkInds_max, pkInds_min)
    elif pol == 1:
        pkInds = findAllPeaks(y)
    elif pol == -1:
        pkInds = findAllPeaks(-y)

    # -- Filter peaks based on threshold
    pkInds = np.delete(pkInds, np.where(np.abs(y[pkInds]) < thr))

    # --- Filter peaks based on minimum distance between peaks
    if (minPkDist > 0) & len(pkInds > 0):
        pk_first = [pkInds[0]]
        dp = np.diff(pkInds)
        pkInds = pkInds[np.where(dp >= minPkDist)[0]+1]
        if len(pkInds) > 0:
            pkInds = np.union1d(pk_first, pkInds)
        else:
            pkInds = pk_first
    pkAmps = y[pkInds]
    return pkInds, pkAmps


def gaussFun(x, mu=0, sigma=1):
    """
    Returns a gaussian kernel either of a specified number of points
    or over a time domain. The mean and standard deviation can be specified

    Parameters
    ----------
    x: Scalar or 1D array
        If scalar, then number of points in the gaussian, else a
        sequence of times over which to compute the gaussian
    mu: Scalar
        Mean of the gaussian
    sigma: Scalar
        Standard deviation of the gaussian

    Returns:
    y: 1D array
        Gaussian kernel
    """
    import numpy as np
    if isinstance(x, int):
        x = np.arange(x)
        x = x-np.median(x)+mu
    sigma = sigma*(np.max(x)-np.min(x))/6
    x = (x-mu)/sigma
    y = (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*(x**2))
    return y


def gausswin(n, sigma=1.5):
    """
    Returns an n-point gaussian kernel of standard deviation = sigma.
    """
    import numpy as np
    #N = np.arange(n)
    #N = N - np.median(N)
    N = np.linspace(-2*np.pi, 2*np.pi, n)
    g = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-N**2/(2*sigma**2))
    return g


def generateEPSP(t, tau_rise, tau_decay, A, translation):
    """
    Given a time vector t, and the rise(tau_rise) and decay
    constants(tau_decay) in the same units, returns an epsp like
    vector of unit amplitude, and the peak time as a tuple
    Inputs:
    t - independent variable (e.g., time)
    tau_rise - Rise time constant
    tau_decay - Decay time constant
    A- Amplitude
    translation - Translates the epsp in time; positive values shift right
    Ref: Roth and Rossum
    http://homepages.inf.ed.ac.uk/mvanross/reprints/roth_mvr_chap.pdf
    """
    import numpy as np

    translation = int(translation)
    if translation >= 0:
        t = t-t[translation]
        t[t < 0] = 0
    else:
        t = t + t[translation]

    epsp = np.exp(-t/tau_decay)-np.exp(-t/tau_rise)
    #t_pk = t[0] + ((tau_decay*tau_rise)/(tau_decay-tau_rise))*np.log(tau_decay/tau_rise)
    #f = 1/(-(np.exp(-(t_pk-t[0])/tau_rise) + np.exp(-(t_pk-t[0])/tau_decay)))
    #epsp = f*epsp
    epsp = A*(epsp-epsp.min())/(epsp.max()-epsp.min())
    return epsp


class interp():
    def oknotP(n, k=4, l=1, eta_max=0):
        """
        Generates a B-spline uniform periodic knot vector

        Parameters
        ----------
        n: Scalar
            The number of defining polygon vertices
        k: Scalar
            Order of the basis function.
            order = degree + 1, so for a cubic, k = 4
        l: Scalar
            Arbitrary length.
        eta_max: Scalar
            Max noise?

        Returns
        -------
        x: 1D array
            The knot vector

        ** See the docstring in oknotP.m for more details. For me, I will mostly be using this to make my function
        fitBSplineToCurve.py work, because this requires oknotP.
        This code has been adapted from Fontaine et al.(2008)

        References:
            Fontaine, E., Lentink, D., Kranenbarg, S., Müller, U.K., van Leeuwen, J.L., Barr, A.H., and Burdick, J.W. (2008).
            Automated visual tracking for studying the ontogeny of zebrafish swimming. J. Exp. Biol. 211, 1305–1316.

        """
        import numpy as np

        x = np.zeros((1, n+k))
        #epsilon = 0;
        mid = np.linspace(-eta_max-l, eta_max, n-k+2)
        delta = np.sum(l)/(n-k+2-1)
        beg = mid[0] + np.arange(-1*(k-1), 0)*delta
        endd = mid[-1] + np.arange(1, k)*delta
        x = np.hstack((beg, mid, endd))
        return x

    def nanInterp1d(x, kind='cubic'):
        """
        Interpolates a 1d signal that has NaN values in it
        Parameters
        ----------
        x: array, (N,)
            Timeseries signal
        kind: scalar
            Type of interpolation. See scipy.interpolate.interp1d

        Returns
        -------
        xx: array, (N,)
            Interpolated signal without NaNs.
        """
        import numpy as np
        from scipy.interpolate import interp1d
        nonNanInds = np.where(np.isnan(x) == False)[0]
        tt = np.linspace(0, 1, len(x))
        t_ = tt[nonNanInds]
        x_ = x[nonNanInds]
        xx = interp1d(t_, x_, kind=kind)(tt)
        return xx

    def downsample_and_interp2d(X, ds_ratio=(0.5, 0.5), **kwargs):
        """
        Downsamples the 2D-array X along each axis based on the specified downsampling
        ratios and returns interpolated array
        Parameters
        ----------
        X: array (M,N)
            Array to downsample and interpolate
        ds_ratio: 2-tuple
            ds_ratio[0], and ds_ratio[1] are the downsampling ratios along the x-
            and y-axes respectively. For instance, ds_ratio[0] = 0.5 means
            the downsampling results in half the number of points (i.e., N/2) along
            the x-axis
        **kwargs: Keyword arguments for scipy.interpolate.griddata
        Returns
        -------
        X_interp: array, (M, N)
            Array resulting from downsampling and interpolation
        """
        import numpy as np
        from scipy.interpolate import griddata
        kwargs['method'] = kwargs.get('method', 'cubic')
        kwargs['rescale'] = kwargs.get('rescale', True)
        x_coords = np.linspace(0, X.shape[1]-1, int(X.shape[1]*ds_ratio[0])).astype(int)
        y_coords = np.linspace(0, X.shape[0]-1, int(X.shape[0]*ds_ratio[1])).astype(int)
        gy, gx = np.meshgrid(np.arange(X.shape[1]), np.arange(X.shape[0]))
        C = []
        for r in y_coords:
            for c in x_coords:
                C.append([r, c])
        C = np.array(C).T
        coords = (C[0], C[1])
        X_interp = griddata(coords, X[coords], (gx, gy), **kwargs)
        return X_interp

    def downsample_gradients_and_interp2d(X, ds_ratio=(0.5, 0.5), **kwargs):
        """
        See downsample_and_interp2d. The difference here is that the gradients of
        X are taken first and the downsampling and interpolation are applied to these
        before they are integrated along the relevant axes and combined to result
        in a reconstructed version of the original array
        """
        import numpy as np
        dy, dx = np.gradient(X)
        dx_down = interp.downsample_and_interp2d(dx, ds_ratio=(ds_ratio[0], 1), **kwargs)
        dy_down = interp.downsample_and_interp2d(dy, ds_ratio=(1, ds_ratio[1]), **kwargs)
        x = np.cumsum(dx_down, axis=1)
        y = np.cumsum(dy_down, axis=0)
        X_interp = (x+y)*0.5
        return X_interp


def levelCrossings(x, thr=0):
    """
    Given a timeseries signal and a threshold, returns indices where signal crosses
    the threshold in both directions.

    Parameters:
    x - Timeseries signal
    thr - Threshold for crossing
    """
    import numpy as np
    inds = []
    inds.append(np.where((x[0:-1] < thr) & (x[1:] > thr))[0])
    inds.append(np.where((x[0:-1] > thr) & (x[1:] < thr))[0])
    return inds


class linalg():
    def angleBetweenVecs(v1, v2):
        import numpy as np
        theta = np.arccos(np.dot(v1.ravel(), v2.ravel())/np.linalg.norm(v1)/np.linalg.norm(v2))
        return theta

    def orthogonalize(v1, v2):
        """
        Returns the orthogonal complement of a vector or set of vectors w.r.t
            another vector or set of vectors. Note that the orthogonal complement
            returned is not orthogonal to the span of the target vectors, but rather
            the returned vectors are orthogonal to each of the target vectors
        Parameters:
        v1 - Vectors to orthogonalize w.r.t
        v2 - Vectors to orthogonalize
        """
        import numpy as np
        v1 = np.c_[v1]
        v2 = np.c_[v2]
        num = np.dot(v1.T, v2)
        den = np.diag(np.dot(v1.T, v1))
        V_perp = [[vec2-(vec1*num[vNum1, vNum2]/den[vNum1])
                   for vNum2, vec2 in enumerate(v2.T)] for vNum1, vec1 in enumerate(v1.T)]
        return np.squeeze(V_perp).T

    def orthogonalizeOnSpace(V, W):
        """
        Given two vector spaces, returns perpendicular component of the second
            space with respect to the first
        Parameters:
        V - Matrix-like collection of vectors that spans some space and onto which
            another space is to be projected to obtain the perpendicular component.
            Shape must be (m, n) where m gives the dimensions of the vectors
            and n is the number of vectors
        W - Matrix-like collection of vectors that are to be projected onto another
            space, and whose perpendicular component is to be computed. As in V,
            shape must be (m, n)
        """
        import numpy as np
        foo = [w-np.dot(V, np.linalg.pinv(V)).dot(w) for w in np.c_[W].T]
        return np.squeeze(foo)

    def orthogonalBasis(X):
        """
        When given an array where each column is a vector, returns a list of two arrays. In
        the first array, each column is the result of orthogonalization w.r.t. to the other
        space spanned by the other columns in the original array. In the second array, the
        first column is orthogonalized w.r.t to the space spanned by the other columns in the
        original array, whereas the other columns are sequentially subjected to the same process
        w.r.t to the transformed array.
        Parameters:
        X - Array-like of shape (M, N) where N is the number of vectors, and M is the dimensionality
            of the vectors
        Returns:
        Y - List of length = 2, where each item is one of the transformed arrays.
        """
        import numpy as np
        X1 = np.zeros(np.shape(X))
        X2 = X.copy()
        vecInds = np.arange(np.shape(X)[1])
        for jj in vecInds:
            otherInds = np.setdiff1d(vecInds, jj)
            X1[:, jj] = linalg.orthogonalizeOnSpace(X[:, otherInds], X[:, jj])
            X2[:, jj] = linalg.orthogonalizeOnSpace(X2[:, otherInds], X2[:, jj])
        return X1, X2

    def orthonormalize(V):
        """
        Given a space V, returns the orthonormalized version using Gram-Schmidt
        """
        import numpy as np
        V_orth = V.copy()
        V_orth[:, 0] = V_orth[:, 0]/np.linalg.norm(V_orth[:, 0])
        N = np.shape(np.c_[V_orth])[1]
        for n in np.arange(1, N):
            V_orth[:, n] = linalg.orthogonalizeOnSpace(V_orth[:, :n], V_orth[:, n])
            V_orth[:, n] = V_orth[:, n]/np.linalg.norm(V_orth[:, n])
        return V_orth

    def orthoNormalizeAgainstMean(X):
        """
        Given an array where each column represents a vector, returns a new array
        where the columns represent the orthonormal basis against the mean of the original
        vectors
        Parameters:
        X - Array of shape (M, N) where N is the number of vectors, and M is the dimensionality of
        the vector
        """
        import numpy as np
        X_mean = np.c_[np.mean(X, axis=1)]
        X_new = linalg.orthonormalize(np.concatenate((X_mean, X), axis=1))[:, 1:]
        return X_new

    def project(v1, v2):
        """
        Return the projection of a vector or a set of vectors onto another vector
            or set of vectors
        Parameters:
        v1 - Vector or matrix-like collection of column vectors;
            Vectors to project onto. Note that the projection happens vector by
            vector and not onto the space spanned by the vector collection
        v2 - Vector or matrix-like collection of column vectors; These are the vectors
            that are actually projected.
        """
        import numpy as np
        v1 = np.c_[v1]
        v2 = np.c_[v2]
        num = np.dot(v1.T, v2)
        den = np.diag(np.dot(v1.T, v1))
        V_proj = [[vec1*num[vNum1, vNum2]/den[vNum1]
                   for vNum2, vec2 in enumerate(v2.T)] for vNum1, vec1 in enumerate(v1.T)]
        return np.squeeze(V_proj)

    def projectOnSpace(V, W):
        """
        Given two vector spaces, returns projection of one on the other
        Parameters:
        V - Matrix-like collection of vectors that spans some space and onto which
            another space is to be projected. Shape must be (m, n) where m gives
            the dimensions of the vectors and n is the number of vectors
        W - Matrix-like collection of vectors that are to be projected onto another
            space. As in V, shape must be (m, n)
        """
        import numpy as np
        foo = [np.dot(V, np.linalg.pinv(V)).dot(w) for w in np.c_[W].T]
        return np.squeeze(foo)

    def unitize(X, axis=None):
        """
        Unitizes a vector or tensor by dividing by the L2 norm (numpy.linalg.norm).
        Can also unitize along specified axis
        Parameters:
        X - Array-like; data to unitize
        axis - Axis along which to unitize; default is None.

        """
        import numpy as np
        if axis == None:
            X_unit = X/np.linalg.norm(X)
        else:
            X_unit = X/broadcastBack(np.linalg.norm(X, axis=axis), axis, np.shape(X))
        return X_unit


def matchInd(matchVal, matchingVec):
    """
    Helper function for nearestMatchingInd. This is defined within there, but when
    trying to parallelize, unless defined outside, cannot seem to pickle. It does the
    same thing as nearestMatchingInd, but for a single value instead of a vector
    """
    import numpy as np
    ind = np.argmin(np.abs(matchVal-matchingVec))
    return ind


def nearestMatchingInds(referenceVec, matchingVec, n_jobs=32):
    '''
    Given two vectors returns indices in the 2nd vector with values closest to
    those in first.
    Parameters
    ----------
    referenceVec: array, (N,)
        Reference vector
    matchingVec: array, (M,)
        Vector whose entries closest to each element of referenceVec are found
        and their indices returned
    Returns
    -------
    inds: array, (N,)
        Indices in matchingVec such that matchingVec[inds] are the closest
        to referenceVec
    '''
    import numpy as np
    if n_jobs == 1:
        inds = np.zeros(np.shape(referenceVec))*np.nan
        chunkSize = int(0.25 * len(referenceVec))
        for num, x in enumerate(referenceVec):
            inds[num] = matchInd(x, matchingVec)
            if np.mod(num, chunkSize) == 0:
                print(np.round((num/len(referenceVec))*100), '% complete')
    else:
        from joblib import Parallel, delayed
        import os
        nCores = int(os.cpu_count()*0.8)
        nCores = np.min([nCores, n_jobs])
        try:
            inds = Parallel(n_jobs=nCores, verbose=0)(delayed(matchInd)(x, matchingVec)
                                                      for x in referenceVec)
        except:
            import dask
            inds = [dask.delayed(matchInd)(x, matchingVec) for x in referenceVec]
            inds = dask.compute(*inds)
    return np.array(inds).astype(int).ravel()


def mapToRange(x, mapRange):
    """
    Given a set of values, x, maps them to the range, mapRange, and returns
    output y such that for any 2 values, g1, g2 in x, and their corresponding
    values f(g1), f(g2) in y, g1/g2 = f(g1)/f(g2).
    Inputs:
    x - Values to map
    mapRange - 2 element array (e.g., [10, 100]) indicating the range to which
        x is to be mapped
    """
    from apCode.SignalProcessingTools import standardize
    import sys
    if len(mapRange) != 2:
        print('mapRange must have 2 elements!')
        sys.exit()
    y = standardize(x)
    y = y*(mapRange[1]-mapRange[0]) + mapRange[0]
    return y


def nextPow2(x):
    '''
    y = nextPow2(x)
    Returns y,  the next power of 2 such that 2**y > x, and 2**(y-1) <= x;
    Similar to the matlab function 'nextpow2'

    '''
    import numpy as np

    if np.size(x) > 1:
        x[np.where(x == 0)] = 1
        y = np.array([np.ceil(np.log2(elem)) for elem in x])
    else:
        if x == 0:
            x = 1
        y = int(np.ceil(np.log2(x)))
    return y


def segmentByEvents(X, eventIndices, nPreEventPts, nPostEventPts, axis=0, n_jobs=1,
                    verbose=0):
    """
    Segments a timeseries or a collection of timeseries based on a set of event
    indices
    Parameters
    X - Array-like; data to be segmented.
    eventIndices - Array or list; set of indices around which to segement data.
    nPreEventPts - Scalar; number of points to collect before each event. If,
        for a segment, there are not enough pre-event points then takes the maximum
        number available.
    nPostEventPts - Scalar; number of points to collect after each event (see above)
    axis - Scalar; If X is an array, the axis specifies the axis along which to
        segment the data
    """
    import numpy as np
    eventIndices = eventIndices.astype(int)

    def segmentByEvent(X, eventIdx, nPreEventPts, nPostEventPts, axis=0):
        #    inds = np.r_[eventIdx-nPreEventPts:eventIdx+nPostEventPts]
        inds = np.arange(eventIdx-nPreEventPts, eventIdx+nPostEventPts)
        inds = np.delete(inds, np.where(inds < 0))
        inds = np.delete(inds, np.where(inds >= np.shape(X)[axis]))
#        delInds = np.setdiff1d(np.arange(np.shape(X)[axis]),inds)
#        y = np.delete(X,delInds,axis = axis)
        y = np.take(X, inds, axis=axis)
        return y
    if n_jobs < len(eventIndices):
        Y = []
        for evt in eventIndices:
            y_now = segmentByEvent(X, evt, nPreEventPts, nPostEventPts, axis=axis)
            Y.append(y_now)
    else:
        from joblib import Parallel, delayed
        from multiprocessing import cpu_count
        n_jobs = np.min([cpu_count(), n_jobs])
#        n_jobs = np.max((int((cpu_count()/2)),2))
        Y = Parallel(n_jobs=n_jobs)(delayed(segmentByEvent)
                                    (X, evt, nPreEventPts, nPostEventPts, axis=axis) for evt in eventIndices)
    return Y


class spectral():
    import apCode.spectral.wavelet as wave

    def fft(y, dt=1, hamming=True):
        """
        Returns the amplitude spectrum, and frequency values
        """
        import numpy as np
        #import apCode.SignalProcessingTools as spt
        nfft = 2**nextPow2(len(y))
        n = int(nfft/2)
        if hamming:
            y = np.multiply(y, np.hamming(len(y)))
            p = 2*np.abs(np.fft.fft(y, n=nfft))[:n]/n
        else:
            p = np.abs(np.fft.fft(y, n=nfft))[:n]/n

        f = np.fft.fftfreq(nfft)[:n]*(1/dt)
        #f = np.linspace(0,1,nfft/2)*(1/dt)*0.5
        return p, f


class stats(object):
    def ci(data, confidence=0.95):
        """
        Returns the confidence interval for a data set, at specified interval
        Parameters
        ---------
        data - Array-like; Data for which to compute confidence interval
        confidence - Scalar; Interval size

        Returns
        -------
        h - Scalar; confidence interval = [mean-h, mean+h]

        """
        import numpy as np
        import scipy as sp
        import scipy.stats

        a = 1.0*np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
        return h

    def iHist(hist, bins, N=None):
        """
        Given a histogram and bins, returns a dataset that would lead to the
        specified histogram ("inverse histogram", if you will)
        """
        import numpy as np
        X = np.array([0]).reshape((-1, 1))
        if not N:
            N = len(hist)
        hist = N*hist*1./hist.sum()
        hist = hist.astype('int')

        for h, b in zip(hist, bins):
            foo = np.tile(b, (h, 1))
            X = np.concatenate((X, foo), axis=0)
        return np.array(X.ravel())

    def saturateByPerc(X, perc_low=0, perc_up=99, axis=None):
        """
        Saturates an input array at specified lower and upper percentile.
        The saturation can also be done along a specified axis
        Parameters:
        X - Array-like; data to be saturated (e.g., an image)
        perc_low - Percentile at which to saturate at the lower end of values
        perc_up - Percentile at which to saturate at the uppwe end of values
        axis - Axis along which to separate. If None, saturates across all
            dimensions
        """
        import numpy as np
        if np.ndim(X) == 1:
            X = X[:, np.newaxis]
        X_sort = np.sort(X, axis=axis)
        if axis == None:
            thr_up = X_sort[np.int(len(X_sort)*perc_up/100)]
            thr_low = X_sort[np.int(len(X_sort)*perc_low/100)]
            X_perc = (X-thr_low)/(thr_up-thr_low)
            X_perc[X_perc < 0] = 0
            X_perc[X_perc > 1] = 1
            X_perc = X_perc * (thr_up-thr_low) + thr_low
        else:
            shape = np.array(np.shape(X))
            vec = np.arange(len(shape))
            sd = np.setdiff1d(vec, axis)
            shape[sd] = 1
            thr_up = np.take(X, [(shape[axis]*perc_up/100).astype(int)], axis=axis)
            thr_up = np.tile(thr_up, shape)
            thr_low = np.take(X, [(shape[axis]*perc_low/100).astype(int)], axis=axis)
            thr_low = np.tile(thr_low, shape)
            X_perc = (X-thr_low)/(thr_up-thr_low)
            #thr_up = thr_up.ravel()[0]
            #thr_low = thr_low.ravel()[0]
            X_perc[X_perc > 1] = 1
            X_perc[X_perc < 0] = 0
            X_perc = X_perc*(thr_up-thr_low) + thr_low
        return X_perc

    def residues(Y, Y_est):
        """
        Given data and estimates from linear regression, returns
            residues as described in Miri et al.
        Parameters:
        Y: Array of shape (nSamples, nVariables). This is the actual data
        Y_est: Same shape as Y. This is the estimate of Y computed from regression,
            or the like.
        Returns:
        r: residues
        """
        import numpy as np
        nSamples = Y.shape[0]
        r = np.c_[Y-Y_est]
        r = np.linalg.norm(r, axis=0)
        r = np.sqrt(r/nSamples)
        return r

    def rSq(Y, Y_est):
        """
        Returns the R-squared for a fit to some data
        Parameters:
        Y: Array of shape (n_samples, n_variables)
        Y_est: Array of same size as Y. This is the estimate of Y from some GLM
            fit
        Returns:
        Rsq: An array of shape(n_variables,)

        """
        sse = stats.sse(Y, Y_est)
        sst = stats.sst(Y)
        r2 = 1-(sse/sst)
        return r2

    def rSq_adj(Y, Y_est, nFeatures):
        """Adjusted R-squared value (accounts for degrees of freedom)"""
        r2 = stats.rSq(Y, Y_est)
        nSamples = Y.shape[0]
        r2_adj = 1-((1-r2)*(nSamples-1)/(nSamples-nFeatures-1))
        return r2_adj

    def standardError(Y, Y_est, X):
        """ Computes the standard error, an estimate of the standard deviation
        of coefficients in regression problems.
        For reference see:
        https://www2.isye.gatech.edu/~yxie77/isye2028/lecture12.pdf
        Parameters
        ----------
        Y: array, (nSamples, nTargets)
            Array of variables (targets) fitted with regression.
        Y_est: array, (nSamples, nTargets)
            Estimated version of Y (estimated by regression, etc)
        X: array, (nSamples, nFeatures)
            Array of explanatory variables (features) used in regression.
        Returns
        -------
        se : (nTargets,nFeatures)
            Standard error, which when the coefficients matrix (regression
            betas) yields the T-values
        """
        import numpy as np
        nSamples = len(Y)
        nFeatures = X.shape[1]
        mse = (stats.sse(Y, Y_est)/(nSamples-nFeatures-1)).reshape(-1, 1)
#        Sxx= np.sum((X-X.mean(axis = 0))**2,axis = 0).reshape(1,-1)
#        foo = mse@(1/Sxx)
        Sxx = np.linalg.pinv(np.dot(X.T, X)).diagonal().reshape(1, -1)
        se_ = np.sqrt(mse@Sxx)
        return se_

    def sse(Y, Y_est):
        """
        Returns the sum of squared errors
        Parameters:
        Y: Array of shape (n_samples, n_variables)
        Y_est: Array of same size as Y. This is the estimate of Y from some GLM
            fit
        Returns:
        sse: An array of shape(n_variables,)

        """
        import numpy as np
        return np.sum((Y-Y_est)**2, axis=0)

    def sst(Y):
        """
        Returns the total sum of squares
        Parameters:
        Y: Array of shape (n_samples, n_variables)
        Returns:
        ssto: An array of shape(n_variables,)
        """
        import numpy as np
        Y_mean = np.tile(np.mean(Y, axis=0), [np.shape(Y)[0], 1])
        return np.sum((Y-Y_mean)**2, axis=0)

    def kde2D(x, y, bandwidth=1.0, xyBins=[100, 100], xLims=[], yLims=[], **kwargs):
        """
        Build 2D kernel density estimate (KDE).
        Uses KernelDensity from sklearn.neighbors
        Parameters
        x, y - 1D arrays. These give the x- and y- coordinates of the points in
            2D for which the KDE is to be obtained
        bandwidth - [], scalar, or 2-element array. The bandwidth for KDE.
            If bandwidth is not specified or is empty, then automatically estimates
            the best bandwith using GridSearchCV sklearn.model_selection
        xyBins - Scalar. # of bins along the x- and y-dimension respectively
        ybins - Scalar. # of bins along the y-dimension
        xlims,ylims - 2-element array-like. If [],t

        Returns
        X - x meshgrid for the KDE matrix
        Y - y meshgrid for the KDE matrix
        K - KDE matrix
        """

        from sklearn.neighbors import KernelDensity
        from sklearn.decomposition import PCA
        from sklearn.model_selection import GridSearchCV
        import numpy as np

        xBins, yBins = xyBins[0], xyBins[1]
        if np.isreal(xBins):
            xBins = xBins*1j
        if np.isreal(yBins):
            yBins = yBins*1j

        if len(xLims) == 0:
            xmin = x.min()
            xmax = x.max()
        elif type(xLims[0]) == list:
            if len(xLims[0]) == 0:
                xmin = x.min()
            xmax = xLims[1]
        elif type(xLims[1]) == list:
            if len(xLims[1]) == 0:
                xmin = xLims[0]
            xmax = x.max()
        else:
            xmin, xmax = xLims[0], xLims[1]

        if len(yLims) == 0:
            ymin = y.min()
            ymax = y.max()
        elif type(yLims[0]) == list:
            if len(yLims[0]) == 0:
                ymin = y.min()
            ymax = yLims[1]
        elif type(yLims[1]) == list:
            if len(yLims[1]) == 0:
                ymin = yLims[0]
            ymax = y.max()
        else:
            ymin, ymax = yLims[0], yLims[1]

        # create grid of sample locations (default: xBins x yBins)
        xx, yy = np.mgrid[xmin:xmax:xBins, ymin:ymax:yBins]
        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
        xy_train = np.vstack([y, x]).T

        # Automatic bandwidth estimation
        if not bandwidth:
            print('Estimating best bandwidth...')
            xy_pca = PCA(n_components=2, whiten=False).fit_transform(xy_train)
            params = {'bandwidth': np.logspace(-1, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(xy_pca)
            bandwidth = grid.best_estimator_.bandwidth
            print('Best bandwidth = {0}'.format(bandwidth))
        print('Bandwidth = {0}'.format(bandwidth))

        print('Estimating kernel density...')
        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(xy_train)

        # score_samples() returns the log-likelihood of the samples
        z = np.exp(kde_skl.score_samples(xy_sample))
        return xx, yy, np.reshape(z, xx.shape)

    def valAtCumProb(x, bins=None, cumProb=0.99, plotBool=True, func='log'):
        """
        Computes the cumulative probability function of an input array and
        returns its value at a specified cumulative probability. Useful to
        determine thresholds, outliers, etc
        Parameters
        ----------
        x: 1D array
            Input variable for which to compute cumulative probability
        bins: Scalar, array, or string
            Specifies bins for histogram. See np.histogram. IF None, then number
            of bins is the same as the length of the variable.
        cumProb: Scalar
            Cumulative probability at which the value of the input variable is
            to be obtained.
        plotBool: Boolean
            If True, then plots cumulative probability
        func: String, ('log' | '')
            Specifies whether the input variable is to be log-transformed or
            not. If 'log' then takes the natural log of the data before
            computing histogram.
        Returns
        -------
        val: scalar
            The value of the input variable correponding to the specified cumulative
            probability
    """
        import numpy as np
        import matplotlib.pyplot as plt
        def e2c(x): return (x[:-1]+x[1:])/2
        x = np.delete(x, np.where(np.isnan(x)))
        if not bins:
            bins = len(x)
        if func == "log":
            p, bins = np.histogram(np.log(x), bins=bins)
        else:
            p, bins = np.histogram(x, bins=bins)
        bins = e2c(bins)
        p = np.cumsum(p/p.sum())
        valInd = np.argmin(np.abs(p-cumProb))
        if func == 'log':
            val = np.round(np.exp(bins[valInd])*10)/10
        else:
            val = np.round(bins[valInd]*10)/10
        if plotBool:
            plt.figure(figsize=(16, 6))
            plt.fill_between(bins, p, y2=0)
            plt.axvline(x=bins[valInd], linestyle=':', color='k')
            plt.axhline(y=cumProb, linestyle=':', color='k')
            plt.text(bins[valInd]+0.1, 1+0.01, '{}'.format(val), fontsize=16)
            if func == 'log':
                xl = (0, bins.max())
                xLabel = '$\ln(x)$'
            else:
                xl = (bins.min(), bins.max())
                xLabel = '$x$'
            plt.xlim(xl)
            plt.ylim(0, 1.05)
            plt.ylabel('$\Sigma P(x)$')
            plt.xlabel(xLabel)
        return val


def standardize(x, axis=None):
    """
    Given x, which can have any dimensions, standardizes the values so that
    min, max are 0,1 respectively
    Inputs:
    x - Values to standardize, can be an array of any dimensions
    preserveSign = If True, will preserve sign of values, such that zero value is fulcrum.
        In other words, separately standardizes the negative and positive values
    """
    import numpy as np

    def func(x): return (x-np.min(x))/(np.max(x)-np.min(x))
    if axis == None:
        out = func(x)
    else:
        out = np.apply_along_axis(func, axis, x)
    return out


class timeseries():
    from apCode.behavior.FreeSwimBehavior import matchByOnset, alignSignalsByOnset

    def alignSignals(signals, refInd=0, padType='zero'):
        """
        Given a list of signals, pads the signals to the length of the longest
        signal and then aligns them by rolling to produce maximum correlation
        among signals.

        Parameters
        ----------
        signals: list, n
            List of n signals, of variable length
        refInd: integer
            Index of signal within the list of signals to use as the first
            reference signal for alignment
        padType: string
            'edge'|'zero'
            If 'edge', the edge-pads signals, or else zero-pads

        Returns
        -------
        out: dictionary
            Output variable containing keys with relevant values

        """
        import numpy as np
        pad = timeseries.pad
        if not isinstance(signals, list):
            signals = list(signals)
        maxLen = np.max(np.array([len(s) for s in signals]))
        sig_mean = signals[refInd]
        sig_mean = pad(sig_mean, pad_width=(0, maxLen-len(sig_mean)),
                       padType=padType)

        S = np.zeros((maxLen, len(signals)))
        lags, padLens, signs = [np.zeros((len(signals),)) for _ in range(3)]
        for count, s in enumerate(signals):
            foo = timeseries.matchByTranslating(sig_mean, s, padType=padType)
            x = foo['signals'][:, 1]
            S[:len(x), count] = x
            sig_mean = np.mean(S[:, :count+1], axis=1)
            lags[count] = foo['lag_best']
            signs[count] = foo['sign']
            padLens[count] = foo['padLens'][1]

        mu = np.mean(S, axis=1)
        correlations = np.array([np.corrcoef(mu, s)[0, 1] for s in S.T])

        def transform(out, signals, padType=None):
            if padType == None:
                padType = out['padType']
            S = np.zeros_like(out['signals'])
            for count, s in enumerate(signals):
                x = pad(s, pad_width=(0, out['padLens'][count]), padType=padType)
                x = out['signs'][count]*np.roll(x, out['lags'][count])
                S[:, count] = x
            return S

        def padToNan(out, signals):
            maxLen = np.shape(signals)[0]
            S = np.zeros_like(signals)
            for count, s in enumerate(signals.T):
                x = s.copy()
                x[maxLen-out['padLens'][count]:] = np.nan
                S[:, count] = x
            return S

        out = {'lags': lags.astype(int), 'signs': signs, 'padLens': padLens.astype(int),
               'signals': S, 'correlations': correlations,
               'padType': padType, 'transform': transform, 'padToNan': padToNan}
        return out

    def alignSignals_old(X, padType='zero'):
        """
        Given a list of signals, returns an array of signals after alignment.
        If the signals are of variable length, then pads the signals to the length
        of the longest signal.

        Parameters
        ----------
        X: list
            List of signals
        padType: string
            ['zero'] | 'edge'

        Returns
        -------
        Y: array (m,n)
            Array of aligned signals, where m is the length of the longest of
            the input signals, and n is the # of signals
        correlations: array (n,)
            Correlation of each signal to the mean of the aligned signals
        signsAndShifts: list of length (n)
            Each element of the list is 2-tuple with the first element of the
            tuple being the sign by which the original signal had was multiplied
            for alignment, and the 2nd element by the amount by which the signal
            had to be circularly shifted
        """
        import numpy as np
        lens = np.array([len(x) for x in X])
        maxLen = lens.max()+1
        x = X[0]
        lenDiff = maxLen-len(x)
        if padType.lower() == 'zero':
            Y = np.pad(x, (0, lenDiff), 'constant', constant_values=0)
        else:
            Y = np.pad(x, (0, lenDiff), 'edge')
        ref = Y
        count = 0
        signs, shifts = [], []
        for x in X:
            foo = timeseries.matchByTranslating(ref, x, padType=padType)
            sign = np.sign(foo[0][foo[1]])
            signs.append(sign)
            shifts.append(-foo[1])
            tmp = sign*np.roll(foo[2][:, 0], -foo[1])
            if count == 0:
                Y = np.hstack((ref.reshape((-1, 1)), tmp.reshape((-1, 1))))
            else:
                Y = np.hstack((Y, tmp.reshape((-1, 1))))
            count = count + 1
            ref = np.mean(Y, axis=1)
        mu = np.mean(Y, axis=1)
        correlations = np.array([np.corrcoef(mu, vec)[0, 1] for vec in Y.T])
        return Y[:, 1:], correlations, (np.array(signs), np.array(shifts))

    def combineSignalSubsets(signals, n=2, N=None):
        """
        Returns unique combinations from a set of signals
        Parameters
        ----------
        signals: array, (T,N)
            Signal array, where T is the # of time points, and N is the
            number of signals
        n: integer
            Number of signals to combine per combination.
        N: integer or none
            Number of combined signals to return. If None, returns all combinations
        """
        import numpy as np
        from itertools import combinations
        nSignals = np.shape(signals)[1]
        combs = np.array(list(combinations(np.arange(nSignals), n)))
        if N == None:
            N = np.shape(combs[0])
        rp = np.random.permutation(np.shape(combs)[0])
        rp = rp[:N]
        combs = combs[rp, :]
        S = np.zeros((np.shape(signals)[0], N))
        for count, c in enumerate(combs):
            for c_sub in c:
                S[:, count] = S[:, count] + signals[:, c_sub]
            S[:, count] = S[:, count]/len(c)
        out = {'signals': S, 'combs': combs}
        return out

    def dilate(x, dil_range=(1, 1.2, 10)):
        """Given a set of signals and a dilation range
        returns dilated versions of the signals
        with dilation factors randomly sampled from the
        dilation range
        Parameters
        ----------
        x: array-like, (nSignals, nSamples), or list, (nSignals,)
            Timeseries to dilate
        dil_range: tuple, (minDilation, maxDilation, nDilationsInRange)
            Dilation range
        Returns
        ------
        x_dil: array or list
            Dilated timeseries
            """
        from scipy.interpolate import interp1d
        import numpy as np
        dils = np.linspace(*dil_range)
        dils[0] = np.max((dils[0], 1))
        x_dil = []
        for x_ in x:
            dil_ = np.random.choice(dils)
            t = np.linspace(0, 1, len(x_))
            tt = np.linspace(0, 1, int(np.round(len(x_)*dil_)))
            x_dil.append(interp1d(t, x_, kind='cubic')(tt)[:len(x_)])
        if isinstance(x, np.ndarray):
            x_dil = np.array(x_dil)
        return x_dil

    def jitter(x, jitter_range=(-2, 20)):
        """
        Given a set of timeseries, and a jitter range, returns jittered versions of
        the timeseries with the amound of jitter randomly pulled from the jitter range.
        """
        import numpy as np
        rngVec = np.arange(*jitter_range)
        jtr = np.random.choice(rngVec, size=len(x), replace=True)
        x_new = []
        for j, x_ in zip(jtr, x):
            x_new.append(np.roll(x_, j))
        if isinstance(x, np.ndarray):
            x_new = np.array(x_new)
        return x_new

    def matchByTranslating(x, y, padType='edge'):
        """
        Given two 1D arrays (such as timeseries), aligns by circularly shifting one w.r.t
        the other
        Parameters
        ----------
        x: Array (n,)
            Reference timeseries.
        y: Array (k,)
            Timeseries to be rolled (circularly shifted).
        padType: string
            'edge'| 'zero'
            If 'edge' then edge pads signals to equalize their lengths, else zero pads.

        Returns
        -------
        out: dictionary
            Has the following keys
        'sign': scalar
            -1 | 1, Multiplying the timeseries by the sign results in positive correlation
            with reference signa
        'lag_max': scalar
            Signal must be rolled by this much to produce maximum correlation.
        'lags': 1D array (n+k-1,)
            All the lags by which the signals were rolled
        'correlations': 1D array (n+k-1,)
            Correlations at the above lags
        'lenDiffs': array (2,)
            Lengths by which signals were padded; one value will be zero
        'signals': array (2,max(n,k))
            Aligned signals
        """
        import numpy as np
        pad = timeseries.pad
        from scipy.signal import correlate

        c = correlate(x, y, mode='full')/(np.linalg.norm(x)*np.linalg.norm(y))
        lenDiff = len(y)-len(x)

        #nLags = len(x) + len(y)-1
        lags = np.arange(-len(y)+1, len(x))
        #lags = np.arange(nLags)
        #lags = lags-int(np.ceil(np.median(lags)))
        ind_max = np.argmax(np.abs(c))
        sign = np.sign(c[ind_max])
        lag_best = lags[ind_max]
        y = sign*np.roll(y, lag_best)
        padLens = np.array((0, 0)).astype(int)
        if lenDiff > 0:
            x = pad(x, pad_width=(0, lenDiff), padType=padType)
            padLens[0] = lenDiff
        else:
            y = pad(y, pad_width=(0, -lenDiff), padType=padType)
            padLens[1] = -lenDiff

        lags = lags[:len(c)]
        xy = np.array((x, y)).T
        out = {'sign': sign, 'lag_best': lag_best, 'lags': lags,
               'correlations': c, 'padLens': padLens,
               'signals': xy}
        return out

    def matchByTranslatingMats(X, Y):
        """
        Given two matrices (such as matrices of wavelet coefficients),
        translates the 2nd w.r.t to the other in the x-direction only and
        returns the correlations at each x-step, the vector of shifts, and
        the shift that produces the best correlation between the matrices

        Parameters
        ----------
        X: (M,N), array
            1st matrix, that acts as reference
        Y: (M,O), array
            2nd matrix that is shifted w.r.t the 1st in the x direction
            so as to find the best correlation. If the 2nd dimension O!=N
            then pads with edge values to make them of equal length
        """
        import numpy as np
        def corr(A, B): return np.corrcoef(A.ravel(), B.ravel())[0, 1]

        tDiff = np.shape(X)[1]-np.shape(Y)[1]
        if tDiff > 0:
            Y = np.pad(Y, pad_width=((0, 0), (0, tDiff)), mode='constant', constant_values=(0,))
        elif tDiff < 0:
            X = np.pad(X, pad_width=((0, 0), (0, -tDiff)), mode='constant', constant_values=(0,))
        lenShift = np.shape(X)[1]
        corrs, shift = [], []
        for x in np.arange(-lenShift, lenShift):
            corrs.append(corr(X, np.roll(Y, (0, x))))
            shift.append(x)

        out = {'lags': np.array(shift), 'correlations': np.array(corrs),
               'lag_max': shift[np.argmax(np.abs(corrs))]}
        return out

    def middleThird(x):
        """
        Returns the middle third of a timseries
        Parameters:
        x: 1D array
            Timeseries
        Returns
        ------
        x_middle: 1D array
            Middle third of the timeseries x
        """
        import numpy as np
        lenSeg = len(x)/3
        lenSeg_int = int(lenSeg)
        if np.mod(lenSeg, 1) != 0:
            print("Does not evenly divide into three. Rounding")
        return x[lenSeg_int:2*lenSeg_int]

    def pad(x, pad_width=(0, 0), padType='zero'):
        """
        Does more or less the same thing as numpy.pad, except includes a simpler
        way to zero-pad
        """
        import numpy as np
        if padType.lower() == 'zero':
            x = np.pad(x, pad_width=pad_width, mode='constant',
                       constant_values=(0, 0))
        elif padType.lower() == 'rand':
            pre = np.ones((pad_width[0],),) + x[0]
            post = np.ones((pad_width[-1])) + x[-1]
            pre = pre + (np.random.rand(len(pre),)-0.5)*np.std(x)
            post = post + (np.random.rand(len(post),)-0.5)*np.std(x)
            x = np.hstack((pre, x, post))
        else:
            x = np.pad(x, pad_width=pad_width, mode=padType)
        return x

    def resample(y, n, kind='cubic'):
        """
        Returns a resampled signal using interpolation, and no filtering
        Parameters
        ----------
        y: (m,), array
            Signal to be resampled
        n: integer
            New length of the signal after resampling
        kind: string
            Kind of interpolation; see scipy.interpolate.interp1d

        Returns
        -------
        yy: (n,), array
            Resampled signal
        """
        from scipy.interpolate import interp1d
        import numpy as np
        x = np.arange(len(y))
        xx = np.linspace(x[0], x[-1], n)
        f = interp1d(x, y, kind='cubic')
        yy = f(xx)
        return yy

    def triplicate(x, flip=True):
        """
        Triplicates a signal by stitching back to back either by flipping the
        first and third segments for smoother stitching or without flipping
        Parameters
        ---------
        x: 1D array
            Timeseries to triplicate
        flip: Boolean
            If True, then flips the 1st and 3rd segments for continuity. Regardless
            of whether or not flip is True, the middle segment will always look
            like the original signal
        Returns:
        y: 1D array
        Triplicated signal

        See also: middleThird
        """
        import numpy as np
        if flip:
            x = np.hstack((np.flipud(x), x, (np.flipud(x))))
        else:
            x = np.hstack((x, x, x))
        return x


def zeroPadToNextPowOf2(y):
    '''
    Given a signal y of length N1, returns the right zeropadded version of y
    of length N2, where N2 is the next power of 2 greater than log2(N1)
    '''
    import numpy as np
    N = len(y)
    #base2 = np.fix(np.log2(N)+0.4999).astype(int)
    #x = np.hstack((y,np.zeros(2**(base2+1)-N)))
    base2 = nextPow2(N)
    x = np.hstack((y, np.zeros(2**(base2)-N)))
    return x


def zscore(x, axis=None):
    """
    Returns the input dataset in z-score units
    Parameters:
    x - Dataset for which to compute z-score
    axis - Axis along which to compute z-score. If axis = None, then computes
        for the entire dataset.

    Avinash Pujala, JRC, 2017
    """
    import numpy as np
    def func(x): return (x-np.mean(x))/(np.std(x))
    if axis == None:
        out = func(x)
    else:
        out = np.apply_along_axis(func, axis, x)
    return out
