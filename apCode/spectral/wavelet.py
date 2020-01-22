# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:15:30 2015

@author: pujalaa
Adapted from routines written by Christopher Torrence, Gilbert Compo, Aslak Grinsted
References: 
Torrence, C and Compo, G.P.,1998: A Practical Guide to Wavelet Analysis 
<I> Bull. Amer. Meteor. Soc</I>
Grinsted A., Moore J.C., Jevrejeva S., 2004: Application of the cross wavelet
transform and wavelet coherence to geophysical time series <I> Nonlinear 
Processes in Geophysics </I> 

"""
import sys as _sys
_codeDir = r'c:/users/pujalaa/documents/code/python/code'
_sys.path.append(_codeDir)
#sys.path.insert(0,'C:/Users/pujalaa/Documents/Code/Python/code/codeFromNV')
#sys.path.insert(0,'C:/Users/pujalaa/Documents/Code/Python/code/util')
#sys.path.insert(0,'C:/Users/pujalaa/Documents/Code/Python/code/wavelets')
       
def ar1nv(x):
    '''
    Estimate the parameters for an lag-1 autoregression model (AR1) model
    g,a = ar1nv(x)
    Parameters
    ----------
    x - 1D array
        Timeseries
    Returns
    -------
    g - Estimate of lag-1 autocorrelation
    a - Estimate of the noise variance
    '''
    import numpy as np
    m = np.mean(x)
    N = len(x)
    x = x - m
        
    c0 = np.dot(x,x)/N
    c1 = np.dot(x[1:N-1],x[2:N])/(N-1)
    g = c1/c0
    a = np.sqrt((1-g**2)*c0)
    return g,a
    
def ar1spectrum(ar1, period):
    ''' 
    AR1 power spectrum
    power = ar1spectrum(ar1, period)
    Adapted from (c) Aslak Grinsted, 2002-2004
    '''
    import numpy as np
    freq = 1/period
    P = (1-ar1**2)/(np.abs(1-ar1*np.exp(-2*np.pi*1j*freq)))**2
    return P

def chisquare_inv(P,V):
    """
    Inverse of chi-square cumulative distribution function (cdf)
    
    X = chisquare_inv(P,V) returns the inverse of the chi-square cdf with V degrees
        of freedom at fraction P. This means that P*100 % of the distribution lies
        between 0 and X
    To check, the answer should satisfy: P = gammainc(X/2, V/2)
    Uses "fminbound" from scipy.optimize and "chisquare_solve" from current namespace
    
    """
    from scipy.optimize import fminbound
    if (1-P) < 1e-4:
        print('P must be < 0.9999')
    
    if (P == 0.95) & (V==2):
        X = 5.99915 # Apparently, this is a no-brainer
        return X
    
    minn = 0.01
    maxx = 1
    X = 1
    tol = 1e-4
    while ((X + tol) >= maxx): # Apparently, should only need to loop through once
        maxx = maxx*10.
        # This calculates value for X, normalized by V
        
        X = fminbound(chisquare_solve,minn,maxx,xtol=tol,args=(P,V))
        minn = maxx
    X = X*V
    
 
def chisquare_solve(X_guess,P,V):
    """
    Internal (objective) function used by chisquare_inv
    
    pdiff = chisquare_solve(X_guess, P, V)
        Given X_guess, a percentile P, and degrees-of-freedom V, returns the 
        difference between calculated percentile and P
        Uses the function scipy.special.gammainc ("gammainc") in MATLAB
        
        Extra factor of V is necessary because X is normalized
        
    Written By C. Torrence, Jan 1998    
    
    """
    import numpy as np
    from scipy.special import gammainc
    
    P_guess = gammainc(V/2,V*X_guess/2) # Incomplete Gamma function
    
    pdiff = np.abs(P_guess - P) # Errori in calculated P
    
    tol = 1e-4
    
    if P_guess >= (1-tol):
        pdiff = X_guess
    return pdiff
        

def clipWave(W,freq,time,freqRange,timeRange,coi = []):
    '''
    W_clip, freq_clip,time_clip,coi_clip = clipWave(W,freq,time,freqRange, \
        timeRange,coi = [])
    Clips the matrix of wavelet coefficients to the specified freq and time 
        range
    Inputs:
    W - 2D matrix of wavelet coefficients
    freq - Freq vector of length = number of rows of W
    time - Time vector of length = number of cols of W
    freqRange  - 2 element array specifying the min and max freq to clip W to
    timeRange - 2 element array specifying the min and max time to clip W to
    coi - Cone of influence; can be empty
    Outputs:
    W_clip - Clipped W
    freq_clip - Clipped freq
    ...
    '''
    import apCode.SignalProcessingTools as spt
    fInds = spt.nearestMatchingInds(freqRange,freq)
    print(fInds)
    tInds = spt.valsToNearestInds(timeRange,time)    
    W_clip = W[fInds[1]:fInds[0],tInds[0]:tInds[1]]
    freq_clip = freq[fInds[1]:fInds[0]]
    time_clip = time[tInds[0]:tInds[1]]
    if len(coi) >0:
        coi_clip = coi[tInds[0]:tInds[1]]
    else:
        coi_clip = coi
    return W_clip, freq_clip,time_clip,coi_clip

def get_fourier_factor(mother = 'morlet', k0 = -1):
    """
    Function for computing the fourier_factor for a given mother wavelet 
    with specified nondimensional frequency
    
    Params
    ------
    mother: String
        ['morlet'] | 'paul' | 'dog'
    k0: Scalar
        Non-dimensional frequency of the mother wavelet
        If k0 = -1 (default), then uses 
        k0 = 6 for 'morlet'
        k0 = 4 for 'paul'
        k0 = 2 for 'dog' (i.e. Difference of Gaussians)
    
    Returns
    -------
    fourier_factor: Scalar
        The fourier_factor, which allows conversion from wavelet scale to period
        as follows:
        period = fourier_factor * scale
    """
    import sys
    import numpy as np
    if mother.lower() == 'morlet':
        if k0== -1:
            k0 = 6
        fourier_factor = (4*np.pi)/(k0 + np.sqrt(2 + k0**2)) # --> see Sec. 3h  
    elif mother.lower() == 'paul':
        if k0 == -1:
            k0 = 4
        m = k0
        fourier_factor = 4*np.pi/(2*m+1)
    elif mother.lower() == 'dog':
        if k0 == -1:
            k0 = 2
        m = k0
        fourier_factor = 2*np.pi*np.sqrt(2/(2*m+1))
    else:
        print('Mother wavelet must be either "morlet", "paul", or "dog"')
        sys.exit()
    return fourier_factor

def normalizeByScale(W,scale):
    """
    Returns the matrix of wavelet coefficients after normalization by scale so
    that the original timeseries can be reconstructed again from the wavelet
    in the most accurate possible way.
    Parameters
    ----------
    W: array, (J,N)
        Array of complex wavelet coefficients, where J is the # of frequency 
        scales and N is the number of time points
    scale: array, (J,)
        Scales returned by wavelet.wt or wavlet.wavelet
    Returns
    -------
    W_norm: array, (J,N)
        W after normalizing by scales
    """
    import numpy as np
    S  = np.sqrt(np.tile(scale.reshape((-1,1)),(1,np.shape(W)[1])))
    #S  = np.tile(scale.reshape((-1,1)),(1,np.shape(W)[1]))
    W_norm = W.copy()
    return W_norm/S    

def plotWave(W,freq,time,coi = [],powScale = 'log',cmap = 'coolwarm', xlabel = 'Time (sec)', ylabel = 'Freq (Hz)'):
    '''
    fh = plotWave(W,freq,time,...)
    Plots the matrix of wavelet coefficients W, using the specified freq and time axes
    '''
    import numpy as np
    import apCode.SignalProcessingTools as spt
    import matplotlib.pyplot as plt
    if powScale.lower() == 'log':
        W = np.log2(np.abs(W))
    else:
        W = np.abs(W)
    period = 1/freq
    dt = time[1]-time[0]
    tt = np.hstack((time[[0,0]]-dt*0.5,time,time[[-1,-1]]+dt*0.5))
    if len(coi) == 0:
        coi = time
    coi_ext = np.log2(np.hstack((freq[[-1,1]],1/coi,freq[[1,-1]])))
    tt = spt.nearestMatchingInds(tt,time)
    coi_ext = spt.nearestMatchingInds(coi_ext,freq)
    freq_log = np.unique(spt.nextPow2(freq))
    fTick = 2**freq_log
    yTick  = 1/fTick
    inds = spt.nearestMatchingInds(yTick,period)
    ytl = (1/period[inds]).astype(int).ravel()
    fig =plt.imshow(W, aspect = 'auto',cmap = cmap)
    fig.axes.set_yticks(inds)
    fig.axes.set_yticklabels(np.array(ytl))
    plt.colorbar()
    
    xTick = np.linspace(0,len(time)-1,5).astype(int)
    xtl = np.round(time[xTick]/(time[-1]/4))*0.5
    fig.axes.set_xticks(xTick)    
    fig.axes.set_xticklabels(xtl.ravel())   
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.show()
    plt.plot(tt,coi_ext,'k--')
    return fig

def ridgify(W, *args, **kwargs):
    """ 
    Applies "frangi" ridge filter from skimage.filters to the array of wavelet
    coefficients. This is useful for extracting instantaneous frequency, phase information.
    Parameters
    ----------
    W: array, (nScales, nTimePoints)
        2D array of wavelet coefficients returned by wavelet.wt or wavelet.xwt
    *arg, **kwargs: Arguments and keyword arguments for skimage.filters.frangi
    Returns
    -------
    RW: array, shape(W)
        R*W, where R is ridge filter. Values are complex.
    R: array, shape(W)
        Ridge filter. Values are real.
    """
    from skimage.filters import frangi
    from numpy import abs
    kwargs['black_ridges'] = kwargs.get('black_ridges', False)
    R = frangi(abs(W), *args, **kwargs)
    return R*W, R

class ts():
    """
    A set of functions for
    """
    def freq(W,freq_):
        """
        Returns the various types of instantaneous frequencies from a 
        wavelet transform
        
        Parameters
        ----------
        W: (M, N) array
            Matrix of complex/absolute wavelet coefficients
        freq_: (M,) array-like
            Vector of frequencies corresponding to each of the rows of W.
        
        Returns
        -------
        f_mean: (N,) array 
            Mean instantaneous frequency (weighted by wavelet power)
        f_mode: (N,) array
            Instantaneous frequency at max power
        f_pks_mean: (N,) array
            Instantaneous frequency computed from weighted average of frequency
            at power peaks at each time point
        """
        import numpy as np
        from apCode.SignalProcessingTools import findPeaks
        
        t = np.arange(np.shape(W)[1])
        [T,F]= np.meshgrid(t,freq_)        
        W = np.abs(W)
        P = W/np.sum(W,axis = 0)
        f_mean = np.sum(F*P,axis = 0)
        f_mode = freq_[np.argmax(W,axis = 0)]
        
        f_pks_mean = np.zeros(np.shape(W)[1],)
        for tInd,p in enumerate(P.T):
            pks = findPeaks(p, pol = 1)[0]
            if np.size(pks)>0:
                wts  = p[pks]/np.sum(p[pks])
                f_pks_mean[tInd] = np.dot(freq_[pks],wts)       
        return f_mean,f_mode,f_pks_mean       

    def phase(W):
        """
        Returns the various types of instantaneous phases from a 
        wavelet transform
        
        Parameters
        ----------
        W: (M, N) array
            Matrix of complex/absolute wavelet coefficients   
            
        Returns
        -------
        ph_mean: (N,) array 
            Mean instantaneous phase (weighted by wavelet power)
        ph_mode: (N,) array
            Instantaneous phase at max power
        ph_pks_mean: (N,) array
            Instantaneous phase computed from weighted average of phase
            at power peaks at each time point
        """
        import numpy as np
        from apCode.SignalProcessingTools import findPeaks
        
        Phi = np.angle(W)
        W = np.abs(W)
        P = W/np.sum(W,axis = 0)
        ph_mean = np.sum(Phi*P,axis = 0)
        ph_mode = Phi[np.argmax(W,axis = 0)[0],:]
        
        ph_pks_mean = np.zeros(np.shape(W)[1],)
        for tInd,p in enumerate(P.T):
            pks = findPeaks(p, pol = 1)[0]
            if np.size(pks)>0:
                wts  = p[pks]/np.sum(p[pks])
                ph_pks_mean[tInd] = np.dot(Phi[pks,tInd],wts)       
        return ph_mean,ph_mode,ph_pks_mean   

def wave_bases(k, scale, mother  = 'morlet', param = -1):
    import numpy as np
    import scipy as sp
    '''
    1D wavelet function, Morlet, Paul, or DOG
    [daughter, fourier_factor,coi,dofmin] = wave_bases(k,scale,mother = wavelet, param = -1)
    
    Parameters
    ----------
    k: array, (k,)
        A vector of Fourier frequencies at which to calculate the wavelet
    mother: string
        Wavelet family; 'morlet', 'paul', 'dog' (type of wavelet to use)
    scale: scalar
        The wavelet scale
    param: scalar
        The nondimensional parameter for the wavelet function;
        param = -1, ==> wave number of 6 for Morlet, 4 for Paul, and 2 for DOG
    
    Returns:
    daughter: array, (k,)
        The wavelet function
    fourier_factor: scalar
        The ratio of Fourier period to scale
    coi: scalar
        The cone-of-influence size at the scale
    dofmin: scalar
        The degrees of freedom for each point in the wavelet power
        (2 for Morlet and Paul, 1 for the DOG)
    #--------------------------------------------------------------------------
    Adapted from Christopher Torrence and Gilbert P. Compo, University of
    Colorado, Program in Atmospheric and Oceanic Sciences [Copyright (C) 1995-1998].
    #--------------------------------------------------------------------------
    '''
    mother = mother.upper()
    n = len(k)
    
    if not isinstance(mother,str):
        print('Mother must be a string. Defaulting to "morlet"')
        mother = 'morlet'
        
    if mother.lower() == 'morlet':
        if param == -1:
            param = 6
        k0 = param
        expnt = -(scale*k - k0)**2/2*(k>0)
        norm = np.sqrt(scale*k[1])*(np.pi**-0.25)*np.sqrt(n) # total energy = N (Eqn 7)
        daughter = norm*np.exp(expnt)
        daughter = daughter*(k>0)   # Heaviside step function
        fourier_factor = (4*np.pi)/(k0 + np.sqrt(2 + k0**2))   # Scale --> Fourier (sec. 3h)
        coi = fourier_factor/np.sqrt(2)
        dofmin = 2
    elif mother.lower() == 'paul':
        if param == -1:
            param = 4
        m = param
        expnt = -(scale*k)*(k>0)
        prodArg= np.arange(2,2*m)
        norm = np.sqrt(scale*k[1]) * (2**m/np.sqrt(m*np.prod(prodArg))) * np.sqrt(n)
        daughter = norm*((scale*k)**m) * np.exp(expnt)
        daughter = daughter*(k>0)
        fourier_factor = 4*np.pi/(2*m+1)
        coi = fourier_factor * np.sqrt(2)
        dofmin = 2
    elif mother.lower() == 'dog':
        if param == -1:
            param = 2
        m = param
        expnt = -(scale*k)**2/2
        norm = np.sqrt(scale*k[1]/sp.special.gamma(m+0.5)) * np.sqrt(n)
        daughter = -norm*(1j**m) *((scale*k)**m) * np.exp(expnt)
        fourier_factor = 2*np.pi*np.sqrt(2/(2*m+1))
        coi = fourier_factor/np.sqrt(2)
        dofmin = 1
    else:
        print('Mother must be either MORLET, PAUL, or DOG (case-insensitive)!')
    
    return daughter, fourier_factor, coi, dofmin

def wavelet(y,t,dj = 1/32, mother = 'morlet',pad = 1, param = -1, freqScale = 'log',
            freqRange = None, **kwargs):
    '''
    Computes the wavelet transform of a timeseries
    [W,period,scale,coi] = wavelet(...)
    Parameters
    ----------
    y: 1D array
        Timeseries to obtain wavelet transform for
    t: Scalar or 1D array
        If scalar,then sampling interval (dt), else time sequence
    pad: Boolean
        Zero padding; pad = 1 results in padding of the signal with zeros till
        the next largest length that is a power of 2.
    dj: Scalar
        Wavelet scale resolution; defaults to 1/24
    mother: String
        Mother wavelet function (Morlet (default), Paul, or DOG) 
    param: Scalar
        Wave number; defaults to 6 for Morlet wavelet
    freqSCale: String, ['log'] | 'lin'
        Base-2 logarithmic or linear frequency scale.
    freqRange - 2 tuple or list, or None:
            Determines the frequency range over which to compute the wavelets.
            If None, then computes for the full possible range, determined by
            Nyquist limit the time length of the timeseries signal.
    **kwargs
        s0 - Smallest scale;  defaults to 2*dt, where dt is sampling interval
        J1 - Starting wavelet scale
    Outputs:
    W - Matrix of wavelet coefficients
    period - Vector of Fourier periods over which WT is computed
    scale  - Vector of wavelet scales used for computation
    coi  - Cone of influence; a vector of values that show where edge effects become significant
    '''
    import numpy as np
    import apCode.SignalProcessingTools as spt
    
    n1 = len(y)
    if np.size(t)==1:
        dt = t
        t = np.arange(n1)*dt
    else:
        dt = t[1]-t[0]    
    
   # Smallest scale
    s0 = kwargs.get('s0', 2*dt) 
    
    J1 = kwargs.get('J1',int(np.fix(np.log2(n1*dt/s0)/dj)))     
   
    
    #...demean and and zero pad timeseries if specified
    y = y - np.mean(y)
    if pad ==1:
        y = spt.zeroPadToNextPowOf2(y)
    
    #...construct wavenumber array used in transform (eqn 5)
    N = len(y)
    k = (np.arange(N/2) + 1) *((2*np.pi)/(N*dt))
    k_neg = -k[np.int(np.fix((N-1)/2)):0:-1]
    k = np.hstack((0,k,k_neg))
    
    #... compute fft of the padded timeseries (eqn 3)
    f = np.fft.fft(y)
    
    #...construct SCALE array & empty PERIOD and WAVE arrays
    fourier_factor = get_fourier_factor(mother = mother, k0 = param)
    
    if freqScale.lower()== 'log':
        scale = s0*2**(np.arange(0,J1+1)*dj)        
        if freqRange != None:
            freq = 1/(scale*fourier_factor)
            minF = np.min(freqRange)
            maxF = np.max(freqRange)
            keepInds = np.where((freq>=minF) & (freq <= maxF))[0]
            freq = freq[keepInds]
            scale = 1./(freq*fourier_factor)
    else:
        maxF = 1/s0
        minF = 1/(n1*dt)
        freq = np.arange(minF,maxF,dj)
        if freqRange != None:
            minF = np.min(freqRange)
            maxF = np.max(freqRange)
            keepInds = np.where((freq>=minF) & (freq <= maxF))[0]
            freq = freq[keepInds]
        scale = 1/(freq*fourier_factor)        
    
    period = scale
    wave = np.zeros((len(scale),N)) # instantiate the wavelet array
    wave = wave + 1j*wave # Make it complex
        
    for sNum, s in enumerate(scale):
        daughter, fourier_factor,coi, dofmin = wave_bases(k,s, mother = mother, param = param)
        wave[sNum,:] = np.fft.ifft(f * daughter)        
        
    period = fourier_factor*scale
    vec1 = np.arange(1,(n1+1)/2-1)
    vec2 = np.arange((n1/2-1),0,-1)
    vec = np.hstack((1e-5,vec1,vec2,1e-5))
    coi = coi*dt*vec
    wave = wave[:,0:n1]    
    return wave, period, scale, coi

def wave_signif(y,dt,scale,sigtest = 0,lag1 =0,siglvl = 0.95,dof = -1, mother = 'morlet',param = -1):
    """
    Significance testing for the 1D wavelet transform wavelet
    
    signif,fft_theor = wave_signif(y,dt,scale,sigtest,lag1,siglvl,dof,mother,param)
    
    Parameters
    ----------
    Y: 1D array or scalar
        If 1D array, then the timeseries, else the variance of the timseries.
    dt: Scalar
        Sampling interval
    scale: 1D array
        Scale indices from previous call to wavelet
    Optional parameters
    -------------------
    sigtest: Scalar, 0, 1, or 2
        If 0, then just do regular chi-squared test, i.e. eqn 18 from Torrence & Compo (1998)
        If 1, then do "time-average" test, i.e. eqn 23. In this case, dof should be set to NA,
            the number of local wavelet spectra that were averaged together. For the Global
            wavelet Spectrum, NA = N, where N is the # of points in the timeseries
        If 2, then do a "scale-average" test, i.e., eqns 25-28. In this case, the DOF should
            be set to a 2 element vector/tuple (s1,s2), which gives the scale range that was
            averaged together. For instance, if one averaged scales between 2 and 8, then
            DOF = (2,8).
    lag1: Scalar
        Lag-1 autocorrelation, used for signif levels. Default = 0.
    siglvl: Scalar
        Significance level to use. Default = 0.95
    dof: Scalar
        Degrees of freedom for signif test.
        If sigtest = 0, then automatically dof = 2 (or 1 for mother = 'dog')
        If sigtest = 1, then dof = NA, the number of times averaged together
        If sigtest = 2, then dof = (s1,s2), the range of scales averaged
        
        NB: If sigtest = 1, then dof can be a vector (same length as "scale"),
            in which case, NA is assumed to vary with scale. This allows one to
            average different numbers of times together at different scales, or
            to take into account things like the Cone of Influence (coi). See
            discussion following eqn 23 in Torrence & Compo (1998)
    
    
    ----------------------------------------------------------------------------
    Copyright (C) 1995-1998, Christopher Torrence and Gilbert P. Compo
    University of Colorado, Program in Atmospheric and Oceanic Sciences.
    This software may be used, copied, or redistributed as long as it is not
    sold and this copyright notice is reproduced on each copy made.  This
    routine is provided as is without any express or implied warranties
    whatsoever.
    ----------------------------------------------------------------------------
    
    """
    import numpy as np
    import sys
    
    if np.ndim(y)==0:
        #n1= 1
        variance = y
    else:
        #n1 = len(y)
        variance = np.std(y)**2
    J1 = len(scale)-1
    
    dj = np.log2(scale[1]/scale[0])
    
    # Get the appropriate parameters (see table 2 from Torrence & Compo (1998))
    if mother.lower() == 'morlet':
        if param == -1:
            param = 6
        k0 = param
        fourier_factor = (4*np.pi)/(k0 + np.sqrt(2 + k0**2)) # Scale to fourier  [Sec. 3h]
        empir = [2., -1, -1, -1]
        if k0 ==6:
            empir[1:] = [0.776, 2.32,0.60]
    elif mother.lower() == 'paul':
        if param ==-1:
            param = 4
        m = param
        fourier_factor = (4*np.pi)/(2*m + 1)
        empir = [2., -1, -1, -1]
        if m ==4:
            empir[1:] = [1.132, 1.17, 1.5]
    elif mother.lower() == 'dog':
        if param ==-1:
            param = 2
        m = param
        fourier_factor = 2*np.pi*np.sqrt(2./(2*m + 1))
        empir = [1., -1, -1, -1]
        if m==2:
            empir[1:] = [3.541, 1.43, 1.4]
        if m ==6:
            empir[1:] = [1.966, 1.37, 0.97]
    else:
        print('"mother" must be either "morlet", "paul", or "dog"')
    
    period = scale*fourier_factor
    dofmin = empir[0]    # Degrees of freedom with no smoothing
    Cdelta = empir[1]    # Reconstruction factor
    gamma_fac = empir[2] # Time-decorrelation factor
    dj0 = empir[-1]      # Scale-decorrelation factor
    freq = dt/period    # Normalized frequency
    fft_theor = (1-lag1**2)/(1-2*lag1*np.cos(freq*2*np.pi) + lag1**2) # [Eqn 16]
    fft_theor = variance*fft_theor
    signif = fft_theor
    if dof == -1:
        dof = dofmin
    
    if sigtest ==0: # No smoothing, DOF = dofmin [Sec.4]
        dof= dofmin
        chiSquare = chisquare_inv(siglvl,dof)/dof
        signif = fft_theor*chiSquare  # Eqn 18
    elif sigtest ==1: # Time-averaged significance
        if np.size(dof)==1:
            dof = np.zeros((1,J1+1)) + dof
        truncate = np.where(dof <1)[0]
        dof[truncate] = np.ones(np.shape(truncate))
        dof = dofmin*np.sqrt(1 + (dof*dt/gamma_fac/scale)**2) # Eqn 23
        truncate = np.where(dof < dofmin)[0]
        dof[truncate] = np.ones(np.shape(truncate)) # Minimum dof is domin
        for a1 in np.arange(1,J1+2):
            chiSquare = chisquare_inv(siglvl,dof[a1])/dof[a1]
            signif[a1] = fft_theor[a1]*chiSquare
    elif sigtest ==2: # Time-averaged siginificance
        if np.size(dof) !=2:
            print('DOF must be set to [S1,S2], the range of scale-averages')
            sys.exit()
        if Cdelta == -1:
            print('Cdelta & dj0 not defined for {0} with param = {1}'.format(mother,param))
            sys.exit()
        s1 = dof[0]
        s2 = dof[1]
        avg  = np.where((scale >= s1) & (scale <=s2))[0] # Scales between s1 and s2
        navg = len(avg)
        if navg ==0:
            print('No valid scales between {0} and {1}'.format(s1,s2))
            sys.exit()
        Savg = 1/sum(1/scale[avg])  # Eqn 25
        Smid = np.exp((np.log(s1) + np.log(s2))/2)  # Power of 2 midpoint
        dof = (dofmin*navg*Savg/Smid)*np.sqrt(1 + (navg*dj/dj0)**2)  # Eqn 28
        fft_theor = Savg*np.sum(fft_theor[avg]/scale[avg])
        chiSquare = chisquare_inv(siglvl,dof)/dof
        signif = (dj*dt/Cdelta/Savg)*fft_theor*chiSquare  # Eqn 26
    else:
        print('"sigtest" must be either 0, 1, or 2')
        sys.exit()
    return signif, fft_theor    
        
    
def wt(y,t,dj = 1/32, mother = 'morlet',pad = 1, param = -1, freqScale = 'log',ar1 = 'auto', **kwargs):
    """
    Continuous Wavelet Transform (CWT)
    
    """    
    import numpy as np
    
    # Default values
    maxScale = None
    
    n = len(y)
    sigma2 = np.var(y)
    
    if np.ndim(t)==0:
        dt = t
        t = np.arange(n)*dt
    else:
        dt = t[1]-t[0]
        
    for key in kwargs.keys():
        if key.lower() == 'maxscale':
            maxScale = kwargs[key]         

    s0= kwargs.get('s0')
    if s0 == None:
        s0 = 2*dt
        
    siglvl = kwargs.get('siglvl')
    if siglvl == None:
        siglvl = 0.95
    
    J1 = kwargs.get('J1')    
    if J1 == None:
        if maxScale == None:
            maxScale = (n*0.17)*2*dt
        J1 = np.round(np.log2(maxScale/s0)/dj)
    if ar1.lower()== 'auto':
        ar1 = ar1nv(y)[0]
        
    wave,period,scale,coi = wavelet(y,t,dj = dj, mother = mother ,param = param,
                                    freqScale = freqScale, **kwargs)
    power = np.abs(wave)**2
    signif = wave_signif(1.0,dt,scale,sigtest=0,lag1=ar1,siglvl= siglvl,mother = mother)
    sig95 = signif[0].reshape((-1,1))*np.ones((1,n))
    sig95 = power/(sigma2*sig95)
    
    return wave, period, scale, coi, sig95
    

def xwt(x,y,t,pad = 1, dj = 1/32,mother = 'morlet',param = -1, \
        noiseType = 'white',AR1 = 'auto',freqScale = 'log', freqRange = None, 
        arrowDensity = [30, 30],arrowSize = 1, 
        arrowHeadSize = 1, **kwargs):
    '''
    Computes the crosswavelet transform for two timeseries
    Wxy, period, scale, coi, sig95 = xwt(x,y,t,**kwargs)
    Parameters
    ----------
    x, y: 1D arrays
        The timeseries to compute xwt for.
    t: 1D array
        Time vector
    pad: Scalar
        If pad = 1, then zero pads series to length that is next higher power of 2. 
        This speeds up the FFT. If pad =0, then does not pad
    dj: Scalar
        Frequency resolution.
    
    Returns
    -------
    Wxy, period, scale, coi, sig95
    
    '''    
    import numpy as np
    
    if np.size(t)==1:
        dt = t
        t = np.arange(0,len(x))*dt
    else:
        dt = t[1]-t[0]
    Wx, period, scale, coix = wavelet(x,t,pad = pad, dj = dj, mother = mother,
                                      param = param, freqScale= freqScale, 
                                      freqRange = freqRange)
    Wy, period, scale, coiy = wavelet(y,t,pad = pad, dj = dj, mother = mother, 
                                      param = param, freqScale = freqScale, 
                                      freqRange = freqRange)
    sigmaxy = np.std(x)*np.std(y)
    Wxy = Wx*np.conj(Wy)
    
    #...Arrow parameters
    ad = np.mean(arrowDensity)
    arrowSize = arrowSize*30*0.03/ad
    arrowHeadSize = arrowHeadSize*arrowSize*220
    
    if AR1.lower() == 'auto':
        AR1 = [ar1nv(x)[0], ar1nv(y)[0]]       
        if np.any(np.isnan(AR1)):
            print('Error: Automatic AR1 estimation failed')            
    coi = np.min([coix,coiy], axis = 0)
    
    if noiseType.lower() == 'red':
        Pkx = ar1spectrum(np.array(AR1[0]),period/dt)
        Pky = ar1spectrum(np.array(AR1[1]),period/dt)
    elif noiseType.lower() == 'white':
        Pkx = ar1spectrum(0,period/dt)
        Pky = ar1spectrum(0,period/dt)
    
    V = 2
    Zv = 3.9999
    signif = sigmaxy*np.sqrt(Pkx*Pky)*Zv/V  # Eqn(5), Grinsted et al., 2004
    sig95 = np.asmatrix(signif).T*np.asmatrix(np.ones((1,len(t))))
    sig95 =  np.abs(Wxy)/np.array(sig95)    
    if mother.lower() != 'morlet':
        sig95 = sig95*np.nan   
    
    return Wxy, period, scale, coi, sig95   
    
    
    
    
    