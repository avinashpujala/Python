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

import sys
sys.path.insert(0,'C:/Users/pujalaa/Documents/Code/Python/code/codeFromNV')
sys.path.insert(0,'C:/Users/pujalaa/Documents/Code/Python/code/util')
sys.path.insert(0,'C:/Users/pujalaa/Documents/Code/Python/code/wavelets')

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
    import SignalProcessingTools as spt
    fInds = spt.valsToNearestInds(freqRange,freq)
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


def plotWave(W,freq,time,coi = [],powScale = 'log',cmap = 'coolwarm', xlabel = 'Time (sec)', ylabel = 'Freq (Hz)'):
    '''
    fh = plotWave(W,freq,time,...)
    Plots the matrix of wavelet coefficients W, using the specified freq and time axes
    '''
    import numpy as np
    import SignalProcessingTools as spt
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
    tt = spt.valsToNearestInds(tt,time)
    coi_ext = spt.valsToNearestInds(coi_ext,freq)
    freq_log = np.unique(spt.nextPow2(freq))
    fTick = 2**freq_log
    yTick  = 1/fTick
    inds = spt.valsToNearestInds(yTick,period)
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



def wt(y,t,pad = 1,dj = 1/24, mother = 'morlet', param = -1, **kwargs):
    '''
    Computes the wavelet transform of a timeseries
    [W,period,scale,coi] = wt(y,t,pad = 1, dj = 1/24, mother = 'morlet',param = -1, **kwargs)
    Inputs:
    y - Timeseries
    t  - Time vector of the same length as the timeseries
    pad - Zero padding; pad = 1 results in padding
    dj - Wavelet scale resolution; defaults to 1/24
    mother - Mother wavelet function (Morlet (default), Paul, or DOG) 
    param  - Wave number; defaults to 6 for Morlet wavelet
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
    import SignalProcessingTools as spt
       
    dt = t[1]-t[0]
    n1 = len(y)
   # k0 = param
    s0 = kwargs.get('s0')
    if s0 is None:
        s0 = 2*dt
    J1 = kwargs.get('J1')
    if J1 is None:
        J1 = np.fix(np.log2(n1*dt/s0)/dj)
    
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
    scale = s0*2**(np.arange(0,J1+1)*dj)    
    period = scale
    waveArray = np.zeros((J1+1,N)) # instantiate the wavelet array
    waveArray = waveArray + 1j*waveArray
    
    for a1 in np.arange(J1+1):
        [daughter, fourier_factor,coi,dofmin] = waveBases(k,scale[a1])
        waveArray[a1,:] = np.fft.ifft(f*daughter)
    
    period = fourier_factor*scale
    vec1 = np.arange(1,(n1+1)/2-1)
    vec2 = np.arange((n1/2-1),0,-1)
    vec = np.hstack((1e-5,vec1,vec2,1e-5))
    coi = coi*dt*vec
    waveArray = waveArray[:,0:n1]
    
    return waveArray, period,scale,coi
    
def waveBases(k, scale, mother  = 'Morlet', param = -1):
    import numpy as np
    import scipy as sp
    '''
    1D wavelet function, Morlet, Paul, or DOG
    [daughter, fourier_factor,coi,dofmin] = wave_bases(k,scale,mother = wavelet, param = -1)
    
    Inputs:
    mother  - A string equal to 'morlet', 'paul', 'dog' (type of wavelet to use)
    k = A vector, the Fourier frequencies at which to calculate the wavelet
    scale = A number, the wavelet scale
    param = The nondimensional parameter for the wavelet function;
        param = -1, ==> wave number of 6 for Morlet, 4 for Paul, and 2 for DOG
    
    Outputs:
    daughter = A vector, the wavelet function
    fourier_factor = The ration of Fourier period to scale
    coi = A number, the cone-of-influence size a the scale
    dofmin = A number, the degrees of freedom for each point in the wavelet power
        (2 for Morlet and Paul, 1 for the DOG)
    #--------------------------------------------------------------------------
    Adapted from Christopher Torrence and Gilbert P. Compo, University of
    Colorado, Program in Atmospheric and Oceanic Sciences [Copyright (C) 1995-1998].
    #--------------------------------------------------------------------------
    '''
    mother = mother.upper()
    n = len(k)
    
    if mother == 'morlet'.upper():
        if param == -1:
            waveNum = 6
        k0 = waveNum
        expnt = -(scale*k - k0)**2/2*(k>0)
        norm = np.sqrt(scale*k[1])*(np.pi**-0.25)*np.sqrt(n) # total energy = N (Eqn 7)
        daughter = norm*np.exp(expnt)
        daughter = daughter*(k>0)   # Heaviside step function
        fourier_factor = (4*np.pi)/(k0 + np.sqrt(2 + k0**2))   # Scale --> Fourier (sec. 3h)
        coi = fourier_factor/np.sqrt(2)
        dofmin = 2
    elif mother == 'paul'.upper():
        if param == -1:
            waveNum = 4
        m = waveNum
        expnt = -(scale*k)*(k>0)
        prodArg= np.arange(2,2*m)
        norm = np.sqrt(scale*k[1]) * (2**m/np.sqrt(m*np.prod(prodArg))) * np.sqrt(n)
        daughter = norm*((scale*k)**m) * np.exp(expnt)
        daughter = daughter*(k>0)
        fourier_factor = 4*np.pi/(2*m+1)
        coi = fourier_factor * np.sqrt(2)
        dofmin = 2
    elif mother == 'dog'.upper():
        if param == -1:
            waveNum = 2
        m = waveNum
        expnt = -(scale*k)**2/2
        norm = np.sqrt(scale*k[1]/sp.special.gamma(m+0.5)) * np.sqrt(n)
        daughter = -norm*(1j**m) *((scale*k)**m) * np.exp(expnt)
        fourier_factor = 2*np.pi*np.sqrt(2/(2*m+1))
        coi = fourier_factor/np.sqrt(2)
        dofmin = 1
    else:
        print('Mother must be either MORLET, PAUL, or DOG')
    
    return daughter, fourier_factor, coi, dofmin

def xwt(x,y,t,pad = 1, dj = 1/24,mother = 'morlet',param = -1, \
        noiseType = 'white',AR1 = 'auto',arrowDensity = [30, 30],\
        arrowSize = 1, arrowHeadSize = 1):
    '''
    Computes the crosswavelet transform for two timeseries
    Wxy, period, scale, coi, sig95 = xwt(x,y,t,**kwargs)
    Inputs:
    x, y - The 2 timeseries
    t - Time vec
    
    '''
    
    import numpy as np
        
    def ar1nv(x):
       '''
       Estimate the parameters for an AR(1) model
       g,a = ar1nv(x)
       Inputs
       x - A timeseries
       Outputs:
       g - Estimate of lag-1 autocorrelation
       a - Estimate of the noise variance
       '''
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
        freq = 1/period
        P = (1-ar1**2)/(np.abs(1-ar1*np.exp(-2*np.pi*1j*freq)))**2
        return P
    dt = t[1]-t[0]
    Wx, period, scale, coix = wt(x,t,pad = pad, dj = dj, \
                                        mother = mother,param = param)
    Wy, period, scale, coiy = wt(y,t,pad = pad, dj = dj, \
                                        mother = mother,param = param)
    sigmaxy = np.std(x)*np.std(y)
    Wxy = (Wx*np.conj(Wy))/sigmaxy
    
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
        Pky = Pkx
    
    V = 2
    Zv = 3.9999
    signif = sigmaxy*np.sqrt(Pkx*Pky)*Zv/V  # Eqn(5), Grinsted et al., 2004
    sig95 = np.asmatrix(signif).T*np.asmatrix(np.ones((1,len(t))))
    sig95 =  np.abs(Wxy)/np.array(sig95)    
    if mother.lower() != 'morlet':
        sig95 = sig95*np.nan   
    
    return Wxy, period, scale, coi, sig95   
    
    
    
    
    