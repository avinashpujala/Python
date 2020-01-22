# -*- coding: utf-8 -*-
"""
Created on Fri May  8 00:11:29 2015
Modified version of 'importEphys.py' from nvladimus
@author: pujalaa
"""

import numpy as np
def import10ch(filename):
    """ Imports *.10ch file and parses it into matlab-structure-like arrays: data['t'][:], etc.
    """
    f = open(filename, 'rb')
    A =  np.fromfile(f, np.float32).reshape((-1,10)).T
    f.close()
    data = np.zeros( np.size(A, axis = 1), dtype = [
                ('t',np.float32), 
                ('ch0',np.float32), 
                ('ch1',np.float32),
                ('fltCh0',np.float32),
                ('fltCh1',np.float32),
                ('camTrigger',np.float32),
				('2pTrigger',np.float32),
                ('stim1',np.float32),
                ('speed',np.float32),
                ('stim2',np.float32),
                ('temp',np.float32)])
    ker = np.exp(-np.r_[-60:61:1] **2/(2*20**2.)) # Gaussian kernel
    ch1 = A[0,:]
    smch1 = np.convolve(ch1,ker/ker.sum(),mode='same')
    pow1 = (ch1 - smch1)**2
    ch2 = A[1,:]
    smch2 = np.convolve(ch2,ker/ker.sum(),mode='same')
    pow2 = (ch2 - smch2)**2    
    data['t'][:] = np.linspace(1,np.size(A[0,:]),np.size(A[0,:]))/6000.
    data['ch0'][:] = ch1
    data['ch1'][:] = ch2
    data['fltCh0'][:] = np.convolve(pow1,ker/ker.sum(),mode='same')
    data['fltCh1'][:] = np.convolve(pow2,ker/ker.sum(),mode='same')    
    data['stim2'][:] = A[4,:]
    data['stim1'][:] = A[5,:]
    data['speed'][:] = A[6,:]
    data['camTrigger'][:] = A[2,:]
    data['2pTrigger'][:] = A[8,:]
    data['temp'][:] = A[9,:]
    return data

def stackInits(data,thrMag=3.8):    
    """ finds stack onset indices in ephys data
    """
    stackInits = np.where(data['camTrigger'][:] > thrMag)[0]
    initDiffs = np.where(np.diff(stackInits) > 1)[0]
    initDiffs = np.concatenate(([0], initDiffs + 1))    
    stackInits = stackInits[initDiffs]
    return stackInits

def getSwims(fltch, th = 2.5):

    peaksT,peaksIndT = getPeaks(fltch)
    thr = getThreshold(fltch,peaksT,90000, th)
    burstIndT = peaksIndT[np.where(fltch[peaksIndT] > thr[peaksIndT])]
    burstT = np.zeros(fltch.shape)
    burstT[burstIndT] = 1
    
    interSwims = np.diff(burstIndT)
    swimEndIndB = np.where(interSwims > 800)[0]
    swimEndIndB = np.concatenate((swimEndIndB,[burstIndT.size-1]))

    swimStartIndB = swimEndIndB[0:-1] + 1
    swimStartIndB = np.concatenate(([0], swimStartIndB))
    nonShort = np.where(swimEndIndB != swimStartIndB)[0]
    swimStartIndB = swimStartIndB[nonShort]
    swimEndIndB = swimEndIndB[nonShort]
  
    bursts = np.zeros(fltch.size)
    starts = np.zeros(fltch.size)
    stops = np.zeros(fltch.size)
    bursts[burstIndT] = 1
    starts[burstIndT[swimStartIndB]] = 1;
    stops[burstIndT[swimEndIndB]] = 1;
    
    return starts, stops, thr

# filter signal, extract power
def smoothPower(ch,kern):
    smch = np.convolve(ch, kern, 'same')
    power = (ch - smch)**2
    fltch = np.convolve(power, kern, 'same')
    return fltch

# get peaks
def getPeaks(fltch,deadTime=40):
    
    aa = np.diff(fltch)
    peaks = (aa[0:-1] > 0) * (aa[1:] < 0)
    inds = np.where(peaks)[0]    

    # take the difference between consecutive indices
    dInds = np.diff(inds)
                    
    # find differences greater than deadtime
    toKeep = (dInds > deadTime)    
    
    # only keep the indices corresponding to differences greater than deadT 
    inds[1::] = inds[1::] * toKeep
    inds = inds[inds.nonzero()]
    
    peaks = np.zeros(fltch.size)
    peaks[inds] = 1
    
    return peaks,inds

# find threshold
def getThreshold(fltch,peaks,wind=90000,shiftScale=2.5):
    ''' edited by nvladimus '''
    th = np.zeros(fltch.shape)
    
    for t in np.arange(0,fltch.size-wind, wind):

        interval = np.arange(0, t+wind)
        peaksInd = np.where(peaks[interval])        
        xH = np.arange(0,0.05,1e-4)
        # make histogram of peak values
        peakHist = np.histogram(fltch[peaksInd], xH)[0]
        # find modal value
        mx = np.min(np.where(peakHist == np.max(peakHist)))        
        # find distance between mode and foot of left side of histogram
        if peakHist[0] < peakHist[mx]/100.0:
            mn = np.max(np.where(peakHist[0:mx] < peakHist[mx]/100.0))  
        else:
            mn = 0
        th[t:t+wind] = xH[mx] + shiftScale * (xH[mx] - xH[mn])
    th[t+wind:] = th[t+wind-1]
    return th