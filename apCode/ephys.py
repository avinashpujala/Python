"""Process electrophysiological recordings of fish behavior and trial structure"""

import numpy as np 
def chopTrials(signal,trialThr=2000):
    """for each unique value in the signal, 
       return the start and stop of each epoch corresponding to that value
    """
    
    allCond = np.unique(signal)        
    chopped = {}
    for c in allCond:
        tmp = np.where(signal == c)[0]
        offs = np.where(np.diff(tmp) > 1)[0]
        offs = np.concatenate((offs, [tmp.size-1]))
        ons = np.concatenate(([0], offs[0:-1] + 1))
        trLens = offs - ons        
        keepTrials = np.where(trLens > trialThr)
        offs = offs[keepTrials]
        ons = ons[keepTrials]
        chopped[c] = (tmp[ons], tmp[offs])
    
    return chopped

def deArtifact(y, stimInds, nPre = 6, nPost = 30, interpKind= 'slinear',
               axis = 1):
    '''
    Remove stimulus artifacts from an array of signals using interpolation
    
    Parameters
    ----------
    y - Array-like; Signals to be de-artifacted
    stimInds - Indices where stimuli occurred
    nPre - Number of points before stimulus from where to begin removing 
        the artifacts
    nPost - Number of points after the stimulus until where to remove 
        the artifacts
    interpKind - Kind of interpolation to use. Same as for the 'kind' 
        parameter of scipy.interpolate.interp1d
    
    Returns
    -------
    y_artless - Signal array with artifacts removed
    
    '''
    import scipy as sp
    import numpy as np
    nPre,nPost = int(nPre),int(nPost)
    x = np.arange(np.shape(y)[axis])
    if len(x) != np.shape(y)[axis]:
        print('Dimension mismatch: Check axis specification!')
    x_sub,y_sub = x.copy(),y.copy()
    artInds = []
    for stimInd in stimInds:
        artOnInd = np.max((stimInd-nPre,0))
        artOffInd = np.min((stimInd+nPost,np.shape(y)[axis]))
        artInds.append(np.arange(artOnInd, artOffInd))
    artInds = np.array(artInds).ravel()
    y_sub = np.delete(y,artInds,axis = axis)
    x_sub = np.delete(x,artInds)
    f = sp.interpolate.interp1d(x_sub,y_sub,kind = interpKind)
    y_new = f(x)
    return y_new

# filter signal, extract power
def smoothPower(ch,kern):
    smch = np.convolve(ch, kern, 'same')
    power = (ch - smch)**2
    fltch = np.convolve(power, kern, 'same')
    return fltch

# get peaks
def getPeaks(fltch,deadTime=80):
    
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
def getThreshold(fltch,wind=180000,shiftScale=1.6):
    
    th = np.zeros(fltch.shape)
    
    for t in np.arange(0,fltch.size-wind, wind):

        interval = np.arange(t, t+wind)
        sqrFltch = fltch ** .5            
        hist, bins = np.histogram(sqrFltch[interval], 1000)
        mx = np.min(np.where(hist == np.max(hist)))
        mn = np.max(np.where(hist[0:mx] < hist[mx]/200.0))        
        th[t:] = (bins[mx] + shiftScale * (bins[mx] - bins[mn]))**2.0
    return th

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

def importCh(filename=[],nCh = 10, Fs = 6000):
    """ Imports *.nch (e.g., .10ch) file and parses it into matlab-structure-like arrays: data['t'][:], etc.
    """
    from tkinter import filedialog
    from tkinter import Tk
    
    fileSuffix = '{}ch'.format(nCh)
    
    if not filename:        
        root = Tk()
        root.filename =  filedialog.askopenfilename(initialdir = "/",
                                                    title = "Select file", 
                                                    filetypes = ((fileSuffix,'*.{}Flt'.format(fileSuffix)),
                                                                 ("all files","*.*")))
        print (root.filename)
        filename = root.filename
        root.destroy()    
    
    f = open(filename, 'rb')
    A =  np.fromfile(f, np.float32)
    N  = nCh*np.floor(len(A)/nCh).astype(int)# In case the file is incompletely recorded    
    A = A[:N].reshape((-1,nCh)).T 
    f.close() 
    
    data  = {}
    data['filename'] = filename
    data['t'] = np.arange(np.shape(A)[1])*(1/Fs)    
    kws = ['ch1','ch2','camTrig','x0','stim0','stim1','epoch','stimID','vel',
           'gain','patch1','patch2','patch3','patch4','ch3','ch4']
    
    N = np.min((nCh,len(kws)))
    for ch in range(N):
        data[kws[ch]]= A[ch,:]
            
    for count in range(ch+1,np.shape(A)[0]):
        data['x{}'.format(count)] = A[count,:]      
    return data

def load(inFile):
    """Load 10chFlt data from disk, return as a [channels,samples] sized numpy array
    """
    fd = open(inFile, 'rb')
    data = np.fromfile(file=fd, dtype=np.float32)
    r = np.remainder(data.size,10)
    missingPts = 10-r
    zeroPad = np.zeros((missingPts))
    data = np.concatenate((data,zeroPad), axis = 0)
    data = data.reshape(data.size/10,10)
    return data
    
def stimInds(bas,thr = 1, stimCh:str = 'stim0', minStimDist:int =5*6000, 
             normalize:bool = True):
    """
    bas: dic
        BehaveAndScan file read by "importCh".
    thr: scalar or str
        Threshold for detecting stimuli. If 'auto', then estimates a threshold (assuming)
        that stimulus polarity is positive (i.e. large positive values are stimuli)
    stimChannels: list of strings
        List of the names of stimulus channels.
    minStimDist: int
        Minimum distance (# of samples) between successive stimuli. Set 0
    normalize:bool
        Whether to normalize the stimulus channel before detecting stimuli.
        If True, converts stim channel to z-score units. Useful option when 
        absolute threshold value is unknown.
    
    """
    import numpy as np
    import apCode.SignalProcessingTools as spt
    from apCode.volTools import getGlobalThr
    
    x = bas[stimCh].copy()
    if normalize:
        x = spt.zscore(x)

    if isinstance(thr, str):
        if thr.lower() == 'auto':
            x_pos = x[np.where(x>=0)]
            thr = getGlobalThr(x_pos)
    pks = spt.findPeaks(x,thr = thr, pol =1, minPkDist = minStimDist)
    if len(pks)>0:
        return pks[0]
    else:
        print('No stimuli found!')
        return None   

def stackInits(frameCh,thrMag=3.8,thrDur=10):    
    """
        Find indices in ephys time corresponding to the onset of each stack in image time
    """
    
    stackInits = np.where(frameCh > thrMag)[0]
    initDiffs = np.where(np.diff(stackInits) > 1)[0]
    initDiffs = np.concatenate(([0], initDiffs+1))    
    stackInits = stackInits[initDiffs]
    keepers = np.concatenate((np.where(np.diff(stackInits) > thrDur)[0], [stackInits.size-1]))
    stackInits = stackInits[keepers]    
    return stackInits
    