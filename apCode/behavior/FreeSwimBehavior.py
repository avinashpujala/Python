# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:50:40 2015

@author: pujalaa
"""
import numpy as np
import sys
import os
import dask
import h5py
from dask.diagnostics import ProgressBar
sys.path.append(r'v:/code/python/code')


def alignSignalsByOnset_old(signals, startSignalInd=0, nPre=30,
                            padType='edge'):
    """
    Given a list of signals, returns a matrix of aligned signals
    Parameters
    ----------
    signals: list
        List of signals
    startSignalInd: integer
        Index of signal to use as the starting reference signal
    nPre: integer
        Number of pre-onset points to keep in each signal
    padType: string
        'zero'|'edge'
        If 'zero', then zero pads signals, else edge pads.

    """
    if not isinstance(signals, list):
        signals = list(signals)
    sigLens = np.array([len(s) for s in signals])
    sig_mean = np.zeros((sigLens.max(),))
    s = signals[startSignalInd]
    sig_mean[:len(s)] = s
    S = np.zeros((sigLens.max(),len(signals)))
    padLens,shifts,signs = [np.zeros((len(signals),)) for _ in range(3)]
    padInds = []
    indVec = np.arange(sigLens.max())
    for count, s in enumerate(signals):
        foo = matchByOnset(sig_mean,s, nPre = 0, padType = padType)
        padLens[count] = foo['padLens'][1]
        shifts[count] = foo['shifts'][1]+nPre
        signs[count] = foo['signs'][1]
        indVec_now = np.roll(indVec,int(shifts[count]))
        padInds.append(indVec_now[sigLens.max()-int(padLens[count]):])
        S[:len(foo['signals'][:,1]),count] = np.roll(foo['signals'][:,1],nPre)
        sig_mean = np.mean(S[:,:count+1],axis = 1)
    mu = np.mean(S,axis = 1)
    c = np.array([np.corrcoef(mu,s)[0,1] for s in S.T])
    out = {'signals': np.array(S), 'padLens': padLens.astype(int),'shifts': shifts.astype(int),
           'signs': signs,'correlations': c,'nPre': nPre,'padType': padType,
           'padInds':padInds}
    def transform(out,signals, padType = None):
        if padType == None:
            padType = out['padType']
        signals_new = []
        for count, s in enumerate(signals):
            if padType == 'edge':
                foo = np.pad(s,pad_width = (0,out['padLens'][count]),
                             mode = 'edge')
            else:
                foo = np.pad(s,pad_width = (0,out['padLens'][count]),
                             mode = 'constant', constant_values = (0,0))
            foo = out['signs'][count]*np.roll(foo,out['shifts'][count])
            signals_new.append(foo)
        return np.array(signals_new).T
    out['transform'] = transform
    return out

def alignSignalsByOnset(signals, startSignalInd = 0, padType = 'edge'):
    """
    Given a list of signals, returns a matrix of aligned signals
    Parameters
    ----------
    signals: list
        List of signals
    startSignalInd: integer
        Index of signal to use as the starting reference signal
    nPre: integer
        Number of pre-onset points to keep in each signal
    padType: string
        'zero'|'edge'
        If 'zero', then zero pads signals, else edge pads.

    """
    if not isinstance(signals, list):
        signals = list(signals)
    sigLens = np.array([len(s) for s in signals])
    maxLen = sigLens.max()
    s = signals[startSignalInd]
    if padType == 'edge':
        sig_mean = np.pad(s,pad_width = (0,maxLen-len(s)),mode = 'edge')
    else:
        sig_mean = np.pad(s, pad_width = (0,maxLen-len(s)), mode = 'constant',
                          constant_values = (0,0))

    S = np.zeros((maxLen,len(signals)))
    padLens,signs = [],[]
    for count, s in enumerate(signals):
        foo = matchByOnset(sig_mean,s,padType = padType)
        pl= foo['padLens']
        foo['signals'] = np.delete(foo['signals'],np.arange(pl[0,0]),axis = 0)
        #lenDiff = maxLen-np.shape(foo['signals'])[0]
        #pl = foo['padLens'][1]
        #pl[1] = pl[1]+lenDiff
        foo['signals'] = foo['signals'][:maxLen,:]
        padLens.append(pl)
        signs.append(foo['signs'][1])
        S[:len(foo['signals'][:,1]),count] = foo['signals'][:,1]
        sig_mean = np.mean(S[:,:count+1],axis = 1)
    mu = np.mean(S,axis = 1)

    c = np.array([np.corrcoef(mu,s)[0,1] for s in S.T])
    out = {'signals': np.array(S), 'padLens': padLens,'signs': signs,
           'correlations': c,'padType': padType}
    def transform(out,signals, padType = None):
        if padType == None:
            padType = out['padType']
        #signals_new = []
        signals_new = np.zeros_like(out['signals'])
        maxLen = np.shape(signals_new)[0]
        for count, s in enumerate(signals):
            pw = out['padLens'][count]
            if padType == 'edge':
                foo = np.delete(s,np.arange(pw[0,0]))
                foo = np.pad(foo, pad_width = pw[1], mode = 'edge')
            else:
                foo = np.delete(s,np.arange(pw[0,0]))
                foo = np.pad(foo,pad_width = pw[1], mode = 'constant',
                             constant_values = (0,0))
            foo = out['signs'][count]*foo
            foo = np.delete(foo,np.arange(maxLen,len(foo)))
            signals_new[:len(foo),count]= foo
        return signals_new

    def padToNan(out, signals_aligned):
        """
        Sets padded values in aligned signals to NaNs for plotting
        Parameters
        ----------
        out: Output from alignSignalsByOnset
        signals_aligned: array, (T,N)
            Aligned signals, where T is # of time points, and N is # of signals
        Returns
        -------
        signals_new: array (T,N)
            Signals where padded values are set to NaN
        """
        lenTime = np.shape(signals_aligned)[0]
        signals_new = []
        for count, sig in enumerate(signals_aligned.T):
            ind = lenTime-int(out['padLens'][count][1,1])
            blah = sig.copy()
            blah[ind:] = np.nan
            signals_new.append(blah)
        signals_new = np.array(signals_new).T
        return signals_new

    out['transform'] = transform
    out['padToNan'] = padToNan

    return out


def analyzeDistribution(x, comps = 5, plotBool = True, xLabel = '', distType ='kde',
                        choose_comps = True):
    """
    When given some 1D data, returns the histogram,the components
    of the Gaussian Mixture Model fit to the data, as well as the handle to the
    figure with the data plotted in case the figure is to be saved later
    Parameters
    ----------
    x - Array of size (N,) or (N,1); 1D on which to run the analysis
    comps - Scalar, or array of size (n_components,); If scalar, then checks AIC
        and BIC for components upto this number, starting from 1. If array, then
        computes the above for the number of components in the array. Subsequently,
        picks the number of components on the minimum for AIC and BIC. If AIC and BIC
        give different results then chooses the smaller number of components for either
        of the two
    choose_comps: Boolean
        If True, then automatically chooses the optimal number of components, else uses
        specified number of components
    plotBool - Boolean; If True, plots the figure, and returns the figure handle
    xLabel: string
        Label for the x-axis
    distType: string
        'kde' | 'hist'; If kde then estimates and plots kernel density, else a regular
        histogram with automatic binning

    Returns
    -------
    gmm - GM model object with relevant info appended. For e.g., the components are
        gmm.components_
        The hist and bins are in gmm.hist_
        The AIC and BIC are in gmm.ic_
    """
    import matplotlib.pyplot as plt
    import apCode.machineLearning.ml as ml
    from sklearn.mixture import GaussianMixture as GMM
    xRange = (np.min(x),np.max(x))
    e2c = lambda x: 0.5*(x[:-1]+x[1:])

    if np.ndim(comps)==0:
        comps = np.arange(comps)+1

    if np.ndim(x)==1:
        x = x.reshape((-1,1))

    ic = ml.gmm.informationVersusNumComp(x,comps=comps);
    if choose_comps:
        n_components = comps[np.min((np.argmin(ic['aic']),np.argmin(ic['bic'])))]
    else:
        n_components = np.max(comps)

    gmm = GMM(n_components=n_components).fit(x)
    gmm.ic_ = ic;
    gmm.comps_ = ml.gmm.components(gmm,x)
    gmm.labels_ = gmm.predict(x)
    N = gmm.n_components-1

    if plotBool:
        fh = plt.figure(figsize=(16,18))
        #---Histogram
        plt.subplot(2,1,1)
        if distType == 'hist':
            p,bins = np.histogram(x,bins = 'auto',density = True, range = xRange)
            w = np.mean(np.diff(bins))
            plt.bar(e2c(bins),p,alpha = 0.3, width =w, label = 'Histogram',color = 'gray')
            plt.xlabel(xLabel)
            plt.ylabel('Prob')
        else:
            from scipy import stats
            kde = stats.gaussian_kde(x.ravel(),bw_method = 'scott')
            bins = np.linspace(x.min(),x.max(),int(np.shape(x)[0]/40))
            p = kde(bins)
            w = np.mean(np.diff(bins))
            #plt.bar(bins,p,alpha = 0.3, width = w, label = 'KDE', color = 'gray')
            plt.fill_betweenx(p, bins, alpha = 0.3, color = 'gray')
            plt.xlabel(xLabel)
            plt.ylabel('Prob density')
        gmm.hist_ = {'p':p, 'bins': bins}

        #---GMM
        #t = np.linspace(x.min(),x.max(),np.shape(x)[0])
        t = np.linspace(bins.min(),bins.max(),np.shape(x)[0])
        gmm.x_ = t
        sortInds = np.argsort(gmm.means_[:,0].ravel())
        mus = gmm.means_[:,0].ravel()
        sigmas = np.sqrt(gmm.covariances_[:,0].ravel())
        for indNum,ind in enumerate(sortInds):
            c = gmm.comps_[:,ind]
            mu = np.round(mus[ind]*10)/10
            sigma = np.round(sigmas[ind]*10)/10
            txt = r'$x_{0}$: {1} $\pm$ {2}'.format(ind, mu, sigma)
            plt.plot(t,c,label = txt, alpha = 0.9)
        #plt.plot(t,gmm.comps_)
        plt.plot(t,np.sum(gmm.comps_,axis = 1),'k--', label = r'$\Sigma_{{k = 0}}^{sup} x_k$'.format(sup = {N}))
        plt.title('Distribution')
        plt.legend();

        #---Information criteria
        plt.subplot(2,1,2)
        #plt.show()
        plt.plot(comps,ic['aic'],'o-',label = 'Akaike')
        plt.plot(comps,ic['bic'],'o-',label = 'Bayesian')
        plt.xticks(comps)
        plt.xlabel('# of components')
        plt.ylabel('Information criterion')
        plt.title('AIC & BIC vs number of components')
        plt.legend()
        return gmm, fh
    else:
        return gmm

def appendTrials(trialDir,nTrials = 5):
    '''
    outDirs = appendTrials(trialDir, nTrials = n) - Creates n-trial appended directories
        and returns names of appended-trial directories
        (default: nTrials = 5)
    '''
    import os, time
    import numpy as np
    import shutil as sh
    startTime = time.time()
    imgFldrs = os.listdir(trialDir)
    imgFldrs_sorted = np.sort(imgFldrs)
    #imgDir_new = src.split(sep = 'Trial')[0]
    imgDir_new = trialDir + '/'
    trlList = []
    subList = []
    for trl in range(len(imgFldrs_sorted)):
        if np.mod(trl,nTrials)== nTrials-1:
            subList.append(trl)
            trlList.append(subList)
            subList = []
        else:
            subList.append(trl)
    if len(subList)>0:
        trlList.append(subList)
    print('Copying images...')
    outDirs = []
    for trlNums in trlList:
        print('Copied trials ', trlNums)
        fldrName = 'trials_' + str(trlNums[0]) + '-'+ str(trlNums[-1])
        dst = imgDir_new + fldrName
        if not os.path.exists(dst):
            os.mkdir(dst)
        else:
            sh.rmtree(dst)
            os.mkdir(dst)
        dst = os.path.join(imgDir_new,fldrName)
        ctr = -1
        if len(trlNums) > 0:
            for fldr in imgFldrs_sorted[trlNums]:
                for imgFile in np.sort(os.listdir(os.path.join(imgDir_new,fldr))):
                    if imgFile.endswith('.jpg'):
                        ctr +=1
                        src = os.path.join(os.path.join(imgDir_new,fldr),imgFile)
                        sh.copy2(src,dst)
                        fileNum = '%.6d' % ctr
                        newFileName = fldrName + '_' + fileNum + '.jpg'
                        os.rename(os.path.join(dst,imgFile),os.path.join(dst,newFileName))
            dst.replace('\\','/')
            outDirs.append(dst)
    return(outDirs)
    print(int(time.time()-startTime),'sec')

def bendInfoFromTotalCurvature(x, thr = (2,0.5), n_ker = 250, fps = 500):
    """
    Given a single trial timeseries of the total body curvature of fish returns
        dictiionary with bend parameters.
    Parameters
    ----------
    x: array, (N,)
        Total curvature timeseries with N points
    thr: tuple, (2,)
        min(thr), max(thr) indicate the lower and upper limits of the nonstationary
        threshold used to detect peaks using convolution
    n_ker: scalar
        Kernel width in points; the signal is convolved with a hemi-gaussian to simulate
        a causal convolution
    fps: scalar
        Frames per second
    Returns
    -------
    d: dict
        Dict with keys and values indicating bend parameters and their correspondign values.
        If no more than 1 peak found, then return empty dictionary
    """
    import apCode.SignalProcessingTools as spt
    thr = np.array(thr)
    n_ker = int(n_ker)
    x_std = spt.zscore(x)
    pks = spt.findPeaks(x_std,pol = 0)[0]
    ### Compute a nonstationary threshold based on convolution of acivity with causal kernel
    x_conv = spt.causalConvWithSemiGauss1d(np.abs(x), n_ker)
    x_thr = spt.standardize(x_conv.max()-x_conv)*(thr.max()-thr.min()) + thr.min()

    ### Delete peaks below threshold
    d = {}
    if len(pks)>1:
        pks = np.delete(pks, np.where(np.abs(x_std[pks])<x_thr[pks]))

        ### Delete peaks with small relative difference in amplitude
        dAmps = np.abs(np.diff(x_std[pks]))
        pks = np.delete(pks,np.where(dAmps<(2*thr.min()))[0]+1)
        amps = np.abs(np.diff(x[pks]))
        amps = np.insert(amps,0,np.abs(x[pks][0]))
        d['bendIdx'] = pks
        d['bendAmp_rel'] = amps
        d['bendAmp_abs'] = x[pks]
        d['bendInt'] = np.insert(np.nan,0,1000*np.diff(pks)/fps)
        d['bendNum'] = np.arange(len(pks))+1
        d['episodeDur'] = 1000*(pks[-1]-pks[0])/fps
        return d
    else:
        print('Only 1 peak found, skipping')
        d = {}
        return d


def centerImagesOnFish(I,fishPos):
    if len(np.shape(I))==3:
        img = I[0]
    elif len(np.shape(I))==2:
        img = I
    else:
        print('Image input must be 2 or 3 dimensional!')

    origin = np.round(np.array(np.shape(img))/2).astype(int)
    I_roll = list(map(lambda x, y: np.roll(x,origin[0]-y[0],axis = 0), I,fishPos))
    I_roll = list(map(lambda x, y: np.roll(x,origin[1]-y[1],axis = 1), I_roll,fishPos))
    return I_roll

def copy_cropped_images_for_training(imgsOrPath, cropSize=(120, 120),
                                     savePath=None, nImgsToCopy=50,
                                     detect_motion_frames=True, **motion_kwargs):
    """
    Return probability images generated using the specified U net.
    Parameters
    ----------
    imgsOrPath: array, (nImgs, *imgDims)
        Images to train U net on
    cropSize: 2-tuple, int or None
        Size to crop images to around the fish. A fish detection algorithm
        is used to detect fish that involves background subtraction, therefore,
        all the images must be from the same experiment and must contain a
        single fish at most
        If None, then does not crop images
    savePath: None or str
        Path to directory where images are to be saved
    detect_motion_frames: bool
        If True, then tries to detect frames capturing fish in motion
    motion_kwargs: dict
        Keyword arguments for estimate_motion
    Returns
    -------
    imgs_crop: array, (nImgsToCopy, *cropSize)
        Cropped images
    """
    from apCode.machineLearning import ml as mlearn
    import apCode.volTools as volt
    import apCode.FileTools as ft
    from apCode import util
    daskArr = False
    if isinstance(imgsOrPath, str):
        imgs = volt.dask_array_from_image_sequence(imgsOrPath)
        daskArr = True
    else:
        imgs = imgsOrPath

    # Compute background
    n = np.minimum(1000, imgs.shape[0])
    inds = np.linspace(0, imgs.shape[0]-1, n).astype(int)
    imgs_back = imgs[inds]
    print('Computing background...')
    if daskArr:
        imgs_back = imgs_back.compute()
    else:
        imgs_back = imgs_back

    bgd = imgs_back.mean(axis=0)
    if detect_motion_frames:
        print('Detecting motion frames...')
        motion = estimate_motion(imgs, **motion_kwargs)
        thr = np.percentile(motion, 90)
        motion_frames = np.where(motion >= thr)[0]
        imgs = imgs[motion_frames]
    nImgsToCopy = np.minimum(nImgsToCopy, imgs.shape[0])
    inds = np.random.permutation(imgs.shape[0])[:nImgsToCopy]
    imgs = imgs[inds]

    if isinstance(imgs, dask.array.core.Array):
        imgs = imgs.compute()

    if cropSize is None:
        imgs_crop = imgs
    else:
        imgs_back = bgd-imgs
        print('Tracking fish...')
        fp = np.array([track.findFish(_) for _ in imgs_back])
        print(f'Cropping images to size {cropSize}')
        imgs_crop = volt.img.cropImgsAroundPoints(imgs, fp, cropSize=cropSize)

    if savePath is not None:
        imgDims = imgs.shape[-2:]
        name_dir = f'images_train_{imgDims[0]}x{imgDims[1]}_{util.timestamp()}'
        path_save = os.path.join(savePath, name_dir)
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        volt.img.saveImages(imgs_crop, imgDir=path_save)
        print(f'Saved images at\n {path_save}')
    return imgs_crop


def cropImagesAroundArena(I):
    '''
    Given an image stack, returns images cropped around the fish arena
    '''
    from scipy import signal
    import apCode.SignalProcessingTools as spt
    import apCode.volTools as volt
    I_mu = np.mean(I,axis= 0)
    #I_grad = np.abs(np.gradient(I_mu))
    #I_grad = (I_grad[0] + I_grad[1])/2
    I_grad = volt.img.findHighContrastPixels(I_mu)[1]
    imgLen = np.shape(I_mu)[0]
    imgWid = np.shape(I_mu)[1]
    ker = signal.gaussian(20,20/6)
    rLims = np.round([imgLen/2 - 0.1*imgLen, imgLen/2 + 0.1*imgLen]).astype(int)
    cLims = np.round([imgWid/2 - 0.1*imgWid, imgWid/2 + 0.1*imgWid]).astype(int)
    I_sub_r = I_grad[rLims[0]:rLims[1], :]
    I_sub_c = I_grad[:,cLims[0]:cLims[1]]
    xProf = np.mean(I_sub_r,axis = 0)
    xProf[[0,-1]] = 0
    xProf = spt.zscore(np.abs(np.diff(np.convolve(xProf,ker,mode = 'same'))))
    yProf = np.mean(I_sub_c,axis =1)
    yProf[[0,-1]] = 0
    yProf = spt.zscore(np.abs(np.diff(np.convolve(yProf,ker,mode = 'same'))))
    xInds = spt.findPeaks(xProf,thr = 1)
    xInds_start = np.delete(xInds,np.where(xInds > 0.3*imgWid))
    if len(xInds_start)==0:
        xInds_start = np.min(xInds)
    else:
        xInds_start = np.min(xInds_start)
    xInds_end = np.delete(xInds, np.where(xInds < 0.6*imgWid))
    np.shape(xInds_end)
    if len(xInds_end)==0:
        xInds_end = np.max(xInds)
    else:
        xInds_end = np.min(xInds_end)
    xInds = np.hstack((xInds_start,xInds_end))

    yInds = spt.findPeaks(yProf,thr = 1)
    yInds_start = np.delete(yInds, np.where(yInds > 0.3*imgLen))
    yInds_end = np.delete(yInds, np.where(yInds < 0.6*imgLen))

    if len(yInds_start)==0:
        yInds_start = np.min(yInds)
    else:
        yInds_start= np.min(yInds_start)
    if len(yInds_end)==0:
        yInds_end = np.max(yInds)
    else:
        yInds_end = np.min(yInds_end)
    yInds= np.hstack((yInds_start,yInds_end))


    x = [np.min(xInds),np.max(xInds)]
    x[0] = np.max([x[0]-10,0])
    x[1] = np.min([x[1]+10,imgWid])
    y = [np.min(yInds),np.max(yInds)]
    y[0] = np.max([y[0]-10,0])
    y[1] = np.min([y[1]+10,imgLen])
    I_crop= I[:,y[0]:y[1],x[0]:x[1]]
    return(I_crop,x,y)


def deleteIncompleteTrlImgs(imgDir,nImagesInTrl = 500*1.5 + 30*60, imgExt = 'bmp'):
    '''
    Given the path to an image directory, and the number of images in each trial, deletes
    images that are not part of a complete trial, perhaps because of unexpected interruptions
    to the image acquisition

    '''
    import apCode.FileTools as ft

    filesInDir= ft.findAndSortFilesInDir(imgDir,ext = imgExt)
    filesInDir = [os.path.join(imgDir,fileInDir) for fileInDir in filesInDir]
    lastFullTrlImg = int(np.floor(len(filesInDir)/nImagesInTrl) * nImagesInTrl)

    print('About to delete', len(filesInDir)-lastFullTrlImg, 'images')
    for fileInDir in filesInDir[lastFullTrlImg:-1]:
        os.remove(fileInDir)
    os.remove(filesInDir[-1])
    print('Done!')

def estimate_motion(imgs, bgd=None, kSize=11, inds_skip=(2, 2, 2)):
    """
    Tries to estimate fish motion from images and returns
    Parameters
    ----------
    imgs: array, (nImgs, *imgDims)
        Images
    bgd: array, (*imgDims)
        Background image to subtract. If None, then computes
    kSize: int
        Kernel size for gaussian blurring of images
    inds_skip: int
        If too many images, then can speed up by skipping this many images
    Returns
    --------
    mot: array, (nImgs, )
        Estimate of motion
    """
    import apCode.volTools as volt
    from scipy.interpolate import interp1d
    inds = np.arange(imgs.shape[0])
    imgs = imgs[::inds_skip[0],::inds_skip[1],::inds_skip[2]]
    if bgd is None:
        n = np.minimum(imgs.shape[0], 1000)
        inds_back = np.linspace(0, imgs.shape[0]-1, n).astype(int)
        inds_back = np.unique(inds_back)
        bgd = imgs[inds_back].mean(axis=0)
    else:
        bgd = bgd[::inds_skip[1], ::inds_skip[2]]
    imgs = imgs-bgd
    dImgs = np.abs(np.diff(imgs, axis=0))
    dImgs = volt.filter_gaussian(dImgs, kSize=kSize, sigma=6)
    m = np.apply_over_axes(np.max, dImgs, [1, 2]).flatten()
    m = np.r_[[0], m, m[-1]]
    x = np.r_[inds[::inds_skip[0]], inds[-1]+1]
    m_interp = interp1d(x, m)(inds)
    return m_interp

def filterFishData(data,dt= 1./1000,Wn=100, btype = 'low', \
       keysToOmit = ['time','axis1','axis2','axis3']):
    '''
    Filters all signals in data except the values for keys in keysToOmit
    '''
    import SignalProcessingTools as spt
    import copy
    data_flt = copy.deepcopy(data)
    keys = list(data_flt[0].keys())
    keys = np.setdiff1d(keys,keysToOmit)
    for fishNum,fishData in enumerate(data_flt):
        for key in keys:
            if key is not 'time':
                trlData = fishData[key]
                try:
                    np.float64(trlData[0][0])
                    data_flt[fishNum][key] = list(map(lambda x:spt.chebFilt(x,dt,Wn,btype = btype.lower()),trlData))
                except:
                    print('Did not filter fish #', fishNum, ', key: ', key)
                    data_flt[fishNum][key] = trlData
    return data_flt


def fish_imgs_from_raw(imgs, unet, bgd=None, prob_thr=0.1, diam=11,
                       sigma_space=1, method='fast', **unet_predict_kwargs):
    """
    Returns prob and fish blob binary images when given raw images and a
    trained U net
    Parameters
    ----------
    imgs: array, (nImgs, *imgDims)
        Raw images
    unet: Keras model object
        Trained U net model
    bgd: array (*imgDims), None  or 'compute'
        Background image
    prob_thr: scalar
        Threshold probability to us to generate binary fish blob images from
        probability images
    diam: int
        Diameter of bilateral filter during processing
    sigma_space: int
        cv2 bilteral filter parameter
    method: str, 'fast' or 'slow'
    unet_predict_kwargs: dict
        Keyword argument for unet.predict function
    Returns
    -------
    imgs_fish, imgs_prob: array, (nImgs, *imgDims)
        Binary fish blob and probability images respectively
    """
    import apCode.volTools as volt
    from skimage.measure import regionprops, label

    def _fish_img_from_raw(img_prob, img_prob2, prob_thr):
        mask = (img_prob2 > prob_thr).astype('uint8')
        lbls = label(mask)
        regions = regionprops(lbls)
        if len(regions)>1:
            perimeters = np.array([region.perimeter for region in regions])
            ind = np.argmax(perimeters)
            region = regions[ind]
        elif len(regions)==1:
            region = regions[0]
        else:
            return img_prob
        cent = np.array(region.centroid)
        regions = regionprops(label((img_prob > prob_thr).astype('uint8')))
        if len(regions)>1:
            cents = np.array([region.centroid for region in  regions])
        elif len(regions)==1:
            cents = np.array(regions[0].centroid)
        else:
            return img_prob
        dists = np.sum((cents.reshape(-1, 2)-cent.reshape(1, 2))**2, axis=1)**0.5
        ind = np.argmin(dists)
        coords = regions[ind].coords
        img_fish = np.zeros_like(img_prob)
        img_fish[coords[:, 0], coords[:, 1]]=1
        return img_fish

    def _fish_img_from_raw_fast(img_prob, img_raw, prob_thr):
        mask = (img_prob > prob_thr).astype('uint8')
        img_lbl = label(mask)
        regions = regionprops(img_lbl, -img_raw*img_prob)
        if len(regions)>1:
            max_ints = np.array([region.max_intensity for region in regions])
            ind = np.argmax(max_ints)
            region = regions[ind]
        elif len(regions)==1:
            region = regions[0]
        else:
            return img_prob
        coords = region.coords
        img_fish = np.zeros_like(img_prob)
        img_fish[coords[:, 0], coords[:, 1]]=1
        return img_fish

    imgs_prob = prob_images_with_unet(imgs, unet, **unet_predict_kwargs)
    imgs_flt = volt.filter_bilateral(imgs_prob, diam=diam,
                                     sigma_space=sigma_space)
    if bgd is not None:
        if bgd is 'compute':
            bgd = track.computeBackground(imgs)
        imgs_back = bgd-imgs
    else:
        imgs_back = imgs
    imgs_fish = []
    if method is 'slow':
        imgs_prob2 = prob_images_with_unet(imgs_back*imgs_prob, unet,
                                           **unet_predict_kwargs)
        imgs_flt2 = volt.filter_bilateral(imgs_prob2, diam=diam,
                                          sigma_space=sigma_space)
        for img_prob, img_prob2 in zip(imgs_flt, imgs_flt2):
            img_fish = dask.delayed(_fish_img_from_raw)(img_prob,
                                                        img_prob2, prob_thr)
            imgs_fish.append(img_fish)
    else:
        for img_prob, img_raw in zip(imgs_flt, imgs):
            img_fish = dask.delayed(_fish_img_from_raw_fast)(img_prob,
                                                             img_raw, prob_thr)
            imgs_fish.append(img_fish)
    with ProgressBar():
        imgs_fish = dask.compute(*imgs_fish)
    imgs_fish = np.array(imgs_fish).astype('bool')
    return imgs_fish, imgs_prob


def flotifyTrlDirs(trialDir, trializeDirs = 'y', timeStampSep = ']_'):
    '''
    flotifyTrlDirs - Given a directory trializes each subdirectory and renames
        images contained within to make compatible with Flote
    flotifyTrlDirs(trialDir, trializeDirs = 'y', timeStampSep)
    Inputs - Directory containing trial/image directories
    trializeDirs = 'y' results in renaming subdirectories as trials
    timeStampSep = Separator character(s) marking the beginning of the timestamp
        on the directory name

    '''
    import time
    import apCode.FileTools as ft
    import apCode.volTools as volt

    def renameTrlDirs(trialDir,timeStampSep = ']_'):
        imgFldrs = ft.subDirsInDir(trialDir)
        imgFldrs_sorted = np.sort(imgFldrs)
        newDirNames = []
        for num, name in enumerate(imgFldrs_sorted):
            if name.find('.') == -1:
                trlNum = '%0.2d' % num
                timeStamp = name.split(timeStampSep)
                if len(timeStamp) > 1:
                    timeStamp = timeStamp[1]
                else:
                    timeStamp = timeStamp[0][-9:-1] + timeStamp[0][-1]
                newName = 'Trial' + '_' + trlNum + '_' + timeStamp
                src = trialDir + '/' + name
                dst = trialDir + '/' + newName
                os.rename(src,dst)
                newDirNames.append(newName)

    def renameImagesInTrlDirs(trialDir):
        imgFldrs= ft.subDirsInDir(trialDir)
        imgFldrs_sorted = np.sort(imgFldrs)
        for fldrNum,fldr in enumerate(imgFldrs_sorted):
            src = trialDir + '/' + fldr
            imgsInFldr = volt.img.getImgsInDir(src)
            imgsInFldr_sorted = np.sort(imgsInFldr)
            for imgNum,imgName in enumerate(imgsInFldr_sorted):
                imgExt= imgName.split('.')[1]
                src2 = src + '/' + imgName
                suffix = '%0.6d' % imgNum
                dst = src + '/' + fldr + '_' + suffix + '.' + imgExt
                os.rename(src2, dst)
            print(fldr)

    ### Renaming trial folders
    print("Renaming trial folders...")
    startTime= time.time()
    if trializeDirs.lower() == 'y':
        renameTrlDirs(trialDir)

    ### Renaming images in trial folders
    print("Renaming images in trial folders...")
    renameImagesInTrlDirs(trialDir)

    print('Flotification complete!')
    print(int(time.time()-startTime), 'sec')

def flotifyTrlDirs_clstr(trialDir, trializeDirs = 'y', timeStampSep = '_'):
    '''
    flotifyTrlDirs - Given a directory trializes each subdirectory and renames
        images contained within to make compatible with Flote
    flotifyTrlDirs(trialDir, trializeDirs = 'y', timeStampSep)
    Inputs - Directory containing trial/image directories
    trializeDirs = 'y' results in renaming subdirectories as trials
    timeStampSep = Separator character(s) marking the beginning of the timestamp
        on the directory name

    '''
    import time
    import apCode.FileTools as ft
    import apCode.volTools as volt

    def renameTrlDirs(trialDir,timeStampSep = ']_'):
        imgFldrs = ft.getsubDirsInDir(trialDir)
        imgFldrs_sorted = np.sort(imgFldrs)
        for num, name in enumerate(imgFldrs_sorted):
            if name.find('.') == -1:
                trlNum = '%0.2d' % num
                timeStamp = name.split(timeStampSep)
                if len(timeStamp) > 1:
                    timeStamp = timeStamp[1]
                else:
                    timeStamp = timeStamp[0][-9:-1] + timeStamp[0][-1]
                newName = 'Trial' + '_' + trlNum + '_' + timeStamp
                src = trialDir + '/' + name
                dst = trialDir + '/' + newName
                os.rename(src,dst)

    def renameImagesInTrlDirs(trialDir):
        def renameImage(imgInfo):
            ''' imgInfo = (imgName,imgNum,fldrName,fldrPath) '''
            imgName = imgInfo[0]
            imgNum = imgInfo[1]
            fldrName= imgInfo[2]
            fldrPath = imgInfo[3]
            imgExt = imgName.split('.')[1]
            imgPath = fldrPath + '/' + imgName
            suffix = '%0.6d' % imgNum
            targetPath = fldrPath + '/' + fldrName + suffix + '.' + imgExt
            os.rename(imgPath,targetPath)

        imgFldrs= ft.getsubDirsInDir(trialDir)
        imgFldrs_sorted = np.sort(imgFldrs)
        for fldrNum,fldr in enumerate(imgFldrs_sorted):
            fldrPath = trialDir + '/' + fldr
            imgsInFldr = volt.getImgsInDir(fldrPath)
            imgsInFldr_sorted = np.sort(imgsInFldr)
            imgNums = range(len(imgsInFldr))
            fldrNameList = [fldr]*len(imgsInFldr)
            fldrPathList = [fldrPath]*len(imgsInFldr)
            imgInfoList = list(zip(imgsInFldr_sorted,imgNums,fldrNameList,fldrPathList))
            sc.parallelize(imgInfoList).foreach(renameImage) # sc should work on spark
            print(fldr)
    ### Renaming trial folders
    print("Renaming trial folders...")
    tic = time.time()
    if trializeDirs.lower() == 'y':
        renameTrlDirs(trialDir)

    ### Renaming images in trial folders
    print("Renaming images in trial folders...")
    renameImagesInTrlDirs(trialDir)

    print('Flotification complete!')
    print(int(time.time()-tic), 'sec')


def flotifyTrlDirs_parallel(trialDir, trializeDirs = 'y', timeStampSep = ']_'):
    '''
    flotifyTrlDirs - Given a directory trializes each subdirectory and renames
        images contained within to make compatible with Flote
    flotifyTrlDirs(trialDir, trializeDirs = 'y', timeStampSep)
    Inputs - Directory containing trial/image directories
    trializeDirs = 'y' results in renaming subdirectories as trials
    timeStampSep = Separator character(s) marking the beginning of the timestamp
        on the directory name

    '''
    import time, multiprocessing
    import apCode.FileTools as ft
    import apCode.volTools as volt
    from joblib import Parallel, delayed

    def renameTrlDirs(trialDir,timeStampSep = ']_'):
        imgFldrs = ft.subDirsInDir(trialDir)
        imgFldrs_sorted = np.sort(imgFldrs)
        newDirNames = []
        for num, name in enumerate(imgFldrs_sorted):
            if name.find('.') == -1:
                trlNum = '%0.2d' % num
                timeStamp = name.split(timeStampSep)
                if len(timeStamp) > 1:
                    timeStamp = timeStamp[1]
                else:
                    timeStamp = timeStamp[0][-9:-1] + timeStamp[0][-1]
                newName = 'Trial' + '_' + trlNum + '_' + timeStamp
                src = trialDir + '/' + name
                dst = trialDir + '/' + newName
                os.rename(src,dst)
                newDirNames.append(newName)

    def renameImagesInTrlDirs(trialDir):
        imgFldrs= ft.subDirsInDir(trialDir)
        imgFldrs_sorted = np.sort(imgFldrs)
        numCores = np.min(multiprocessing.cpu_count(),30)
        for fldrNum,fldr in enumerate(imgFldrs_sorted):
            src = trialDir + '/' + fldr
            imgsInFldr = volt.getImgsInDir(src)
            imgsInFldr_sorted = np.sort(imgsInFldr)
            srcList, dstList = [],[]
            for imgNum,imgName in enumerate(imgsInFldr_sorted):
                imgExt= imgName.split('.')[1]
                src2 = src + '/' + imgName
                suffix = '%0.6d' % imgNum
                dst = src + '/' + fldr + '_' + suffix + '.' + imgExt
                srcList.append(src2)
                dstList.append(dst)
            argList = list(zip(srcList,dstList))
            #print(argList)
            Parallel(n_jobs=numCores,verbose = 3)(delayed(os.rename)(x[0],x[1]) for x in argList)
            #os.rename(src2, dst)
            print(fldr)

    ### Renaming trial folders
    print("Renaming trial folders...")
    startTime= time.time()
    if trializeDirs.lower() == 'y':
        renameTrlDirs(trialDir)

    ### Renaming images in trial folders
    print("Renaming images in trial folders...")
    renameImagesInTrlDirs(trialDir)

    print('Flotification complete!')
    print(int(time.time()-startTime), 'sec')


def getArenaEdge(img, nIter=100, tol=1e-2, filt_sigma=10,
                 q_min=0.05, q_max=0.95, plotBool=False, cmap='gist_earth'):
    """
    When given an image (such as the background subtracted image), returns the
    coordinates of a cicle fit to the edge

    Parameters
    ----------
    img: 2D array
        Reference image containing the arena edge
    nIter: scalar
        Number of iterations to loop through when fitting circle, see
        apCode.cv.feature.fitCircle
    tol: scalar
        Error tolerance; see nIter
    plotBool: boolean
        If true, plots the fit circle atop the reference image

    Returns
    -------
    coords: array, shape (2, N)
        Coordinates of the fit circle, 1st and 2nd rows are the x- and
        y-coordinates respectively
    """
    # import apCode.volTools as volt
    # from scipy.ndimage import gaussian_filter
    from skimage.feature import canny
    from apCode.cv.feature import fitCircle
    import matplotlib.pyplot as plt
    img_edge = canny(img, sigma=filt_sigma, low_threshold=q_min,
                     high_threshold=q_max, use_quantiles=True)
    # img_edge = gaussian_filter(sobel(img), 5)
#    print('Estimating threshold...')
    # thr = volt.getGlobalThr(img_edge)
    # img_edge[img_edge < thr] = 0
    # img_edge[img_edge >= thr] = 1
    coords = np.where(img_edge)
#    print('Fitting circle to arena edge...')
    coords_new = fitCircle(coords)[0]

    if plotBool:
        if isinstance(cmap, str):
            cmap = eval(f'plt.cm.{cmap}')
        plt.imshow(img, cmap=cmap)
        plt.axis('image')
        plt.axis('off')
        plt.scatter(*coords_new[::-1, ::10], s=5, c='r')
        plt.scatter(coords_new[1][0], coords_new[0][0], s=20, c='r',
                    label='Fit circle')
        plt.legend(loc=0)
    return np.array(coords_new)


def get1stBendInfo(data, key='curvature', ampZscoreThr=2, slopeZscoreThr=1.5):
    '''
    get1stBendInfo - For the specified key of input data, gets relevant info
    for 1st bend
    firstBendInfo = getFirstBendInfo(data,field = 'curvature',ampZscoreThr = 5,
        slopeZscoreThr =5)
    Inputs:
    data - 'data' variable containing .trk file info loaded by FreeSwimBehavior
    key - Gets info for the specified key
    ampZcoreThr - Amplitude threshold in zscore
    slopeZscoreThr - Slope threshold in zscsore
    Outputs:
    firstBendInfo - Contains useful info about the first bend in response to
    stim
        firstBendInfo[fishNum][0] = Onnset index
        fistBendInfo[fishNum][1] = First peak amplitude
        firstBendInfo[fishNum][2] = First peak index
        firstBendInfo[fishNum][3] = 1 or -1, indicating sign of peak (in the case of
            fish curvature, this corresponds to left or right turn respectively)
        firstBendInfo[fishNum+1] = List of variable names
    '''
    out = []
    for fishData in data:
        temp =[]
        for sig in fishData[key]:
            onsetInd = getOnsetInd(sig,ampZscoreThr=ampZscoreThr,slopeZscoreThr=slopeZscoreThr)
            pkAmp,pkInd = get1stPkInfo(sig, onsetInd, ampZscoreThr=ampZscoreThr)
            if pkAmp > 0:
                turnId = 1 # Left turn
            else:
                turnId = -1 # Right turn
            temp.append([onsetInd,pkAmp,pkInd, turnId])
        out.append(temp)
    varNames = ['onsetInd','peakAmp','peakInd','turnId']
    out.append(varNames)
    return out


def get1stPkInfo(signal,onsetInd, ampZscoreThr = 5):
    '''
    get1stPkInfo - Given a timeseries signal, finds the amplitude of the 1st response
        after the specified onset index
    pkAmp, pkInd = get1stAmpAfterOnset(signal, onsetInd,ampZscoreThr=5)
    Inputs:
    signal - Timeseries signal
    onsetInd - Index of response onset
    ampZscoreThr- Amplitude threshold in zscore (default = 5)
    '''
    import SignalProcessingTools as spt
    if ~np.isnan(onsetInd):
        sigPks = spt.findPeaks(np.abs(signal[onsetInd:-1]),thr = ampZscoreThr,minPkDist=5)
        if len(sigPks)==0:
            print('No signal pks found, lower amp threshold')
            pkInd, pkAmp = np.nan, np.nan
        else:
            pkInd = sigPks[0] + onsetInd
            pkAmp = signal[pkInd]
    else:
        pkInd,pkAmp = np.nan,np.nan
    return pkAmp,pkInd


def getHeadTailCurvatures(data):
    '''
    Given fish data, modifies it such that head and tail curvature are added as
        key-value pairs for each fish and trial
    '''
    import volTools as volt
    def headTailCurvatures(axis1,axis2,axis3):
        f = lambda th:np.array(volt.pol2cart(1,th*np.pi/180))
        g = lambda v0,v1: np.round(np.angle((v0[0] + v0[1]*1j)*np.conj(v1[0] + v1[1]*1j))*180/np.pi*100)/100
        h = lambda th1,th2,th3: [g(f(th1),f(th2)), g(f(th2),f(th3))]
        angles = np.array([h(th[0],th[1],th[2]) for th in zip(axis1,axis2,axis3)])
        return np.transpose(angles,[1,0])
    for fishNum,fish in enumerate(data):
        data[fishNum]['curv_head'] = []
        data[fishNum]['curv_tail'] = []
        for trlNum in range(len(fish['axis1'])):
            #axes = (fish['axis1'],fish['axis2'],fish['axis3'])
            headAngles,tailAngles =\
                headTailCurvatures(fish['axis1'][trlNum],fish['axis2'][trlNum],fish['axis3'][trlNum])
            data[fishNum]['curv_head'].append(headAngles)
            data[fishNum]['curv_tail'].append(tailAngles)
    return data

def getOnsetInd(signal,ampZscoreThr = 5, slopeZscoreThr =1):
    '''
    getOnsetInd - Given a timeseries signal, finds the onset index of the first response
    onsetInd = getOnsetInd(signal, ampZscoreThr = 5, slopeZscoreThr = 5)
    Inputs:
    signal - Timeseries signal
    ampZscoreThr - Amplitude threshold as zscore
    slopeZscoreThr - Slope threshold as zscore
    '''
    import numpy as np
    import apCode.SignalProcessingTools as spt

    dS = np.diff(signal)
    sigPks = spt.findPeaks(np.abs(spt.zscore(signal)),thr=ampZscoreThr,minPkDist=5)[0]
    zeroFlag = 0
    if len(sigPks)==0:
        print('No signal pks found, lower amp threshold')
        zeroFlag = 1
        sigPks = np.nan
    slopePks_pos = spt.findPeaks(spt.zscore(dS),thr =slopeZscoreThr,minPkDist=5)[0]
    slopePks_neg = spt.findPeaks(spt.zscore(-dS),thr =slopeZscoreThr,minPkDist=5)[0]
    slopePks = np.union1d(slopePks_pos,slopePks_neg)
    if len(slopePks)==0:
        print('No onsets found, lower slope threshold')
        zeroFlag = 1
        slopePks = np.nan

    if zeroFlag == 0:
        latDiff = sigPks[0]-slopePks
        negInds = np.where(latDiff<0)[0]
        latDiff = np.delete(latDiff,negInds)
        if len(latDiff)==0:
            onsetInd = np.nan
        else:
            onsetInd =slopePks[np.where(latDiff==np.min(latDiff))[0]]
    else:
        onsetInd = np.nan
    return onsetInd

def getPxlSize(img,diam = 50,**kwargs):
    """
    Given a reference image where the edge of the roughly circular fish arena is discernible, returns
    the size of the pixel using the specified diameter of the arena edge

    Parameters
    ----------
    img: 2D array
        Reference image
    diam: scalar
        Diameter of circular arena (Pixel length will be returned in the same units used here)
    **kwargs: Keyword arguments for the function getArenaEdge

    Returns
    -------
    pxlSize: scalar
        Size of the image pixel in units of specified diameter
    coords: array, shape = (2,N)
        Coordinates of the circle fit to the arena. 1st and 2nd rows are x- and y- coordinates respectively
    """
    import apCode.volTools as volt
    coords = getArenaEdge(img,**kwargs)
    coords_centered = coords - coords.mean(axis =1).reshape((-1,1))
    rho = volt.cart2pol(coords_centered[0],coords_centered[1])[1]
    diam_fit =  2*rho.mean()
    pxlSize = diam/diam_fit
    return pxlSize, coords

def getSwimInfo(data,ampZscoreThr = 1, slopeZscoreThr = 1.5,frameRate=1000, \
    stimFrame=99, minLatency = 5e-3, maxSwimFreq =100, outputMode = 'dict'):
    '''
    getSwimInfo_dict - When given the data variable output by loadMultiFishTrkFiles,
        returns a variable that contains complete information about each swim
        episode in each trial for each fish
    swimInfo = getSwimInfo(data,ampZscoreThr = 1, slopeZscoreThr = 1.5,\
        frameRate =1000, stimFrame, minLatency = 5e-3, maxSwimFreq = 100)
    Inputs:
    data - 'data' variable containing .trk file info loaded by
        loadMultiFishTrkFiles. 'data' is a list variable with each element
        corresponding to a fish. Each element is dictionary, that must contain
        the following keys for the function to work,'curvature', 'axis1',
        'axis2', 'axis3'
    ampZcoreThr - Amplitude threshold in zscore (for determining peaks in signal)
    slopeZscoreThr - Slope threshold in zscsore  (for determining onset in signal)
    frameRate - Frame/sampling rate of the signal
    stimFrame - Frame/index of the point corresponding to stimulus onset
    minLatency - Minimum latency after the stimulus from when to start looking for peaks
    maxSwimFreq - Max expected swim frequency (for avoiding double peak detection
                    because of possible noise)
    outputMode - 'dict' or 'list', specifying which mode to output data.
        Default is 'dict', wherein swimInfo variable-type hierarchy is
        list:list:dict:dict, where as 'list' mode outputs data as
        list:list:list:list (See Outputs below:)
    Outputs:
    swimInfo - Contains pertinent information about swim episodes in signal.
        swimInfo is a list with the following structure...
        In 'dict' mode (default):
        swimInfo[fishNum][trlNum]['sideKey']['variableKey']
            fishNum - Fish number (same length as input variable 'data')
            trlNum - Trial number (same length as data['someKey'])
            sideKey - 'left' or right corresponding to bend info to the left or
            right respectively
            variableKey - 'pkLat','pkAmp','angVel' corresponding either to peak
            latencies from stim onset, peak amplitudes, angular velocity
            (i.e. peak amp/time from previous peak or valley)
        In 'list' mode:
        swimInfo[fishNum][trlNum][sideNum][variableNum]
            fishNum - Fish number (same length as input variable 'data')
            trlNum - Trial number (same length as data['someKey'])
            sideNum - 0 or 1 corresponding to bend info to the left or right
                respectively
            variableNum - 0, 1, 2 corresponding either to peak latencies from
                stim onset, peak amplitudes, angular velocity (i.e. peak amp/time
                    from previous peak or valley)

    '''
    import apCode.SignalProcessingTools as spt

    minOnsetInd = int(minLatency*frameRate) + stimFrame
    minPkDist = int((1/maxSwimFreq)*frameRate)
    out = []
    for fishNum,fishData in enumerate(data):
        temp =[]
        for trlNum, sig in enumerate(fishData['curvature']):
            sideDict = {}
            sideDict['left'] = {}
            sideDict['right'] = {}
            pks_left, pks_left_relAmp = spt.findPeaks(spt.zscore(sig),\
                thr =ampZscoreThr, minPkDist=minPkDist)

            delInds = np.where(pks_left < minOnsetInd)
            pks_left = np.delete(pks_left,delInds)
            pks_left_relAmp = np.delete(pks_left_relAmp,delInds)
            pks_right,pks_right_relAmps = spt.findPeaks(spt.zscore(-sig),\
                thr =ampZscoreThr,minPkDist=minPkDist)
            delInds = np.where(pks_right < minOnsetInd)
            pks_right = np.delete(pks_right,delInds)
            pks_right_relAmps = np.delete(pks_right_relAmps,delInds)
            pks_left_ms = ((pks_left-stimFrame)*(1000/frameRate)).astype(float)
            pks_right_ms = ((pks_right-stimFrame)*(1000/frameRate)).astype(float)
            pks_left_amp = sig[pks_left]
            pks_right_amp = sig[pks_right]
            pks_both = np.union1d(pks_left,pks_right)
            onsetInd = getOnsetInd(sig,ampZscoreThr = ampZscoreThr,\
                slopeZscoreThr=slopeZscoreThr)
            if np.isnan(onsetInd):
                onsetInd = np.array([])
                print('Re-seeking onset by lowering slope thresh, fish #', fishNum, 'trl #', trlNum)
            multFac = 0.9
            dynThr = slopeZscoreThr
            count = 0
            while (len(onsetInd)==0) & (count <10):
                count = count+1
                print('thr=', dynThr)
                onsetInd= getOnsetInd(sig, ampZscoreThr=ampZscoreThr, \
                    slopeZscoreThr=dynThr)
                dynThr = np.round(dynThr*multFac*100)/100
                if np.isnan(onsetInd):
                    onsetInd= np.array([])
            try:
                pks_both_onset = np.union1d(onsetInd,pks_both)
                w = np.diff(sig[pks_both_onset])/np.diff(pks_both_onset*(1./frameRate))
                w_left = w[np.where(w>0)]
                w_right =w[np.where(w<0)]
                if np.any(w==0):
                    print('Zero value for ang vel found, fish #', fishNum, \
                    'trl #', trlNum )
            except:
                w_left = 0
                w_right = 0
            pkAmp = sig[pks_both[0]]
            if pkAmp > 0:
                turnId = 'left' # Left turn
            else:
                turnId = 'right' # Right turn
            if outputMode.lower() == 'dict':
                sideDict['left'] = {'pkLat': pks_left_ms, \
                    'pkAmp': pks_left_amp,'pkAmp_rel': pks_left_relAmp,\
                    'angVel': w_left}
                sideDict['right'] = {'pkLat': pks_right_ms,
                    'pkAmp': pks_right_amp,'pkAmp_rel': pks_right_relAmps, \
                    'angVel': w_right}
                sideDict['turnId']= turnId
                sideDict['onset'] = 1000*(onsetInd-stimFrame)/frameRate
                temp.append(sideDict)
            elif outputMode.lower() == 'list':
                onsetLat = 1000*(onsetInd-stimFrame)/frameRate
                temp.append([[pks_left_ms, pks_left_amp,pks_left_relAmp,w_left],\
                    [pks_right_ms,pks_right_amp,pks_right_relAmps,w_right],\
                    [turnId],[onsetLat]])
            else:
                print('Please specify correct input for outputMode')
        out.append(temp)
    varNames = [['Left pk lats', 'Left pk amps', 'Left omegas'], \
        ['Right pk lats', 'Right pk amps', 'Right omegas'], \
        ['TurnId: 1 = Left first, -1 = Right First']]
    out.append(varNames)
    return out

def loadFishData(trkFileDir,fileStem = 'singlefish'):
    '''
    Reads multiple, multi-trial .trk files created by Flote and returns data list
        where each element is a dictionary that corresponds to data from a
        different fish. The keys of the dictionary indicate the parameter for
        which the timeseries was extracted and the values are lists where each
        element of the list corresponds to a different trial
        Inputs:
        trkFileFileDir - Directory containing the various .trk files
    '''
    import csv, os
    import numpy as np
    def readMultiTrialTrkFile(filePath):

        '''
        readMultiTrialTrkFile - reads single fish, multitrial .trk file created by Flote and returns in
            a dictionary
        Inputs:
        filePath - full file path for .trk file, i.e. os.path.join (fileDir, fileName)
        '''
        hdrs = []
        data = []
        supData = []
        with open(filePath) as csvFile:
            reader = csv.reader(csvFile,delimiter = '\t')
            for rowNum,row in enumerate(reader):
                try:
                    np.float64(row[0])
                    data.append(row)
                except:
                    supData.append(np.array(data))
                    data = []
                    hdrs.append(row)
            supData.append(np.array(data))
        supData.pop(0)
        return np.transpose(supData,[2, 0, 1]),hdrs

    def listToDict(supData,hdrs):
        dictData = {}
        for hdrNum,hdr in enumerate(hdrs[0]):
            try:
                dictData[hdr] = np.float64(supData[hdrNum])
            except:
                dictData[hdr] = supData[hdrNum]
        return dictData

    fishData = []
    filesInDir = os.listdir(trkFileDir)
    filesInDir = list(np.sort(list(filter(lambda x: x.startswith(fileStem),filesInDir))))
    for file in filesInDir:
        dicData = {}
        print('Reading file', file, '...')
        filePath = os.path.join(trkFileDir,file)
        trlData,hdrs = readMultiTrialTrkFile(filePath)
        dicData = listToDict(trlData,hdrs)
        dicData['fileName']  = filePath
        #print(dicData.keys())
        fishData.append(dicData)
    return fishData

def loadMultiTrialTrkFile(filePath):
    '''
    loadMultiTrialTrkFile - reads single fish, multitrial .trk file created by Flote and returns in
        a dictionary
    data = loadMultiTrialTrkFile(filePath)
    Inputs:
    filePath - full file path for .trk file, i.e. os.path.join (fileDir, fileName)
    '''
    import csv
    import numpy as np
    hdrs = []
    data = []
    supData = []
    with open(filePath) as csvFile:
        reader = csv.reader(csvFile,delimiter = '\t')
        for rowNum,row in enumerate(reader):
            try:
                np.float64(row[0])
                data.append(row)
            except:
                supData.append(np.array(data))
                data = []
                hdrs.append(row)
        supData.append(np.array(data))
    supData.pop(0)

    dicData = {}
    for hdrNum, hdr in enumerate(hdrs[0]):
        for dsNum, dataSet in enumerate(supData):
            if dsNum == 0:
                dicData[hdr] = []
            try:
                dicData[hdr].append(np.float64(dataSet[:,hdrNum]))
            except:
                dicData[hdr].append(dataSet[:,hdrNum])
            dicData['fileName'] = filePath

    print('Data dictionary with the following keys...', \
        '\n', dicData.keys())
    return dicData

def matchByOnset_old(x,y,nPre = 30, padType = 'zero'):
    """
    Given two signals with single swim episodes, aligns them w.r.t to their onsets, and
    flips the signals such that the the slope at the onset is positive. If more than one swim
    in signals then aligns w.r.t first swim onset.
    Parameters
    ----------
    x: (n,) array
        1st signal
    y: (k,) array
        2nd signal.The signals need not be the same length, the function pads appropriately.
    nPre: scalar
        Number of points in the signal to include before the onset
    padType: string
        "zero"|"edge"
        If "zero" then zero-pads signals, else edge-pads.
    Returns
    -------
    out: (np.max(n,k),2) array
        Array with the two length-equalized and aligned signals arranged as columns.
    """
    import numpy as np
    from apCode.FreeSwimBehavior import swimOnAndOffsets
    lenDiff = len(x)-len(y)
    padLens = np.array([0,0])
    if lenDiff<0:
        padLens[0] = -lenDiff
        if padType.lower() == 'zero':
            x = np.pad(x,pad_width=(nPre,-lenDiff), mode = 'constant',
                       constant_values = (0,0))
            y = np.pad(y,pad_width = (nPre,0), mode = 'constant',
                       constant_values = (0,0))
        else:
            x = np.pad(x,pad_width=(nPre,-lenDiff), mode = 'edge')
            y = np.pad(y, pad_width = (nPre,0), mode = 'edge')
    elif lenDiff>0:
        padLens[1] = lenDiff
        if padType.lower() == 'zero':
            y = np.pad(y,pad_width=(nPre,lenDiff), mode = 'constant',
                       constant_values = (0,0))
            x = np.pad(x,pad_width = (nPre,0), mode = 'constant',
                       constant_values = (0,0))
        else:
            y = np.pad(y,pad_width=(nPre, lenDiff), mode = 'edge')
            x = np.pad(x,pad_width = (nPre,0), mode = 'edge')
    on_x, on_y = swimOnAndOffsets(x)[0], swimOnAndOffsets(y)[0]
    if np.size(on_x)>1:
        on_x = on_x[0]
    if np.size(on_y)>1:
        on_y = on_y[0]
    dx, dy = np.gradient(x), np.gradient(y)
    sign_x = sign_y = 1
    if dx[on_x] < 0:
        sign_x = -1
        x = -x
    if dy[on_y] < 0:
        sign_y = -1
        y = -y
    x = np.roll(x,-on_x+nPre)
    y = np.roll(y,-on_y+nPre)
    c = np.corrcoef(x,y)[0,1]

    out = {'signals': np.array((x,y)).T, 'signs':np.array((sign_x, sign_y)),
           'shifts': np.array((-on_x+nPre, -on_y+nPre)).ravel(), 'correlation': c,
           'padType': padType, 'padLens': padLens,'nPre': nPre}

    def transform(out,signals, padType = None):
        """
        Applies the same transform to another set of signals

        Parameters
        ----------
        out: Dictionary
            Output from the master function matchByOnset. Stores important values.
        signals: list
            New signals to apply transform to.
        padType: string
            'zero'| 'edge' | None
            If None, then uses the same padding as used on out.
        Returns
        -------
        signals_new: array
            Transformed signals
        """
        signals_new = []
        for count, s in enumerate(signals):
            if not padType:
                padType = out['padType']
            if padType == 'zero':
                foo = np.pad(s,pad_width = (nPre,out['padLens'][count]), mode = 'constant',
                             constant_values = (0,))
            else:
                foo = np.pad(s,pad_width = (nPre,out['padLens'][count]), mode = padType)

            foo = out['signs'][count]*np.roll(foo,out['shifts'][count])
            signals_new.append(foo)
        return np.array(signals_new)
    out['transform'] = transform
    return out

def matchByOnset(x,y, padType = 'zero'):
    """
    Given two signals with single swim episodes, aligns them w.r.t to their onsets, and
    flips the signals such that the the slope at the onset is positive. If more than one swim
    in signals then aligns w.r.t first swim onset.
    Parameters
    ----------
    x: (n,) array
        1st signal
    y: (k,) array
        2nd signal.The signals need not be the same length, the function pads appropriately.
    nPre: scalar
        Number of points in the signal to include before the onset
    padType: string
        "zero"|"edge"
        If "zero" then zero-pads signals, else edge-pads.
    Returns
    -------
    out: (np.max(n,k),2) array
        Array with the two length-equalized and aligned signals arranged as columns.
    """
    import numpy as np
    from apCode.behavior.FreeSwimBehavior import swimOnAndOffsets
    on_x, on_y = swimOnAndOffsets(x)[0], swimOnAndOffsets(y)[0]
    if np.size(on_x)>1:
        on_x = on_x[0]
    if np.size(on_y)>1:
        on_y = on_y[0]
    dx, dy = np.gradient(x), np.gradient(y)
    sign_x = np.sign(dx[on_x])
    sign_y = np.sign(dy[on_y])

    pre_x = on_y-on_x
    pre_y = -pre_x
    pre_x = int(np.max((0,pre_x)))
    pre_y = int(np.max((0,pre_y)))
    preLens = np.array((pre_x,pre_y)).astype(int)
    len_x = len(x)+pre_x
    len_y = len(y)+pre_y
    post_x = len_y-len_x
    post_y = -post_x
    post_x = np.max((0, post_x))
    post_y = np.max((0,post_y))
    postLens = np.array((post_x,post_y)).astype(int)
    if padType.lower() == 'zero':
        x  = sign_x*np.pad(x,pad_width = (pre_x,post_x), mode = 'constant',
                    constant_values = (0,0))
        y = sign_y*np.pad(y,pad_width = (pre_y, post_y), mode = 'constant',
                   constant_values = (0,0))
    else:
        x  = sign_x*np.pad(x,pad_width = (pre_x,post_x), mode = 'edge')
        y = sign_y*np.pad(y,pad_width = (pre_y, post_y), mode = 'edge')

    c = np.corrcoef(x,y)[0,1]
    padLens = np.array((preLens,postLens)).T

    out = {'signals': np.array((x,y)).T, 'signs': np.array((sign_x, sign_y)),
           'correlation': c, 'padType': padType, 'padLens': padLens}

    def transform(out,signals, padType = None):
        """
        Applies the same transform to another set of signals

        Parameters
        ----------
        out: Dictionary
            Output from the master function matchByOnset. Stores important values.
        signals: list
            New signals to apply transform to.
        padType: string
            'zero'| 'edge' | None
            If None, then uses the same padding as used on out.
        Returns
        -------
        signals_new: array
            Transformed signals
        """
        signals_new = []
        for count, s in enumerate(signals):
            if not padType:
                padType = out['padType']
            if padType == 'zero':
                foo = np.pad(s, pad_width = (out['padLens'][count][0], out['padLens'][count][1]),
                             mode = 'constant', constant_values = (0,0))
            else:
                foo = np.pad(s, pad_width = (out['padLens'][count][0],out['padLens'][count][1]),
                             mode = 'edge')

            foo = out['signs'][count]*foo
            signals_new.append(foo)
        return np.array(signals_new).T
    out['transform'] = transform
    return out

def openMatFile(path, name_str = None, mode = 'r'):
    """
    Uses h5py to open .mat files and returns as variable
    Parameters
    ----------
    path: string
        Directory where .mat file is located. If directory
        has multiple .mat files then allows user to choose
    name_str: string
        Searches for files with names containing this string
    Returns
    -------
    out: h5py file with Groups and Datasets
    """
    import h5py, os
    import apCode.FileTools as ft
    filesInDir = ft.findAndSortFilesInDir(path,ext = 'mat', search_str = name_str)
    if len(filesInDir)>1:
        print(filesInDir)
        ind_file = input('Enter index of file to open: ')
        ind_file = int(ind_file)
        matFile = filesInDir[ind_file]
    elif len(filesInDir)==0:
        print('No files found, check path!')
    else:
        matFile = filesInDir[0]
    try:
        out = h5py.File(os.path.join(path,matFile), mode = 'r+')
    except:
        out = h5py.File(os.path.join(path,matFile), mode= 'r')
#    print('Data has the following keys: {}'.format(list(out.keys())))
    return out

def plotAllTrials(data,key = 'curvature', yShift = 0):
    '''
    plotAllTrials - Plots all the timeseries specified by a key

    plotAllTrials(data, key = 'curvature', yShift = 0)
    Inputs:
    data -The data variable
    key - Plots all timeseries specified by the key
    yShift - The amount to shift the traces by along the y-axis
    '''
    import matplotlib.pyplot as plt
    var = data[key]
    plt.figure(figsize = (14,6))
    for trlNum,trl in enumerate(var):
        plt.plot(data['time'],trl-(yShift*trlNum),'k')


def prepareForUnet_1ch(I, sz = (512,512), n_jobs = 1,verbose = 5):
    """
    Resizes and adjusts dimensions of image or image stack for training or predicting with u-net
    """
    import apCode.volTools as volt
    if np.ndim(I)==2:
        I = I[np.newaxis,...]
    if n_jobs >1:
        from joblib import Parallel, delayed
        I = Parallel(n_jobs = n_jobs, verbose = verbose)(delayed(volt.img.resize)(img,sz) for img in I)
    else:
        I = [volt.img.resize(img, sz) for img in I]
    I = np.array(I)
    return I[...,np.newaxis]


def prepareForUnet_2ch(images, images_back, sz = (512,512)):
    """
    Resizes and reshapes images for U-net (2 image channels)
    training or prediction.
    Parameters
    ----------
    images: array, ([T,] M,N)
        Stack of raw images for training or predicting with the kaggle U-net
    images_back: array, ([T,], M,N)
        If T > 1|images, and T == 1|images_back, then assumes common background
    sz: tuple or array, (m,m)
        Dimensions of resized array to feed to the U-net
    Returns
    -------
    I: array, ([T,], M, N, 2), where T >= 1, and I[...,0] = raw images,
        and I[...,1] is background subtracted images
    """
    import apCode.volTools as volt
    images = volt.img.resize(images, sz)
    images_back = volt.img.resize(images_back,sz)
    if np.ndim(images)==2:
        images = images[np.newaxis,...]
    if np.ndim(images_back)==2:
        images_back = np.tile(images_back,(images.shape[0],1,1))
    images_bs = images-images_back.astype(int)
    images = images[...,np.newaxis]
    images_bs = (images_bs[..., np.newaxis])
    return np.concatenate((images,images_bs),axis = 3).astype(int)


def prob_images_with_unet(imgs, unet, **unet_predict_kwargs):
    """
    Return probability images generated using the specified U net. Presently,
    only works for 1 image channel U net
    Parameters
    ----------
    imgs: array, (nImgs, *imgDims)
        Images to segment with U net
    unet: unet model or path to unet model (keras)
        Pre-trained U net for use in segmenting images
    unet_predict_kwargs: dict
        Key word arguments for unet.predict function. See documentation on
        keras models
    Returns
    -------
    imgs_prob: array, (nImgs, *imgDims)
        Probability images
    """
    from apCode.machineLearning import ml as mlearn
    import apCode.volTools as volt
    if imgs.ndim==2:
        imgs = imgs[np.newaxis,...]
    imgDims = imgs.shape[-2:]
    uDims = unet.input_shape[1:3]
    resized=False
    if np.sum(np.abs(np.array(uDims)-np.array(imgDims)))>0:
        resized = True
        # print(f'Resizing imgs: {imgDims} --> {uDims}')
        imgs = volt.img.resize(imgs, uDims)

    # print('Predicting...')
    imgs_prob = np.squeeze(unet.predict(imgs[..., None],
                                        **unet_predict_kwargs))

    if resized:
        print('Resizing back...')
        imgs_prob = volt.img.resize(imgs_prob, imgDims)
    return imgs_prob


def prob_images_with_unet_patches(imgs, unet, verbose=2):
    """
    Return probability images generated using the specified U net.
    Parameters
    ----------
    imgs: array, (nImgs, *imgDims)
        Images to segment with U net
    unet: unet model or path to unet model (keras)
        Pre-trained U net for use in segmenting images
    Returns
    -------
    imgs_prob: array, (nImgs, *imgDims)
        Probability images
    """
    from apCode.machineLearning import ml as mlearn
    import apCode.volTools as volt
    if imgs.ndim==2:
        imgs = imgs[np.newaxis,...]
    imgDims = imgs.shape[-2:]
    patchSize = unet.input_shape[1:3]
    pow2 = np.log2(np.max(imgDims))
    resized = False
    if (np.mod(pow2, 1)!=0) | (pow2 < np.log2(patchSize[0])):
        pow2 = int(np.floor(pow2)+1)
        imgDims_rs = (2**pow2,)*2
        print('Resizing images...')
        imgs_rs = volt.img.resize(imgs, imgDims_rs)
        resized=True
    else:
        imgDims_rs = imgDims
        imgs_rs = imgs
    patchify = mlearn.PatchifyImgs(patchSize=patchSize, stride=patchSize[0],
                                   verbose=0)

    print('Patchifying images...')
    imgs_patch = patchify.transform_for_nn(imgs_rs)

    print('Predicting...')
    imgs_prob = np.squeeze(unet.predict(imgs_patch[..., None], verbose=verbose))
    imgs_prob = patchify.revert_for_nn(imgs_prob)

    print('Resizing back...')
    imgs_prob = volt.img.resize(imgs_prob, imgDims)
    return imgs_prob


def radiatingLinesAroundAPoint(pt, lineLength, dTheta = 15, dLine = 1):
    '''
    Given the coordinates of a point, returns the list of coordinates of a series of lines
        radiating from that point
    lines = radiatingLinesAroundAPoint(pt, lineLength, dTheta = 15, dLine=1)
    Inputs:
    pt - x,y coordinates of a point from which the lines should radiate
    lineLength - length in pixels of the the line segments
    dTheta - angular spacing of the lines around the point. For instance setting
        dTheta = 90, returns 4 lines at right angles to each other
    dLine - Radial distance between points in the line
    '''
    import apCode.volTools as volt
    import numpy as np
    lines = []
    xInds = []
    yInds = []
    thetas = np.arange(0,360,dTheta)
    lineLengths = np.arange(1,lineLength+1,dLine)
    for theta in thetas:
        inds = list(map(lambda x:volt.pol2cart(x,theta), lineLengths))
        xInds = np.array(list(ind[0] for ind in inds)) + pt[0]
        yInds = np.array(list(ind[1] for ind in inds)) + pt[1]
        line = np.array([xInds, yInds])
        #print(inds)
        #inds[0] = inds[0] + pt[0]
        #inds[1] = inds[1] + pt[1]
        lines.append(line)
    return lines


def rotate2DPoints(pointArray,theta):
    '''
    rotate2DPoints - when given an array of points in 2D space, returns the set of points
        after rotation by specified angle in degrees, theta
    pointArray_rot = rotate2DPoints(pointArray,theta)
    Inputs:
    pointArray - 2 X n or n X 2 array where n is the number of points in 2D (x,y coordinates)
    theta - Angle in degrees to rotate point array by.
    '''
    import numpy as np
    pointArray = np.asmatrix(pointArray)
    if np.shape(pointArray)[0]>2 & np.shape(pointArray)[1]==2:
        pointArray = np.transpose(pointArray)
    theta = theta*(np.pi/180)
    T_rot = np.asmatrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    pointArray_rot = T_rot*pointArray
    return np.array(pointArray_rot)

def saveImagesForTraining(imgDir, imgInds):
    """
    Copies specified images from a directory into a subdirectory for later use in
    training NNs
    Parameters
    ----------
    imgDir: string
        Path to a directory of images
    imgInds: array, (N,)
        Indices of images to copy.
    Returns
    -------
    None
    """
    import shutil as sh
    import os
    import apCode.FileTools as ft
    import time

    dir_proc = os.path.join(imgDir,'proc')
    dir_train = os.path.join(dir_proc,'Training set_{}'.format(time.strftime('%Y%m%d')))
    dir_train_images = os.path.join(dir_train, 'images')

    if not os.path.exists(dir_proc):
        os.mkdir(dir_proc)
    if not os.path.exists(dir_train):
        os.mkdir(dir_train)
    if not os.path.exists(dir_train_images):
        os.mkdir(dir_train_images)

    imgNames = ft.findAndSortFilesInDir(imgDir, ext = 'bmp')
    [sh.copy(os.path.join(imgDir, imgName),dir_train_images) for imgName in imgNames[imgInds]]

def sortFastAndSlowFrames(imgDir,numFastFrames = 600, offset_fast = 0, numSlowFrames =3600,\
                          ext = 'bmp'):
    '''
    sortFastAndSlowFrames - Moves fast frame rate frames into a newly created subfolder
    fastDir, nTrls = sortFastAndSlowFrames(imgDir,...)
    Inputs:
    imgDir - Directory containing fast and slow images
    numFastFrames - Num of fast frames in each trial
        (num of pre-stim trials + num of post-stim trials)
    offset_fast - The number corresponding to the first fast frame
    numSlowFrames - The number of slow trials in each trial
    ext - Image extension

    '''
    import os, sys
    import shutil as sh
    import time
    import numpy as np
    ## Read all file names in dir, omit non-img files (based on extension),
    ##  sort in alphanumberic order, and collect into a list
    print("Obtaining img file names...")
    startTime= time.time()
    if ext.startswith('.') == False:
        ext = '.' + ext
    alreadyMoved = np.sort(list(filter(lambda x: x.endswith('.moved'), os.listdir(imgDir))))
    if len(alreadyMoved)>0:
        sys.exit('Failed! The files seem to have already been moved, if not, \
        delete ''.moved'' file in image directory and retry')
    imgsInFldr = np.sort(list(filter(lambda x: x.endswith(ext), os.listdir(imgDir))))
    frameList = np.arange(len(imgsInFldr))
    trlLen = numFastFrames + numSlowFrames
    remList = np.mod(frameList, trlLen)
    fastInds = np.array(np.where(remList <= (numFastFrames-1)))[0]
    fastList = frameList[fastInds + offset_fast]
    slowList = np.setdiff1d(frameList,fastList)
    if np.mod(len(imgsInFldr),numFastFrames+numSlowFrames)!=0:
        sys.exit('# of images in folder does not evenly divide by # of images \
        in a trial, check to make sure inputs are correct!')
    nTrls = len(imgsInFldr)/(numFastFrames + numSlowFrames)
    print(int(nTrls), 'trials detected!')
    print('Obtaining fast frames...')
    fastFrames = imgsInFldr[fastList]
    print('Obtaining slow frames...')
    slowFrames = imgsInFldr[slowList]
    ts = time.strftime('%m-%d-%y-%H%M%S',time.localtime())
    fastDir = os.path.join(imgDir,'fastDir_' + ts)
    os.mkdir(fastDir)
    slowDir = os.path.join(imgDir,'slowDir_' + ts)
    os.mkdir(slowDir)

    print('Moving fast frames...')
    print(fastDir)
    trlCount = 0
    for frameNum,frame in enumerate(fastFrames):
        try:
            sh.move(os.path.join(imgDir,frame),fastDir)
        except:
            print(frame ,'not transfered')
        if np.mod(frameNum,numFastFrames)==0:
            trlCount = trlCount + 1
            print('Trl', trlCount, 'complete')

    print('Moving slow frames...')
    print(slowDir)
    trlCount = 0
    for frameNum,frame in enumerate(slowFrames):
        try:
            sh.move(os.path.join(imgDir,frame),slowDir)
        except:
            print(frame,'not transfered')
        if np.mod(frameNum,numSlowFrames)==0:
            trlCount = trlCount+1
            print('Trl', trlCount, 'complete')

    open(os.path.join(imgDir,'.moved'),mode = 'w').close()
    print (int((time.time()-startTime)/60), 'min')
    print(time.asctime())
    return fastDir, slowDir, int(nTrls)

def sortFastAndSlowFrames_parallel(imgDir,numFastFrames = 600, offset_fast = 0, numSlowFrames =3600,\
                          ext = 'bmp',trlChunkSize = [],numCores = 20):
    '''
    sortFastAndSlowFrames - Moves fast frame rate frames into a newly created subfolder
    fastDir, nTrls = sortFastAndSlowFrames(imgDir,...)
    Inputs:
    imgDir - Directory containing fast and slow images
    numFastFrames - Num of fast frames in each trial
        (num of pre-stim trials + num of post-stim trials)
    offset_fast - The number corresponding to the first fast frame
    numSlowFrames - The number of slow trials in each trial
    ext - Image extension
    numCores - Number of CPU cores to use (if fewer than this many cores are available,
        then uses all those)

    '''
    import os, sys
    import shutil as sh
    import time
    import numpy as np
    from joblib import Parallel, delayed
    import multiprocessing as mp
    ## Read all file names in dir, omit non-img files (based on extension),
    ##  sort in alphanumberic order, and collect into a list
    print("Obtaining img file names...")
    startTime= time.time()
    if ext.startswith('.') == False:
        ext = '.' + ext
    alreadyMoved = np.sort(list(filter(lambda x: x.endswith('.moved'), os.listdir(imgDir))))
    if len(alreadyMoved)>0:
        sys.exit('Failed! The files seem to have already been moved, if not, \
        delete ''.moved'' file in image directory and retry')
    imgsInFldr = np.sort(list(filter(lambda x: x.endswith(ext), os.listdir(imgDir))))
    frameList = np.arange(len(imgsInFldr))
    trlLen = numFastFrames + numSlowFrames
    remList = np.mod(frameList, trlLen)
    fastInds = np.array(np.where(remList <= (numFastFrames-1)))[0]
    fastList = frameList[fastInds + offset_fast]
    slowList = np.setdiff1d(frameList,fastList)
    if np.mod(len(imgsInFldr),numFastFrames+numSlowFrames)!=0:
        sys.exit('# of images in folder does not evenly divide by # of images \
        in a trial, check to make sure inputs are correct!')
    nTrls = len(imgsInFldr)/(numFastFrames + numSlowFrames)
    print(int(nTrls), 'trials detected!')
    print('Obtaining fast frames...')
    fastFrames = imgsInFldr[fastList]
    print('Obtaining slow frames...')
    slowFrames = imgsInFldr[slowList]
    ts = time.strftime('%m-%d-%y-%H%M%S',time.localtime())
    fastDir = os.path.join(imgDir,'fastDir_' + ts)
    os.mkdir(fastDir)
    slowDir = os.path.join(imgDir,'slowDir_' + ts)
    os.mkdir(slowDir)
    fastFrames = [os.path.join(imgDir, frame) for frame in fastFrames]
    slowFrames = [os.path.join(imgDir,frame) for frame in slowFrames]
    numCores = np.min([mp.cpu_count(),numCores])
    print('Using', numCores, 'cpu cores')
    print('Moving fast frames to', fastDir)

    Parallel(n_jobs = numCores,verbose = 5)(delayed(sh.move)(src,fastDir) for src in fastFrames)

    print('Moving slow frames to', slowDir)
    Parallel(n_jobs = numCores,verbose =5)(delayed(sh.move)(src,slowDir) for src in slowFrames)
    open(os.path.join(imgDir,'.moved'),mode = 'w').close()
    print (int((time.time()-startTime)/60), 'min')
    print(time.asctime())
    return fastDir, slowDir, int(nTrls)

def sortFastAndSlowVibAndDark(fishDir,numFastFrames = 750, numSlowFrames = 30*60,\
    imgExt = 'bmp',process ='parallel'):
    '''
    Given the directory containing the images for all fish sorted by fish, but not by
    frame rate or stim type, sorts the images upto the level of the stim. Here, the script
    assumes that there are only two stim types, vibrational ('vib') and dark flash ('dark')
    and that the stimuli are presented alternately, starting with 'vib' first
    '''
    import os, time
    import apCode.FileTools as ft
    import numpy as np
    imgDirs = [os.path.join(fishDir,imgDir) for imgDir in ft.getSubDirsInDir(fishDir)]
    if len(imgDirs)==0:
        print('No subdirectories found in the path. The path must contain directories presumably corresponding to individual fish, with each fish directory containing the images to be sorted!')
    tic = time.time()
    for imgDir in imgDirs:
        print('Sorting fast and slow frames \n ', imgDir)
        if process.lower() == 'parallel':
            [fastDir,slowDir,nTrls] = sortFastAndSlowFrames_parallel(imgDir,\
            numFastFrames=numFastFrames, numSlowFrames=numSlowFrames, ext = imgExt)
        else:
            [fastDir,slowDir,nTrls] = sortFastAndSlowFrames(imgDir,\
            numFastFrames=numFastFrames, numSlowFrames=numSlowFrames, ext = imgExt)
        nTrls = int(nTrls)
        print('Sorting vib and dark trls in fast dir...')
        #filesInDir = ft.findAndSortFilesInDir(fastDir, ext = imgExt)
        trlLists = [list(np.arange(0,nTrls,2)),list(np.arange(1,nTrls,2))]
        dstLists = [os.path.join(fastDir,'vib'),os.path.join(fastDir,'dark')]
        sortIntoTrls(fastDir,numFastFrames,trlLists,dstLists)

        print('Sorting vib and tap trls in slow dir')
        #filesInDir = ft.findAndSortFilesInDir(slowDir, ext = imgExt)
        trlLists = [list(np.arange(0,nTrls,2)),list(np.arange(1,nTrls,2))]
        dstLists = [os.path.join(slowDir,'vib'),os.path.join(slowDir,'dark')]
        sortIntoTrls(slowDir,numSlowFrames,trlLists,dstLists)
    print((time.time()-tic)/60, 'sec')
    time.asctime()

def sortIntoTrls(trlDir, trlSize, trlLists = [], dstLists = [], ext = 'bmp', chunkSize = 1):
    '''
    Chunks all files within 'trlDir' into 'trlSize' lists and based on the sublists
    specified in 'trlLists' moves subsets of trials into destination locations
    specified by 'dstLists'
    Inputs:
    trlDir - Directory where images from all trials are located
    trlSize - Number of images/frames in a trial.
    trlLists - List of sublists containing indices of trls to be moved. If trLists = [],
        then creates it automatically based on 'chunkSize' (default =1), which
        determines how many trials to put in subdirectory.
    dstLists - List of destination paths where to move the different trial
        sublists. If trlList = [], a new trList is generated as described above.
        If for len(trlList) != len(dstLists) then generates a new dstLists.
    chunkSize - Scalar; the number of trials to chunk into a single
        folder when trlLists and dstLists are empty.
    '''
    import os, sys, time
    import shutil as sh
    import numpy as np
    import apCode.FileTools as ft
    ## Read all files in trlDir
    filesInDir = ft.findAndSortFilesInDir(trlDir,ext= ext)
    N = len(filesInDir)
    print (N, 'files in specified dir')
    if np.mod(N,trlSize) == 0:
        nTrls = N/trlSize
        print(nTrls, 'trials detected')
    else:
        sys.exit("'trlSize' does not evenly divide into the number of files in the \
        dir, check to make sure inputs are specified correctly!")

    if len(trlLists)==0:
        trlLists = np.arange(0,nTrls).astype(int)
        trlLists= sublistsFromList(trlLists,chunkSize)
    if (len(trlLists) != len(dstLists)):
        dstLists = []
        for item in trlLists:
            foo = ''
            for item_item in item:
                foo = foo + str(item_item) + '-'
            foo = foo[:-1]
            dstLists.append(os.path.join(trlDir,'Trl_' + foo))

    ## Sublist files into trials
    filesInDir_trl = sublistsFromList(filesInDir,trlSize)
    tic = time.time()
    if type(trlLists[0]) is list:
        for trlListNum, trlList in enumerate(trlLists):
            print('Moving the following trials', trlList)
            currList = [filesInDir_trl[item] for item in trlList]
            currList = np.array(currList).ravel()
            dst = dstLists[trlListNum]
            if not os.path.exists(dst):
                os.mkdir(dst)
            for item in currList:
                sh.move(os.path.join(trlDir,item),dst)
    else:
        print('Moving the following trials', trlLists)
        currList = [filesInDir_trl[item] for item in trlLists if np.intersect1d(item,trlLists)]
        dst = dstLists[trlListNum]
        if not os.path.exists(dst):
            os.mkdir(dst)
        for item in currList:
            sh.move(os.path.join(trlDir,item),dst)

    print(int(time.time()-tic), 'sec')


def sublistsFromList(inputList,chunkSize):
    '''
    Given a list, chunks it into sizes specified and returns the chunks as items
        in a new list
    '''
    import numpy as np
    subList,supList = [],[]
    for itemNum, item in enumerate(inputList):
        if np.mod(itemNum+1,chunkSize)==0:
            subList.append(item)
            supList.append(subList)
            subList = []
        else:
             subList.append(item)
    supList.append(subList)
    supList = list(filter(lambda x: len(x)!=0, supList)) # Remove zero-length lists
    return supList

def swimOnAndOffsets(x, ker_len = 50, thr =1, thr_slope = 0.5, plotBool = False):
    """
    Given a timeseries containing swim episodes, returns the estimated onsets
    and offsets of the swims
    Parameters
    ----------
    x: (N,) array-like
        Timeseries
    ker_len: scalar
        Size of the kernel to smooth swim with. A 100 ms kernel works well,
        which at 500 Hz sampling rate is 50 points.
    thr: scalar
        Threshold in units of std for considering something as signal and
        not noise in the smoothed timeseries (i.e., after convolution of the
        timeseries with the causal kernel).
    thr_slope: scalar
        Threshold in units of std when looking at the derivative of the smoothed
        timeseries
    plotBool = boolean
        If True, the plots std-normalized signal and convolved signal with overlaid on
        and offsets
    Returns
    -------
    ons: (n,) array
        Onsets of swims in timeseries
    offs: (n,) array
        Offsets of swims in timeseries
    signs: (n,) array
        Sign of slope the timeseries at onset. This can be used to align
        signals, for say plotting or averaging.
    """
    import apCode.SignalProcessingTools as spt
    import numpy as np
    import matplotlib.pyplot as plt
    ker = spt.gaussFun(np.arange(2*ker_len),mu=0, sigma=1)
    #ker = np.exp(-np.linspace(0,1,2*ker_len)*4)
    ind_max = np.argmax(ker)
    ker = ker[ind_max:int(len(ker/2))]
    #ker = ker[ind_max:]
    ker = ker/np.sum(ker)
    x_ker = np.convolve(np.abs(x),ker, mode = 'full')
    x_ker = x_ker[:len(x)]
    x_ker = x_ker/np.std(x_ker)
    ons,offs = spt.levelCrossings(x_ker,thr=thr)
    #print(ons,offs)
    epLens = np.array([off-on for on, off in zip(ons,offs)])

    #--- Discard brief excursions that are unlikely to be swims
    tooShortInds = np.where(epLens < ker_len/2)[0]
    ons = np.delete(ons,tooShortInds)
    offs = np.delete(offs,tooShortInds)

    dx_ker = np.gradient(x_ker)
    dx_ker = dx_ker/np.std(dx_ker)
    ddx_ker = np.gradient(dx_ker)
    ddx_ker = ddx_ker/np.std(ddx_ker)
    pks_up = spt.findPeaks(ddx_ker, pol=1)[0]
    pks_down = spt.findPeaks(ddx_ker, pol = -1)[0]

    if (np.size(ons)==1) & (np.size(offs)==0):
        #print('No offsets, setting to signal end.')
        offs = [len(x)-1]
    elif (np.size(ons)==0) & (np.size(offs)==1):
        #print('No onsets, setting to singal start.')
        ons = [0]
    ons_new, offs_new, signs = [],[],[]
    for on, off in zip(ons, offs):
        if np.size(off)==0:
            off = len(x)
        elif np.size(on)==0:
            on = 0

        pks_before = pks_up[np.where(pks_up<=on)[0]]
        if np.size(pks_before)==0:
            ons_new.append(on)
        else:
            ons_new.append(pks_before[-1])

        pks_after = pks_down[np.where(pks_down>=off)[0]]
        if np.size(pks_after)==0:
            offs_new.append(off)
        else:
            offs_new.append(pks_after[0])
        signs.append(np.sign(np.gradient(x)[on]))

    if plotBool:
        plt.plot(x/np.std(x), label = 'Original')
        plt.plot(x_ker, alpha = 0.2, label = 'Convolved')
        for on in ons_new:
            plt.axvline(x=on, color='g', linestyle=':', label='Onset',
                        alpha=0.5)
        for off in offs_new:
            plt.axvline(x=off, color='r', linestyle=':', label='Offset',
                        alpha=0.5)
        plt.legend()
    return np.array(ons_new), np.array(offs_new), np.array(signs)

def tail_angles_from_raw_imgs_using_unet(imgDir, unet, ext='bmp', imgInds=None,
                                         motion_threshold_perc=None,
                                         nImgs_for_back=1000, block_size=750,
                                         search_radius=10, n_iter=2,
                                         cropSize=140, midlines_nPts=50,
                                         midlines_smooth=20, resume=False):
    """
    Process images using U net upto the extraction of tail angles and save to
    an existing or new HDF file in a subfolder ('proc') of the image diretory
    Parameters
    ----------
    imgDir: str
        Directory of raw images
    unet: Keras model
        Trained U net model
    ext: str
        File extension of images in the specified directory
    imgInds: array (n,) or None
        Indices of select images to process
    motion_threshold_perc: scalar (between 0-100) or None
        If not None, then detects motion from images using estimate_motion and
        uses this value as the percentile threshold for motion. This will
        restrict processing to frames with motion and 100 frames around motion
        on- and offset. A good value is 60
    nImgs_for_back: int
        At most this many images are used for computing background
    block_size: int
        Size of image blocks to process at once for easing memory load
    search_radius: scalar
        Parameter r in track.findFish
    n_iter: int
        Eponymous parameter in track.findFish
    cropSize: int
        Size to which probability images are to be cropped
    midlines_nPts: int
        Final length of midlines in points
    midlines_smooth: int
        Smoothing factor for midlines. See geom.smoothen_curve
    resume: bool
        Not yet implemented
    Returns
    -------
    Path to HDF file storing relevant info
    """
    from apCode.volTools import dask_array_from_image_sequence
    from apCode.FileTools import findAndSortFilesInDir, sublistsFromList
    from apCode.util import timestamp
    from apCode.hdf import createOrAppendToHdf
    from apCode.behavior.headFixed import midlinesFromImages
    from apCode import geom
    from apCode.SignalProcessingTools import interp, levelCrossings

    print('Reading images into dask array')
    imgs = dask_array_from_image_sequence(imgDir, ext='bmp')
    nImgs_total = imgs.shape[0]

    print('Computing background')
    bgd = track.computeBackground(imgs, n=nImgs_for_back).compute()

    if motion_threshold_perc is not None:
        motion = estimate_motion(imgs, bgd=bgd)
        n_peri = 100
        thresh_motion = np.percentile(motion, motion_threshold_perc)
        ons, offs = levelCrossings(motion, thresh_motion)
        inds_motion = []
        for on, off in zip(ons, offs):
            inds_motion.extend(np.arange(on - n_peri, off + n_peri))
        inds_motion = np.array(inds_motion)
        inds_motion = np.delete(inds_motion, np.where((inds_motion<0) |
                               (inds_motion>len(motion))), axis=0)
        inds_motion = np.unique(inds_motion)
    else:
        inds_motion = np.arange(imgs.shape[0])

    if imgInds is None:
        imgInds = np.arange(imgs.shape[0])
    imgInds = np.intersect1d(imgInds, inds_motion)
    imgs = imgs[imgInds]
    p = np.round(100*len(imgInds)/nImgs_total)
    print(f'{p} % of images being processed')
    procDir = os.path.join(imgDir, 'proc')
    if not os.path.exists(procDir):
        os.mkdir(procDir)
        fn_hFile = f'procData_{util.timestamp()}.h5'
    else:
        fn_hFile = findAndSortFilesInDir(imgDir, search_str='procData',
                                         ext='h5')
        if len(fn_hFile)>0:
            fn_hFile = fn_hFile[-1]
        else:
            fn_hFile = f'procData_{timestamp()}.h5'
    path_hFile = os.path.join(procDir, fn_hFile)
    with h5py.File(os.path.join(path_hFile), mode='a') as hFile:
        hFile = createOrAppendToHdf(hFile, 'img_background', bgd, verbose=True)
        if motion_threshold_perc is not None:
            hFile = createOrAppendToHdf(hFile, 'motion_from_imgs', motion,
                                        verbose=True)
        inds_blocks = sublistsFromList(imgInds, block_size)
        ta, fp = [], []
        inds_kept = []
        for iBlock, inds_ in enumerate(inds_blocks):
            print(f'Block # {iBlock+1}/{len(inds_blocks)}')
            imgs_now = imgs[inds_].compute()
            imgs_fish, imgs_prob = fish_imgs_from_raw(imgs_now, unet, bgd=bgd,
                                                      verbose=0)
            imgs_fish_grad = -imgs_now*imgs_fish
            fp = track.findFish(imgs_fish_grad, back_img=None,
                                r=search_radius, n_iter=n_iter)
            try:
                # In case fish was not detected in a few images
                fp = interp.nanInterp1d(fp)
            except:
                # Fails for edge NaNs because extrapolation needed
                pass
            non_nan_inds = np.where(np.isnan(fp.sum(axis=1))==False)[0]
            print('Cropping images...')
            imgs_crop = track.cropImgsAroundFish(imgs_fish[non_nan_inds],
                                                 fp, cropSize=cropSize)
            pxls_sum = np.apply_over_axes(np.sum, imgs_crop, [1, 2]).flatten()
            non_zero_inds = np.where(pxls_sum > 3)[0]
            inds_kept_block = non_nan_inds[non_zero_inds]
            print('Extracting midlines...')
            midlines, inds_kept_midlines =\
                track.midlines_from_binary_imgs(imgs_crop[inds_kept_block],
                                                n_pts=midlines_nPts,
                                                smooth=midlines_smooth)
            inds_kept_block = inds_kept_block[inds_kept_midlines]
            frameInds_kept = np.array(inds_)[inds_kept_block]
            inds_kept.extend(frameInds_kept)

            print('Computing tail angles...')
            kappas = track.curvaturesAlongMidline(midlines, n=midlines_nPts)
            tailAngles = np.cumsum(kappas, axis=0)
            keyVals = [('imgs_fish_crop', imgs_crop), ('imgs_prob', imgs_prob),
                       ('fishPos', fp), ('midlines', midlines.T),
                       ('tailAngles', tailAngles.T),
                       ('frameInds_processed', frameInds_kept)]
            for key, val in keyVals:
                if (key in hFile) & (iBlock==0):
                    del hFile['key']
                else:
                    hFile = createOrAppendToHdf(hFile, key, val, verbose=True)
    return hFilePath

class track():
    import apCode.volTools as volt
    def assessTracking(curve, nFramesInTrl = 750, responseWin = (0,300)):
        """
        Parameters
        ----------
        curve: array, (L,N)
            Tail curvatures; L is number of points on fish's tail and N is total number of time points for experiment
        nFramesInTrl: scalar
            Number of frames in each trial.
        Returns
        -------
        r: (M,)
            A vector of M values wherein m is the total number of trials, and each value is a measure of how
            good the tracking for a given trial is. Computed based on lag-1 autocorrelation.
        """
        import numpy as np
        import apCode.FileTools as ft
        curve_trl = ft.sublistsFromList(curve.T, nFramesInTrl)
        r =[]
        for c in curve_trl:
            if responseWin == None:
                c_flat = np.array(c).T.flatten()
            else:
                c_flat = np.array(c).T.flatten()[np.arange(responseWin[0], responseWin[-1])]
            r.append(np.corrcoef(c_flat[:-1],c_flat[1:])[0,1])
        return np.array(r)

    def _computeBackground(imgDir, n=1000, ext= 'bmp', n_jobs=32, verbose=0,
                           override_and_save=True, func='mean'):
        """
        Given the path to an image directory, reads uniformly spaced images and returns
        the average image, which can be used an a background image. Also saves this
        background image in the 'proc' subdirectory.
        Parameters
        ----------
        imgDir: string
            Path to a directory of images
        n: scalar
            Number of images to use for computing background
        ext: string
            Extension of the images to be read
        n_jobs: scalar
            Number of workers for parallel processing
        verbose: scalar
            Verbosity of output during parallel processing
        override_and_save: Boolean
            If True, saves background image, even if one already exists in the
            path
        func: str, 'mean', 'max', 'median' or 'min'
            Function to use when computing background
        Returns
        -------
        img_back: array, (M,N)
            Background image
        """
        import apCode.FileTools as ft
        from joblib import Parallel, delayed
        from skimage.io import imread, imsave
        ### Check to see if there's already a background image
        procDir = os.path.join(imgDir,'proc')
        if not os.path.exists(procDir):
            os.mkdir(procDir)

        possible_backgrounds = ft.findAndSortFilesInDir(procDir,
                                                        search_str='background',
                                                        ext=ext)
        func = eval(f'np.{func}')
        if (np.size(possible_backgrounds) == 0) | override_and_save:
            imgNames = ft.findAndSortFilesInDir(imgDir, ext = ext)
            n = np.min((n,len(imgNames))) # Number of images to use to compute background
            imgPaths = list(map(lambda x: os.path.join(imgDir,x),
                                imgNames[np.unique(np.linspace(0,len(imgNames)-1,n).astype(int))]))
            if n_jobs >1:
                foo = Parallel(n_jobs = n_jobs, verbose = verbose)(delayed(imread)(ip) for ip in imgPaths)
            else:
                foo  = [imread(ip) for ip in imgPaths]
            img_back = func(np.array(foo), axis=0)
            imsave(os.path.join(procDir,'background.bmp'),img_back.astype('uint8'))
        else:
            img_back = imread(os.path.join(procDir,possible_backgrounds[0]))
#            print('Read preexisting background image')
#            imsave(os.path.join(procDir,'background.bmp'),img_back.astype('uint8'))
        return img_back

    def computeBackground(imgsOrDir, n=1000, ext= 'bmp', n_jobs=32, verbose=0,
                          override_and_save=True, func='mean'):
        """
        See _computeBackground
        """
        if isinstance(imgsOrDir, str):
            bgd = track._computeBackground(imgsOrDir, n=n, ext=ext,
                                           n_jobs=n_jobs,
                                           override_and_save=override_and_save,
                                           func=func)
        else:
            imgs = imgsOrDir
            nImgs = imgs.shape[0]
            n=np.minimum(n, nImgs)
            inds = np.unique(np.linspace(0, nImgs-1, n).astype(int))
            bgd = imgs[inds].mean(axis=0)
        return bgd

    def cropImgsAroundFish(images, fishPos, **kwargs):
        """
        See volt.img.cropImgsAroundPoints
        """
        from apCode.volTools import img as _img
        out = _img.cropImgsAroundPoints(images, fishPos, **kwargs)
        return out

    def curvaturesAlongMidline(midlines, n=50, n_jobs=32, verbose=0):
        """
        Returns curvatures along the midlines
        Parameters
        ----------
        midlines: list, (N,)
            Midlines rostrocaudally bisecting the tail of the fish.
            Each element of the list has dimensions (K,2), where K is a variable
            number representing the number of points making up the midline
        n: scalar
            Number of points after cubic spline interpolation. If n = None, then no interpolation
        n_jobs, verbose: See Parallel and delayed from joblib
        """
        from apCode.geom import dCurve, interpolate_curve
        import numpy as np
        from joblib import Parallel, delayed
        if (np.ndim(midlines)!=1) & isinstance(midlines,list):
            midlines = [midlines]
        len_midlines = np.array([len(ml) for ml in midlines])
        inds_long = np.where(len_midlines >4)[0]
        if n!=None:
            K = np.zeros((n,len(midlines)))*np.nan
            midlines = Parallel(n_jobs = n_jobs, verbose = verbose)\
            (delayed(interpolate_curve)(ml,n = n ) for ml in midlines[inds_long])
            kappas = Parallel(n_jobs= n_jobs, verbose = verbose)\
            (delayed(dCurve)(ml) for ml in midlines)
            kappas = np.array(kappas).T
            K[:,inds_long]= kappas
            kappas = K
        else:
            kappas = Parallel(n_jobs=n_jobs,verbose = verbose)(delayed(dCurve)(ml)\
                              for ml in midlines[inds_long])
            K = np.array([[np.nan]]*len_midlines)
            K[inds_long] = kappas
            kappas = K
        return kappas

    def _findFish(img, r=10, n_iter=2):
        """See findFish"""
        from apCode.geom import circularAreaAroundPoint
        def ff(img, seed, r):
            if np.any(seed == None):
                seed = np.unravel_index(np.argmax(img), img.shape)
            coords_around = circularAreaAroundPoint(seed, r).astype(int)
            for dim in range(len(coords_around)):
                inds_del = np.where((coords_around[dim] < 0) | (coords_around[dim] >= img.shape[dim]))
                coords_around = np.delete(coords_around, inds_del,axis = 1)
            wts = img[coords_around[0], coords_around[1]]
            if wts.sum()==0:
                return np.ones((2,))*np.nan
            else:
                wts = wts/wts.sum()
                return np.dot(coords_around, wts)
        fp = None
        for n in range(n_iter):
            fp = ff(img, fp, r)
        return fp[::-1]

    def findFish(imgsOrDir, back_img='compute', r=10, n_iter=2, **back_kwargs):
        """
        Returns x,y coordinates in an image of a fish's head centroid.
        Designed to work with fish images where eyes are the dimmest pixels,
        but the algorithm uses the brightest point for estimating fish head
        centroid, so if using imgs without background, flip pixel intensities
        first so that eyes become the brightest points.
        Parameters
        ----------
        imgOrPath: string or array (image) of shape (m,n)
            Path to image or image array in which to find fish.If img
            is given, but not background then assumes that fish pixels
            are brighter than background.
        back_img: array, (nRows, nCols), None, or 'compute'
            Background image to remove using back_img-img. If None, then assumes
            that background has been removed beforehand. If 'compute', then
            computes from images using track.computeBackground
        r: scalar
            Radius of a circular region around max-intensity pixel in the image.
            The pixel intensities are used as weights to compute the fish head
            centroid coordinates
        **back_kwargs: dict
            Keywords for track.computeBackground
        Returns
        -------
        fp: array-like, (2,)
            x-y coordinates (x,y) of fish's head centroid in image.

        """
        from skimage.io import imread
        import apCode.volTools as volt
        if isinstance(imgsOrDir, str):
            imgs = volt.dask_array_from_image_sequence(imgsOrDir)
        else:
            imgs = imgsOrDir
        if back_img is not None:
            if back_img is 'compute':
                back_img = track.computeBackground(imgs, **back_kwargs)
            imgs = back_img - imgs
        fp = [dask.delayed(track._findFish)(img, r=r, n_iter=n_iter)
              for img in imgs]
        fp = dask.compute(*fp, scheduler='processes')
        return np.array(fp)

    def fishImgsForBarycentricMidlines(I,I_prob, p_cutoff = 0.95, n_jobs = 32, verbose = 0):
        """
        When given raw fish images and probability images predicted by a trained U-net,
        returns images with single blobs that likely correspond to fish pixels.
        These images can then be used for midline estimation using the circular
        barycentric method (i.e., track.midlinesUsingBarycenters).
        """
        import numpy as np
        from apCode.behavior.FreeSwimBehavior import track

        def fishImgForBarycentricMidline(img, img_prob, p_cutoff):
            img_prob[img_prob < p_cutoff] = 0
            img = -img*img_prob
            img = track.guess_fish_blob(img)*img
            return img
        if np.ndim(I)==2:
            return fishImgForBarycentricMidline(I,I_prob,p_cutoff)
        else:
            if (n_jobs <2) | (n_jobs < I.shape[0]):
                I_fish = [fishImgForBarycentricMidline(img, img_prob, p_cutoff) for img, img_prob in zip(I, I_prob)]
            else:
                from joblib import Parallel, delayed
                I_fish = Parallel(n_jobs = n_jobs, verbose = verbose)
                (delayed(fishImgForBarycentricMidline)(img, img_prob, p_cutoff)
                for img, img_prob in zip(I, I_prob))
        return np.array(I_fish)

    def find_and_crop_imgs_around_fish(imgsOrDir, ext='bmp', back_img=None,
                                       r=10, n_iter=2, nImgs_for_back=1000,
                                       **crop_kwargs):
        """
        Parameters
        ----------
        imgsOrDir: array, (nImgs, *imgDims) or str
            Images or path to image directory
        ext = 'str'
            File extension for image files
        back_img: array, (*imgDims), None, or 'compute'
            Background image to subtract from images. If None, then no
            subtraction. If 'compute', computes background using
            track.computeBackground
        r: int
            Search radius for iterative estimation of fish head centroid. See
            track.findFish
        n_iter: int
            Number of search iterations for finding fish. See track.findFish
        nImgs_for_back: int
            Number of images to use for background image computation. See
            track.computeBackground
        crop_kwargs: dict
            Keyword arguments for track.cropImgsAroundFish
        Returns
        --------
        out: dict
            Dictionary with the following keys
            fishPos: array, (nImgs, 2)
                Fish position in images
            imgs_crop: array, (nImgs, *imgDims)
                Cropped images
        """
        if isinstance(imgsOrDir, str):
            imgs=volt.dask_array_from_image_sequence(imgsOrDir, ext='bmp')
        else:
            imgs = imgsOrDir
        fp = track.findFish(imgs, back_img=back_img, r=r, n_iter=2,
                            n=nImgs_for_back)
        imgs_crop = track.cropImgsAroundFish(imgs, fp, **crop_kwargs)
        out = dict(fishPos=fp, imgs_crop=imgs_crop)
        return out

    def fishImgsForMidline(I_raw, I_prob, filt_size=1, crop_size = None, n_jobs = 32,
                           verbose = 0, nThr = None, method = 'otsu'):
        """
        Given raw and probability images returned by U_net, returns images that are ready to
        use for midline estimation
        Parameters
        ----------
        I_raw: array, ([T,] M,N)
            Raw fish images
        I_prob: array, ([T,], m, n), where m <=m, and n <=N
            Probability images returned by trained U-net. The images can be
            smaller than raw images because of sub-sampling.Resizes before using for
            multiplication with raw images
        filt_size: scalar
            Size of gaussian filter used for smoothing
        crop_size: scalar
            Size to crop images to for easier midline detection. Default is None, in
            which case images are not cropped. Assumes only one fish in image.
        n_jobs, verbose: see Parallel, delayed
        method: string, 'custom' or 'otsu' (default)
            Algorithm for determining fish blob

        Returns
        -------
        I_fish: array, ([T,], crop_size, crop_size)
            Fish images ready for midline estimation
        fp: array, ([T,],2), if T = 1, then (2,)
        """
        import numpy as np
        from apCode.SignalProcessingTools import standardize
        from skimage.filters import gaussian
        from skimage.morphology import erosion
        from apCode.behavior.FreeSwimBehavior import track
        from apCode.volTools import img as img_

        def fishImgForMidline(img_raw, img_prob, filt_size, crop_size, method):
            if method == 'custom':
                img_prob[img_prob<0.95]= 0
            elif method == 'otsu':
                img_prob = img_.otsu(img_prob, mult = 0.2)

            img = erosion(standardize(-img_raw)*img_prob)
            if img.sum()==0:
                return img,np.ones(2,)*np.nan
            mask = (img>0).astype(int)
            img = gaussian(img,filt_size)*mask
            fp = track.findFish(img, r = 20)
            if not crop_size == None:
                img = track.cropImgsAroundFish(img, fp, cropSize = crop_size)
            if np.sum(img)!=0:
                img = track.guess_fish_blob(img)*img
            return img, fp

        if np.ndim(I_raw)==2:
            return fishImgForMidline(I_raw, I_prob, filt_size, crop_size)
        else:
            if (n_jobs > 1) & (n_jobs < I_raw.shape[0]):
                from joblib import Parallel, delayed
                I_fish, fp = zip(*Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(fishImgForMidline)
                (img_raw,img_prob,filt_size,crop_size, method)
                for img_raw, img_prob in zip(I_raw, I_prob)))
            else:
                I_fish, fp = zip(*[fishImgForMidline(img_raw, img_prob, filt_size, crop_size, method)
                for img_raw, img_prob in zip(I_raw, I_prob)])
            return np.array(I_fish), np.array(fp)

    def guess_fish_blob(img_prob, approx_fishLen = 100):
        """
        In a probability image (img_prob*-img) that consists of distinct islands,
        any of which could contain the fish, this function uses certain criteria to
        guess the correct island and mask with only this island
        Parameters
        ----------
        img_prob: array, (m,n)
            Probability image, or img_prob*img, where img is intensity image
        approx_fishLen:scalar
            Approximate fish length in pixel units. In my imaging conditions, approx
            pixel length = 0.05, and approx fish length = 5mm, so approx fish length
            in pixel units is int(5/0.05)
        """
        import numpy as np
#        import apCode.volTools as volt
        from skimage.measure import regionprops, label
        from skimage.morphology import thin
        import apCode.geom as geom
        from scipy.stats import rankdata

        def normalized_midline_profile_slope(img):
            ml = np.array(np.where(thin(img))).T
            try:
                ml_ord = ml[geom.sortPointsByWalking(ml),:]
            except:
                ml_ord = geom.sortPointsByKDTree(ml)[0]
            lp = img[ml_ord[:,0], ml_ord[:,1]]
            c = np.array([np.argsort(lp),lp]).T
            c_fit,mb = geom.fitLine(c)[:2]
            res = np.mean(np.sum((c_fit-c)**2,axis = 1)**0.5)
            m_norm = mb[0]/res
            return np.abs(m_norm)

        def midline_len(img):
            ml = np.array(np.where(thin(img))).T
            ml_ord = ml[geom.sortPointsByDist(ml),:]
            return np.sum(np.sum(np.diff(ml_ord,axis = 0)**2,axis = 1)**0.5)

        img_prob[np.where(np.isnan(img_prob))]=0
        img_bool = img_prob*0
        img_bool[np.where(img_prob>0)]=1
        rp = regionprops(label(img_bool),img_bool*img_prob)

        ### Remove small regions
        if len(rp)>1:
            delInds = np.zeros((len(rp),)).astype(int)
            for count, rp_ in enumerate(rp):
                if len(rp_.coords)<=6:
                    delInds[count] = 1
            rp = np.delete(rp, np.where(delInds)[0])

        if len(rp)==1:
            temp = img_bool*0
            temp[rp[0].coords[:,0], rp[0].coords[:,1]] = 1
            return temp.astype('uint8')
        elif len(rp)>1:
            try:
                maxInt = np.array([rp_.max_intensity for rp_ in rp])
#                blob_ind = np.argmax(maxInt)
                #normSlope = np.array([normalized_midline_profile_slope(rp_.intensity_image) for rp_ in rp])
                midlineLen = np.array([midline_len(rp_.image) for rp_ in rp])
                len_similarity = 1-(np.abs(approx_fishLen-midlineLen)/approx_fishLen)
                ranks = np.array(0.8*[rankdata(maxInt), 0.2*rankdata(len_similarity)])
                blob_ind = np.argmax(ranks.sum(axis=0))
            except:
                maxInt = np.array([rp_.max_intensity for rp_ in rp])
                blob_ind =  np.argmax(maxInt)
            temp = img_bool*0
            temp[rp[blob_ind].coords[:,0], rp[blob_ind].coords[:,1]] = 1
            return temp.astype('uint8')
        else:
            print('No blobs in image')
            return img_bool.astype('uint8')

    def interpolateMidlines(midlines, q = 90, kind:str = 'cubic', N:int = 100):
        """
        When given midlines returned by FreeSwimBehavior.track.midlinesFromImages, returns an array of
        midlines adjusted for the most common length and interpolated to fill NaNs
        Parameters
        ----------
        midlines: list (T,)
            T midlines of varying length
        n_jobs, verbose: See Parallel, delayed from joblib
        kind = string
            Kind of interpolation to use; see scipy.interpolate.interp1d
        fill_value: float, None (default), or "extrapolate"
            Values to fill with when outside the convex hull of the interpolation

        Returns
        -------
        midlines_interp, midlines_extrap: arrays,  (T, N, 2)
            Interpolated array of midlines (midlines_interp) and length-adjusted (by extrapolation)
            array of midlines respectively

        """
        import numpy as np
        from scipy.interpolate import griddata
        from apCode.geom import interpExtrapCurves

        def interp2D(C,kind):
            coords = np.where(np.isnan(C)==False)
            gy, gx = np.meshgrid(np.arange(C.shape[1]), np.arange(C.shape[0]))
            C_interp = griddata(coords, C[coords],(gx, gy), method = kind)
            return C_interp
        M =interpExtrapCurves(midlines,q = q, kind = kind, N = N)
        M_interp= np.apply_along_axis(interp2D,1,M,kind)
        return M_interp

    def interpolateMidlines_old(midlines, n_jobs = 32, verbose = 0, kind = 'cubic', fill_value = None):
        """
        When given midlines returned by FreeSwimBehavior.track.midlinesFromImages, returns an array of
        midlines adjusted for the most common length and interpolated to fill NaNs
        Parameters
        ----------
        midlines: list (T,)
            T midlines of varying length
        n_jobs, verbose: See Parallel, delayed from joblib
        kind = string
            Kind of interpolation to use; see scipy.interpolate.interp1d
        fill_value: float, None (default), or "extrapolate"
            Values to fill with when outside the convex hull of the interpolation

        Returns
        -------
        midlines_interp, midlines_extrap: arrays,  (T, N, 2)
            Interpolated array of midlines (midlines_interp) and length-adjusted (by extrapolation)
            array of midlines respectively

        """
        import numpy as np
        from scipy.interpolate import interp1d, griddata
        from joblib import Parallel, delayed
        from apCode.SignalProcessingTools import stats

        ### Custom functions
        dists = lambda ml: np.insert(np.cumsum(np.sum(np.diff(ml,axis = 0)**2,axis = 1)**0.5),0,0)
        interp_fit = lambda dml, ml, x, kind, bounds_error, axis, fill_value: interp1d(dml,ml,kind = kind, bounds_error = bounds_error, axis = axis, fill_value = fill_value)(x)
        totalDeviations = lambda midlines: np.sum(np.sum(np.diff(midlines,axis =0)**2,axis = 2)**0.5,axis = 1)
        ### Default inputs
        bounds_error, axis = False, 0
        if fill_value == None:
            fill_value = np.nan

        dMidlines = np.array(Parallel(n_jobs= n_jobs, verbose =0)(delayed(dists)(ml) for ml in midlines))
        midLens, dMids= zip(*[(_[-1], np.mean(np.diff(_))) for _ in dMidlines])
        midLens, dMids = np.array(midLens), np.array(dMids)
        x_50 = stats.valAtCumProb(midLens, func = 'lin', cumProb = 0.5, plotBool = False)
        x = np.arange(0, x_50, np.min(dMids))
#        print('Midlines: mean length = {}, length for 50 % = {}'.format(np.mean(midLens), x_50))
        M_extrap = np.array(Parallel(n_jobs = n_jobs, verbose = verbose)\
                         (delayed(interp_fit)(dml,ml,x,kind,bounds_error, axis, fill_value)\
                          for dml, ml in zip(dMidlines, midlines)))
        X,Y = M_extrap[...,0], M_extrap[...,1]
        gy, gx = np.meshgrid(np.arange(X.shape[1]), np.arange(X.shape[0]))
        totDev = totalDeviations(M_extrap)
        thr = stats.valAtCumProb(totDev, func = 'lin', plotBool = False)
        inds_keep = np.where(totDev < thr)[0]
#        print("{}/{} midlines interpolated".format(len(midlines)-len(inds_keep), len(midlines)))
        M_bool = X*0
        M_bool[inds_keep,...] = 1
        coords = np.where(M_bool==1)
        X_interp = griddata(coords, X[coords],(gx, gy), method = kind)
        Y_interp = griddata(coords, Y[coords],(gx, gy), method = kind)
        M = M_extrap*0
        M[...,0] = X_interp
        M[...,1] = Y_interp
        inds_nan = np.where(np.isnan(M))
        M[inds_nan] = M_extrap[inds_nan]
        return M, M_extrap

    def midlines_from_binary_imgs(imgs, n_pts=50, smooth=20,
                                  orientMidlines=True):
        """
        Get smoothened and length-adjusted midlines from binary fish images
        Parameters
        ----------
        imgs: array, (nImgs, *imgDims)
            Binary images
        n_pts: int
            Number of points on the midline
        smooth: scalar
            Curve smoothing factor. geom.smoothen_curve
            """
        from apCode import geom
        from apCode.behavior.headFixed import midlinesFromImages
        midlines = midlinesFromImages(imgs, orientMidlines=orientMidlines)[0]
        mLens = np.array([len(ml) for ml in midlines])
        inds_kept = np.where(mLens>=6)[0]
        # midlines_interp = geom.interpolateCurvesND(midlines[inds_kept],
        #                                            mode='2D', N=n_pts)
        midlines_interp = [dask.delayed(geom.interpolate_curve)(ml, n=n_pts)
                           for ml in midlines[inds_kept]]
        midlines_interp= [dask.delayed(geom.smoothen_curve)(ml, smooth_fixed=smooth)
                          for ml in midlines_interp]
        midlines_interp = dask.compute(*midlines_interp)
        midlines_interp = np.array(midlines_interp)
        # midlines_interp = geom.equalizeCurveLens(midlines_interp)
        return midlines_interp, inds_kept

    def midlinesFromImages(images, smooth=20, n=50, n_jobs=32, verbose=0,
                           orientMidlines=True):
        """
        Returns midlines from fish images generated by fishImgsForMidline
        Parameters
        ----------
        images: array, ([T,] M, N)
            Fish images generated by track.fishImgsForMidline
        n: scalar, integer
            Number of points in midline
        smooth: scalar
            Determines smoothing of midline. Larger values lead to more smoothing.
        n_jobs, verbose : See Parallel, delayed
        Returns
        -------
        midlines: array, ([T], n, 2)
            Array of midlines with the same number of points in each midline because
            of smoothing and interpolation
        ml_dist: tuple, (2,)
            First element is raw, pruned, and sorted midlines
            Second element is cumulative sum of distances (in pixel lengths) between
            successive midline points
        """
        from skimage.morphology import thin
        import apCode.geom as geom
        import numpy as np
        from apCode.geom import smoothen_curve

        getDists = lambda point, points: np.sum((point.reshape(1,-1)-points)**2,axis = 1)**0.5

        def identifyPointTypesOnMidline(ml):
            dist_adj = np.sqrt(2)+0.01
            L = np.array([len(np.where(getDists(ml_,ml) < dist_adj)[0]) for ml_ in ml])
            endInds = np.where(L==2)[0].astype(int)
            branchInds = np.where(L==4)[0].astype(int)
            middleInds = np.where(L==3)[0].astype(int)
            return middleInds, endInds, branchInds

        def midlineFromImg(img,smooth,n):
            ml = np.array(np.where(thin(img))).T
            inds_mid, inds_end, inds_branch = identifyPointTypesOnMidline(ml)
            ml = np.delete(ml,inds_branch,axis = 0)
            if len(ml)<3:
                return ml,(ml,[])
            inds_mid, inds_end, inds_branch = identifyPointTypesOnMidline(ml)
            wts = img[ml[:,1], ml[:,0]]
            ind_brightest = np.argmax(wts)
            if len(inds_end) ==0:
                ind_start = ind_brightest
            else:
                ind_start = inds_end[np.argmin(getDists(ml[ind_brightest,:], ml[inds_end,:]))]
            ord_sort = geom.sortPointsByWalking(ml, ref = ml[ind_start,:])
            ml_sort = ml[ord_sort,:]
            d = np.sum(np.diff(ml_sort,axis =0)**2,axis = 1)**0.5
            jumpInd = np.where(d>2*(np.sqrt(2)+0.01))[0]
            if len(jumpInd)>0:
                jumpInd = jumpInd[np.argmax(d[jumpInd])]
                len_pre = len(ml_sort[:jumpInd])
                len_post =len(ml_sort[(jumpInd+1):])
                if len_pre >= len_post:
                    ml_sort = ml_sort[:jumpInd,:]
                else:
                    ml_sort = ml_sort[(jumpInd+1):,:]

#            ml_smooth = interpolate_curve(ml_smooth) No point in interpolating if not changing length
            pt_brightest = np.unravel_index(np.argmax(img), img.shape)
            d_one = np.sum((ml_sort[0,:]-pt_brightest)**2)
            d_end = np.sum((ml_sort[-1,:]-pt_brightest)**2)
            if d_one > d_end:
#                ml_smooth = np.flipud(ml_smooth)
                ml_sort = np.flipud(ml_sort)
                d = np.flipud(d)
            d = np.cumsum(np.insert(d,0,0))
            ml_sort = np.fliplr(ml_sort)
            ml_dist = (ml_sort, d)
            if len(ml_sort)>4:
                ml_smooth = smoothen_curve(ml_sort,smooth = smooth)
            else:
                ml_smooth = ml_sort
            return ml_smooth, ml_dist

        def orientMidlines_(midlines):
            inds = np.arange(1,len(midlines))
            count = 0
            midlines_adj = midlines.copy()
            for ind in inds:
                d = np.sum((midlines_adj[0][0]-midlines_adj[ind][0])**2)
                d_flip = np.sum((midlines_adj[0][0]-np.flipud(midlines_adj[ind])[0])**2)
                if d> d_flip:
                    count = count + 1
                    midlines_adj[ind] = np.flipud(midlines_adj[ind])

            #print('{} midlines flipped'.format(count))
            return midlines_adj

        if np.ndim(images) == 2:
            midlines, ml_dist = midlineFromImg(images,smooth,n)
        else:
            if (n_jobs >1) & (n_jobs >= images.shape[0]):
                from joblib import Parallel, delayed
                midlines, ml_dist = zip(*Parallel(n_jobs = n_jobs, verbose = verbose)
                (delayed(midlineFromImg)(img, smooth,n) for img in images))
            else:
                midlines, ml_dist = zip(*[midlineFromImg(img, smooth, n) for img in images])
        midlines = np.array(midlines)
        if orientMidlines:
            midlines = orientMidlines_(midlines)
        return midlines, ml_dist


    def midlinesUsingBarycenters(images, r = 4, nCircles = 6, avoid_angle = 100, smooth =2,
                                 n_jobs = 32, verbose = 0):
        """
        Fish midlines using barycenter approach described by Adrien Jouary and German Sumbre.
        Parameters
        ----------
        images: array, ([T,], M, N)
            Fish images in which to compute midlines. Assumes a single fish in each image.
            T = number of images
        r: scalar
            Radius of all circles but the first one, which has radius 2*r
        nCircles: scalar
            Number of circles to use to cover the fish
        n_jobs, verbose: See Parallel, delayed
        Parameters
        ----------
        midlines: array, ([T,],nCircles+2,2)
            Coordinates of the midlines in each of the images
        """
        import apCode.geom as geom
        import apCode.behavior.FreeSwimBehavior as fsb
        import numpy as np
        def mlbc(img, r, nCircles):
            fp = fsb.track.findFish(img, r = r*2, n_iter = 1)
#            fp = np.flipud(np.unravel_index(np.argmax(img),img.shape))
            bc = np.zeros((nCircles+2,2))
            bc[0] = fp
            bc[1] = geom.baryCenterAlongCircleInImg(img,fp,r*2, avoid_angle= avoid_angle)[0]
            for r_ in range(2,nCircles+2):
                if np.any(np.isnan(bc[r_-1])):
                    bc[r_:,:] = np.nan
                    return bc
                else:
                    bc[r_] = geom.baryCenterAlongCircleInImg(img, bc[r_-1], r, ctr_prev= bc[r_-2], avoid_angle= avoid_angle)[0]
            return bc

        if np.ndim(images)==2:
            return mlbc(images,r,nCircles)
        else:
            if (n_jobs ==1) | (n_jobs < images.shape[0]):
                midlines = [mlbc(img, r, nCircles) for img in images]
            else:
                from joblib import Parallel, delayed
                midlines = Parallel(n_jobs = n_jobs, verbose=verbose)(delayed(mlbc)(img, r, nCircles) for img in images)
            midlines= np.array(midlines)
            return midlines

    def subtractBackground(I,method = 'mean',imgInds = None, wts = None):
        """
        Returns the background subtracted image stack
        Parameters
        ----------
        I: 3D array, shape = (T,M,N), where T = # of images in stack,
            M,N are the # of rows and columns respectively
        method: String
            ['mean'] | 'median' | 'mode'; specifies the method to use for generating
            the background image
        imgInds: Array-like, shape = (t,)
            If specified, then uses only images with these indices to compute the background
            image
        wts: Array-like, shape = (T,)
            If specified, weight the image by these values to compute the background image
        Returns
        ------
        I_back: 3D array of same shape as I
            Image stack with background removed

        """
        import numpy as np
        from scipy.stats import mode
        I_orig = I.copy()
        if imgInds != None:
            I = I[imgInds,:,:]
        if wts != None:
            wts = wts/np.sum(wts)
            wts = wts.reshape((-1,1,1))
            I = np.multiply(I,wts)
        img_back = I[0]*0
        if method.lower() == 'mean':
            img_back = np.mean(I,axis = 0)
        elif method.lower() == 'median':
            img_back = np.median(I,axis = 0)
        elif method.lower() == 'mode':
            img_back = np.squeeze(mode(I,axis =0)[0])
        else:
            print('Method not specified properly. Please check!')
        return I_orig-img_back,img_back
