# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:42:12 2015

@author: pujalaa
"""
import dask
def agglomerativeClustering(X,n_clusters, linkage = 'ward'):
    """
    Essentially AgglomerativeClustering from sklearn.cluster, but with a few
    more variables of interest appended to the output
    Parameters:
        See sklearn.cluster.AgglomerativeClustering
    """
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering as ag
    out = ag(linkage = linkage,n_clusters = n_clusters).fit(X)
    ctrs = np.zeros((n_clusters,np.shape(X)[1]))
    for lbl in np.unique(out.labels_):
        ctrs[lbl,:] = np.mean(X[np.where(out.labels_==lbl)[0],:],axis = 0)
    out.cluster_centers_ = ctrs
    return out

def colorCellsInImgStack(cellInds,imgStack,cellInfo, clrs, processing = 'serial',
                         idxOrder = 'F', dispColorMaps = False):
    """
    Given the requisite info, returns an image stack in which the cells specified by
    a set of input indices are colored by the colors specified in the input color map.
    The format of the variables
    Inputs/Parameters:
    cellInds - Indices of cells to color. These indices must be a subset of the indices
        in the input variable cellInfo
    imgStack - 3D imgStack of shape (z,x,y) to which cells belong
    cellInfo - cellInfo variable that has been imported into python by my subroutines
        and which correponds to cell.info created by Takashi's scripts
    clrs - Array-like or list of array-likes. If array-like, then shape must be
        (nCellInds x 4) rgba color map where each row in clrs indicates the
        color in which to paint a cell of the matching index in cellInds. The
        last column in clrs corresponds to alpha value. If clrs is a list then
        each item is the aforementioned rgba array. Thus, returns as many
        cell-painted image stacks as there are color maps in clrs
    idxOrder - 'C' or 'F'; determines whether indices should be seen as indexed in
        row-major (C-style) or column-major (Fortran-style)
    processing - 'serial' or 'parallel'; the latter results in parallel processing
        ('parallel' not working yet)
    dispColorMaps - Boolean; determines whether or not to display the colormap(s)
        being used in seaborn palplot style
    """
    import numpy as np
    import apCode.volTools as volt
    # from apCode.SignalProcessingTools import standardize
    import sys
    import matplotlib.pyplot as plt

    def colorCellInImgStack(cInd, imgStack, cellInfo, clr,
                            idxOrder=idxOrder):
        import numpy as np
        if isinstance(clr, tuple):
            clr = np.array(clr)
        pxlInds = cellInfo['inds'][cInd].ravel().astype(int)
        sliceInds = np.tile(cellInfo['slice'][cInd].astype(int),np.shape(pxlInds))
        imgDims = np.shape(imgStack)[1:3]
        coords = np.vstack((np.unravel_index(pxlInds,imgDims,order =idxOrder),sliceInds))
        nCh = np.shape(imgStack)[3]
        for clrChan in np.arange(nCh):
            imgStack[sliceInds[0],coords[0],coords[1],clrChan] =  clr[clrChan]
        return imgStack

    def getColoredStack(cellInds,I,cellInfo,clrMap,idxOrder = idxOrder):
        if processing.lower().find('parallel') == -1:
            dispChunk = int((0.5*len(cellInds)))
            for cNum, cInd in enumerate(cellInds):
                if np.mod(cNum,dispChunk) == 0:
                    print(str(cNum) + '/' + str(len(cellInds)))
                clrMap_arr = np.array(clrMap)
                I = colorCellInImgStack(cInd,I,cellInfo,clrMap_arr[cNum,:])
        else:
            from joblib import Parallel, delayed
            import multiprocessing
            nCores = multiprocessing.cpu_count()
            nCores = np.min((nCores,len(cellInds)))
            I = Parallel(n_jobs=nCores, verbose=5)
            (delayed(colorCellInImgStack)(cInd,I,clrMap_arr[cNum,:],
             idxOrder = idxOrder) for cNum, cInd in enumerate(cellInds))
        return I

    #I_norm = standardize(imgStack)
    I_norm = imgStack.copy()
    if np.ndim(I_norm)==2:
        I_norm = I_norm[np.newaxis,:,:]
    elif np.ndim(I_norm)>3:
        sys.stderr.write('Image stack cannot have more than 3 dimensions:')
        sys.exit()

    I = np.transpose(np.tile(I_norm,[4,1,1,1]),[1,2,3,0])*0
    print('Creating 4 channel rgba img stack...')
    for imgNum,img in enumerate(I_norm):
        if np.mod(imgNum,10)==0:
            print(int(100*(imgNum)/len(I_norm)), '%')
        foo = volt.img.gray2rgb(img)
        alphaSlice = np.ones(np.shape(img))
        I[imgNum,:,:,0:3] = foo
        I[imgNum,:,:,-1] = alphaSlice
    print('100%')

    if dispColorMaps:
        plt.figure()
        volt.palplot(clrs)
        plt.title('Colormaps being used')

    if isinstance(clrs, list):
        I_out = [getColoredStack(cellInds,I.copy(),cellInfo,clrMap,idxOrder=idxOrder) for clrMap in clrs]
    else:
        clrMap = clrs
        I_out = getColoredStack(cellInds,I.copy(),cellInfo,clrMap,idxOrder= idxOrder)

    return I_out

def detrendCa(caSig,stimInds):
    """
    Given a timeseries (such as a Ca2+ dF/F signal) and a set of indices at which activity
        can be expected to be near the presumed baseline (such as stimulus onset indices that mark
        the beginning of a trial), returns the signal after correction for slow fluctuations in
        baseline.
    Parameters:
    caSig - 1D array, Timeseries (such as Ca2+ signal) with fluctuating baseline
    stimInds -Indices where Ca2+ signal can be expected to be close to the baseline
    Returns:
    caSig_new - Timeseries corrected for fluctuations.
    """
    import numpy as np
    from scipy.interpolate import interp1d
    t = np.arange(len(caSig))
    stimInds = np.unique(np.concatenate(([0],stimInds[1:]-1, [len(t)-1])))
    f = interp1d(stimInds,caSig[stimInds], kind = 'cubic')
    return caSig-f(t)

def doubleExp(time, tau1, tau2, wt1):
    """
    Given a time vector and relevant parameters, generates a double exponential decay
    Parameters:
    time - time vector over which to generate the exponential
    tau1 - First time constant
    tau2 - Second time constant
    wt1 - Fractional weight of the first time constant for averaging
    """
    import numpy as np
    wt2 = 1-wt1
    time = time - time[0]
    e = wt1*np.exp(-time/tau1) + wt2*np.exp(-time/tau2)
    return e

def estimateCaDecayKinetics(time, signals, p0 = None, thr = 2, preTime = 10,
                            postTime = 40):
    """
    Given a time vector and Ca signal matrix of shape = (C,T), where
        C = # of cells, and T = # of time points (must match length of time
        vector), returns output of shape = (nSamples, 2), where the 1st and
        2nd columns contain the fast and slow decay tau estimates after
        fitting Ca2+ signals with  double exponential
    Parameters:
    time - Time vector of length T
    signals - Ca signals array of shape (nSamples,T)
    p0 - Array-like, (tau_fast, tau_slow, wt_fast), where tau_fast is the
        fast decay time constant (in sec), tau_slow is the slow decay
        constant, and wt_fast is the weight of the fast exponential (<1)
        for fitting the signal as a weighted sum of the fast and slow
        exponential. Default is None, in which case fitting optimization
        begins without initial estimate
    thr - Threshold for peak detection in Ca signals, in units of zscore
    preTime - Pre-peak time length of the Ca signals to include for segmentation
    postTime - Post-peak "           "          "               "
    Avinash Pujala, JRC, 2017

    """
    import numpy as np
    from scipy.optimize import curve_fit as cf
    import apCode.SignalProcessingTools as spt
    import apCode.AnalyzeEphysData as aed

    def doubleExp(time, tau1, tau2, wt1):
        wt2 = 1-wt1
        time = time - time[0]
        e = wt1*np.exp(-time/tau1) + wt2*np.exp(-time/tau2)
        return e

    def listToArray(x):
        lens = [len(item) for item in x]
        lenOfLens = len(lens)
        lens = lens[np.min((lenOfLens-1,2))]
        a = np.zeros((len(x),lens))
        delInds = []
        for itemNum,item in enumerate(x):
            if len(item) == lens:
                a[itemNum,:] = item
            else:
                delInds.append(itemNum)
        a = np.delete(a,delInds,axis = 0)
        return a, delInds
    if np.ndim(signals)==1:
        signals = np.reshape(signals,(1,len(signals)))
    dt = time[2]-time[1]
    pts_post = np.round(postTime/dt).astype(int)
    pts_pre = np.round(preTime/dt).astype(int)
    x_norm = spt.zscore(signals,axis = 1)
    x_seg, params, x_seg_fit = [],[],[]
    nSamples = np.shape(signals)[0]
    excludedSamples = np.zeros((nSamples,1))
    for nSample in np.arange(nSamples):
        inds_pk = spt.findPeaks(x_norm[nSample,:],thr = thr,ampType = 'rel')[0]
        if len(inds_pk)==0:
            print('Peak detection failed for sample #', nSample, '. Try lowering threshold')
            excludedSamples[nSample] = 1
        else:
            blah = aed.SegmentDataByEvents(signals[nSample,:],inds_pk,pts_pre,pts_post,axis = 0)
            blah = listToArray(blah)[0]
            blah = np.mean(blah,axis=0)
            x_seg.append(blah)
            ind_max = np.where(blah == np.max(blah))[0][0]
            y = spt.standardize(blah[ind_max:])
            t = np.arange(len(y))*dt
            popt,pcov = cf(doubleExp,t,y,p0 = [10,20, 0.5], bounds = (0,20))
            if popt[0]> popt[1]:
                popt[0:2] = popt[2:0:-1]
                popt[-1] = 1-popt[-1]
            params.append(popt)
            foo = doubleExp(t,popt[0],popt[1],popt[2])
            x_seg_fit.append(foo)
    excludedSamples = np.where(excludedSamples)[0]
    includedSamples = np.setdiff1d(np.arange(nSamples),excludedSamples)
    x_seg,delInds = listToArray(x_seg)
    params = np.delete(np.array(params),delInds,axis = 0)
    delInds = includedSamples[delInds]
    if len(delInds)>0:
        print('Sample #', delInds, 'excluded for short segment length. Consider decreasing pre-peak time length')
    excludedSamples = np.union1d(delInds,excludedSamples)

    x_seg = spt.standardize(np.array(x_seg),axis = 1)
    x_seg_fit = np.array(listToArray(x_seg_fit)[0])
    out = {'raw': x_seg,'fit': x_seg_fit,'params': np.array(params),'excludedSamples': excludedSamples}
    return out



def getCoefWeightedClrMap_posNeg(regr, cMap = 'jet_r', normed = True,
                                         alphaModulated = True,
                                         rsqModulated = False,
                                         scaling = 'joint'):
    """
    Given a regression object created by apCode.spim.imageAnalysis.regress,
    appeds 2 color maps to the object as attributes. Each of the color maps
    is created by weighting a set of colors (mathing the number of regression
    features) either by the postive (regrObj.clrMap_pos) or negative
    (regrObj.clrMap_neg) regression cofficients. The set of colors to be weighted
    are chosen from a color map passed to the function.

    Inputs:
    regr - Regression object created by regress in (apCcode/spim/imageAnalysis).
    cMap - Color map to pull colors from which are then used the basis for coloring
        the regressors based on their contribution to the output. Examples
        cMap = 'jet' (default)  or cMap = plt.cm.Accent
    normed = Boolean;  determines whether or no the coefficients to be used for
        weighting should be standarized such that for a given set of positive or
        negative coefficients, the weights are in the range of 0 and 1. If False
        uses raw coefficient values
    alphaModulated - Boolean; determines whether or not the alpha values for the
        output color maps are to be weighted by 0 to 1 standardized R square values
        for the regression.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from apCode.SignalProcessingTools import standardize
    import apCode.volTools as volt

    if isinstance(cMap, str):
        cMap = plt.cm.get_cmap(cMap)

    B = np.abs(regr.coef_.copy())

    if rsqModulated:
        B = B*np.tile(regr.Rsq_,[np.shape(B)[1],1]).transpose()

    if normed:
        if scaling.lower() == 'joint':
            B = standardize(B)
        elif scaling.lower() == 'independent':
            B = standardize(B,axis = 0)

    X = regr.X_

    cPos = np.tile(np.linspace(0,200,np.shape(X)[1]),[np.shape(regr.coef_)[0],1])
    volt.img.palplot(cMap(cPos[0,:].astype(int)))
    plt.title('Color basis', fontsize = 14)
    plt.show()
    B_sum = np.sum(B,axis = 1)
    clr_posNeg = cMap((np.sum(B*cPos,axis = 1)/B_sum).astype(int))
    volt.img.palplot(clr_posNeg)
    plt.title('Color map', fontsize = 14)
    plt.show()

    if alphaModulated:
        clr_posNeg[:,3] = standardize(regr.Rsq_)
    else:
        clr_posNeg[:,3] = 1

    regr.clrMap_posNeg_ = clr_posNeg

    return regr


def labelCellsInImgStack(cellInds,imgStack,cellInfo, vals, processing = 'serial',
                         idxOrder = 'F', dispColorMaps = False, splitPosNeg = False):
    """
    Given the requisite info, returns an image stack in which the cells specified by
    a set of input indices are colored by the colors specified in the input color map.
    The format of the variables
    Inputs/Parameters:
    cellInds - Indices of cells to color. These indices must be a subset of the indices
        in the input variable cellInfo
    imgStack - 3D imgStack of shape (z,x,y) to which cells belong
    cellInfo - cellInfo variable that has been imported into python by my subroutines
        and which correponds to cell.info created by Takashi's scripts
    vals - Pixel values to assign to cells. If shape = (nCells,1), then assigns unique
        value to each cell. if shape = (1,1) then assigns same value to all cells
    idxOrder - 'C' or 'F'; determines whether indices should be seen as indexed in
        row-major (C-style) or column-major (Fortran-style)
    processing - 'serial' or 'parallel'; the latter results in parallel processing
        ('parallel' not working yet)
    dispColorMaps - Boolean; determines whether or not to display the colormap(s)
        being used in seaborn palplot style
    splitPosNeg = Boolean; if True then splits positive and negative values and returns
        a list of two separate image stacks, with the 1st element for positive values
        and the 2nd for negative.

    Avinash Pujala, JRC, 2017
    """
    import numpy as np
    import apCode.volTools as volt
    #from apCode.SignalProcessingTools import standardize
    import sys
    import matplotlib.pyplot as plt

    def labelCellInImgStack(cInd,imgStack,cellInfo,val,idxOrder = idxOrder):
        pxlInds = cellInfo['inds'][cInd].ravel().astype(int)
        sliceInds = np.tile(cellInfo['slice'][cInd].astype(int),np.shape(pxlInds))
        imgDims = np.shape(imgStack)
        coords = np.vstack((np.unravel_index(pxlInds,imgDims[1:],order =idxOrder),sliceInds))
        imgStack[sliceInds[0],coords[0],coords[1]] = val
        return imgStack

    def getLabeledStack(cellInds,I,cellInfo,vals,idxOrder = idxOrder, processing = processing):
        if processing.lower().find('parallel') == -1:
            dispChunk = int((0.5*len(cellInds)))
            for cNum, cInd in enumerate(cellInds):
                if np.mod(cNum,dispChunk) == 0:
                    print(str(cNum) + '/' + str(len(cellInds)))
                print('Values', vals)
                I = labelCellInImgStack(cInd,I,cellInfo,vals[cNum],idxOrder = idxOrder)
        else:
            from joblib import Parallel, delayed
            import multiprocessing
            nCores = multiprocessing.cpu_count()
            nCores = np.min((nCores,len(cellInds)))
            I = Parallel(n_jobs=nCores, verbose=5)
            (delayed(labelCellInImgStack)(cInd,I,vals[cNum],
             idxOrder = idxOrder) for cNum, cInd in enumerate(cellInds))
        return I

    I_norm = imgStack.copy()
    if np.ndim(I_norm)==2:
        I_norm = I_norm[np.newaxis,:,:]
    elif np.ndim(I_norm)>3:
        sys.stderr.write('Image stack cannot have more than 3 dimensions:')
        sys.exit()

    if dispColorMaps:
        plt.figure()
        volt.palplot(vals)
        plt.title('Colormaps being used')

    if len(np.shape(cellInds))==0:
        cellInds = np.reshape(cellInds,(1,))

    if len(np.shape(vals)) ==0:
        vals = np.tile(vals,(len(cellInds),))

    if splitPosNeg:
        I_out = []
        posInds = np.where(vals>=0)[0]
        foo = getLabeledStack(cellInds[posInds],I_norm.copy(), cellInfo, vals[posInds],idxOrder = idxOrder,
                              processing = processing)
        I_out.append(foo)

        negInds = np.where(vals<0)[0]
        foo= getLabeledStack(cellInds[negInds],I_norm.copy(), cellInfo, vals[negInds],idxOrder = idxOrder,
                              processing = processing)
        I_out.append(foo)
    else:
        I_out = getLabeledStack(cellInds,I_norm, cellInfo,vals,idxOrder=idxOrder, processing = processing)

    return I_out

class plt(object):
    """
    Class of functions for making plots

    """
    def plotCentroids(time, centroids, stimTimes, time_ephys, ephys, scaled = False,
                      colors = None, xlabel = '',ylabel = '', title = ''):
        """
        Plots centroids resulting from some clustering method
        Parameters:
        time - Time vectors for centroids (optical samplign interval)
        centroids - Array of shape (M, N), where M is the number of centroids, and N is the #
            number of features (or time points)
        stimTimes - Times of stimulus onsets for overlaying vertical dashed lines
        time_ephys - Time axis for ephys data (usually sampled at higher rate)
        ephys - Ephys time series
        scaled - Boolean; If true, scales centroids individually, else scales jointly.
        colors - Array of shape (M,3) or (M,4). Colormap to use for plotting centroids

        """
        import apCode.SignalProcessingTools as spt
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt
        if scaled:
            centroids = spt.standardize(centroids,axis = 1)
        else:
            centroids = spt.standardize(centroids)

        ephys = spt.standardize(ephys)

        n_clusters = np.shape(centroids)[0]
        if np.any(colors == None):
            colors = np.array(sns.color_palette('colorblind',np.shape(centroids)[0]))
        elif np.shape(colors)[0] < np.shape(centroids)[0]:
            colors = np.tile(colors,(np.shape(centroids)[0],1))
            colors = colors[:np.shape(centroids)[0],:]

        if np.any(time == None):
            time = np.arange(np.shape(centroids)[1])

        plt.style.use(['dark_background','seaborn-poster'])
        for cc in np.arange(np.shape(centroids)[0]):
            plt.plot(time,centroids[cc,:]-np.mean(centroids[cc,:])-cc,color = colors[cc,:])
        plt.plot(time_ephys,ephys-np.mean(ephys)-cc-1,color = colors[0,:])
        yt = np.arange(n_clusters + 1)
        ytl = list(yt)
        ytl[-1] = 'ephys'
        plt.yticks(-yt, ytl)
        plt.xlabel(xlabel,fontsize = 16)
        plt.ylabel(ylabel, fontsize = 16)
        plt.box('off')
        plt.title(title, fontsize = 20)
        plt.grid('off')
        plt.xlim(time[0],time[-1])
        for st in stimTimes:
            plt.axvline(x = st, ymin = 0, ymax =1, alpha = 0.3,
                        color = 'w',linestyle = '--')

def readCellData(inDir):
    '''
    readCellData - Reads Takashi's cell data (not cell info)
    cellData = readCellData(inDir)

    '''
    import time, os, h5py
    import numpy as np
    import scipy.io as sio
    startTime = time.time()
    f = h5py.File(os.path.join(inDir,'cell_resp3.mat'))
    blah = f['cell_resp3']
    cell = {}
    cell['raw'] = np.zeros(np.shape(blah))
    blah.read_direct(cell['raw'])
    cellInfo = sio.loadmat(os.path.join(inDir,'cell_info_processed.mat'))
    cellInfo = cellInfo['cell_info'][0]
    for fldNum,fldName in enumerate(cellInfo.dtype.names):
        cell[fldName] = []
        for cellNum in range(len(cellInfo)):
            cell[fldName].append(cellInfo[cellNum][fldNum])
    print('Finished reading cell data')
    print(int(time.time()-startTime),'sec')
    return cell

def readPyData(inDir,fileName = 'pyData.mat'):
    '''
    readPyData - read matlab processed data from pyData.mat
    '''
    blah = {}
    import h5py, os
    filePath = os.path.join(inDir,fileName)
    f = h5py.File(filePath)
    data = f['data']
    for key in list(data.keys()):
        blah[key]  = data[key]
    return blah


def regress(X, Y, sampleWeight=None, n_jobs=1, individualRegression=False,
            method='standard', regularization='standard',
            alpha=1.0, **kwargs):
    """
    Perform a simple linear regression using sklearn.linear_model
    Inputs:
    X - Training data; numpy array or sparse matrix of
        shape (n_samples, n_features)
    Y  - Target data; numpy array of shape (n_samples, n_targets)
    sampleWeight - Individual weights for each sample; numpy array
        of shape (n_samples)
    individualRegression - Boolean; If True, will compute individual
        regressions for each of the regressors and return Rsq values for each
        individual regressor in regr.Rsq_ind_
    method - 'standard' or 'miri'. If 'miri', returns results of regression from
        method by Miri et al.(2011), where in each regressor is in turn regressed
        with the other regressors forming an orthonormal basis w.r.t if. If
        'standard', then regresses at once with regressors/features as is.
    regularization - 'standard', 'lasso', 'ridge'. Read up on sklearn
        documentation for details.
    alpha - Float, optional. Regularization strength, larger values result in more
        regularization. See sklearn.linear_model.Ridge or sklearn.linear_model.Lasso.
    **kwargs: See LinearRegression, Lasso, or Ridge from sklearn.linear_model
        Some commonly used **kwargs are
        normalize: boolean (default = False)
        fit_intercept: boolean (default = True)
        n_jobs: int (default = None)
            Number of parallel cores.
    References:
    Miri, A., Daie, K., Burdine, R.D., Aksay, E., and Tank, D.W. (2011).
        Regression-Based Identification of Behavior-Encoding Neurons During
        Large-Scale Optical Imaging of Neural Activity at Cellular Resolution.
        Journal of Neurophysiology 105, 964–980.

    """
    from sklearn import linear_model
    import numpy as np
    import apCode.SignalProcessingTools as spt
    def regress_ols(X, Y):
        """
        Get stats like T values and P values using statsmodels.api.OLS
        Parameters
        ----------
        X: array, (nSamples, nFeatures)
            Regressor/predictor variable array.
        Y: array, (nSamples, nTargets)
            Reponse variable array
        Returns
        --------
        R: list, (nTargets,)
            Each element is the result of fitting statsmodels.api.OLS on (Y_i, and X)
            where i = 1,2,...,nTargets.
        """
        import statsmodels.api as sm
        def fit_ols(y, X):
            res = sm.OLS(y, X).fit()
            return res
        X = sm.add_constant(X)
        R = []
        for y in Y.T:
            res = fit_ols(y, X)
            R.append(res)
        return R

    def getClrMapsForEachRegressor(betas, normed=True, cMap='PiYG',
                                   scaling=1, betaThr=None):
        """
        Given the coeffiecients(betas) from regression, returns a list of color
        maps, with each color map corresponding to the betas for a single
        regressor. These can be used by colorCellsInImgStack to create image
        stacks with cells colored by betas.
        Parameters:
        betas - Array-like with shape (nSamples, nFeatures).
        normed - Boolean; If True, normalizes betas such that for each feature
            the values range from -1 to 1.
        scaling - Not yet implemented
        betaThr - None(default),scalar,'auto'; Determines if any thresholding
            should be applied based on beta values. If None, then no
            thresholding, if scalar, then for beta values whose magnitude is
            less than this scalar, the alpha value in the color maps is set to
            zero. If 'auto' then automatically determines threshold
        """
        import apCode.SignalProcessingTools as spt
        import matplotlib.pyplot as plt
        import apCode.volTools as volt

        def getClrMap(x,cMap,maskInds):
            cm = np.zeros((len(x),4))*np.nan
            negInds = np.where(x<0)[0]
            posInds = np.where(x>=0)[0]
            x[negInds] = spt.mapToRange(np.hstack((x[negInds],[0,-1])),[0,127])[0:-2]
            x[posInds] = spt.mapToRange(np.hstack((x[posInds],[0,1])),[128,255])[0:-2]
            cm[negInds] = cMap(x[negInds].astype(int))
            cm[posInds] = cMap(x[posInds].astype(int))
            if len(maskInds) == 0:
                pass
            else:
                cm[maskInds,-1] = 0
            return cm

        if isinstance(cMap,str):
            cMap = plt.cm.get_cmap(cMap)

        if normed:
            betas = spt.standardize(betas, preserveSign = True, axis = 0)*scaling
        clrMaps = []
        for beta in betas.T:
            if betaThr == None:
                maskInds = []
            elif betaThr == 'auto':
                betaThr = volt.getGlobalThr(np.abs(beta))
                maskInds = np.where(np.abs(beta)<betaThr)
            else:
                maskInds = np.where(np.abs(beta)<betaThr)
            clrMaps.append(getClrMap(beta,cMap,maskInds))
        #clrMaps = [getClrMap(beta,cMap,betaThr) for beta in betas.transpose()]
        return clrMaps

    if regularization.lower() == 'ridge':
        regr = linear_model.Ridge(alpha = alpha,**kwargs)
    elif regularization.lower() == 'lasso':
        regr = linear_model.Lasso(alpha = alpha,**kwargs)
    else:
        regr = linear_model.LinearRegression(**kwargs)

    if method.lower() == 'miri':
        print('Computing regression using Miri method...')
        nFeatures = np.shape(X)[1]
        featureInds = np.arange(nFeatures)
        shuffleInds = featureInds.copy()
        Y = Y-np.mean(Y,axis = 0)
        M,B,S = [],[],[]
        for featureInd in featureInds:
            shuffleInds = np.concatenate((np.r_[featureInd],
                                          np.setdiff1d(featureInds,featureInd)))
            S.append(shuffleInds)
            print(shuffleInds)
            x = spt.linalg.orthonormalize(X[:,shuffleInds])
            regr.fit(x,Y)
            M.append(regr.coef_[:,0])
            B.append(regr.coef_[:,0])
        M,B,S = np.array(M).T,np.array(B).T,np.array(S)
        regr.coef_ = M
        regr.intercepts_ = B
        regr.intercept_ = np.zeros(np.shape(regr.intercept_))
        regr.shuffledInds_ = S
        Y_est = regr.predict(X)
    else:
        #print('Computing regression using all regressors...')
        regr.fit(X,Y)
        Y_est = regr.predict(X)

    if individualRegression:
        print('Computing regression for individual regressors...')
        R =[]
        for n, x in enumerate(X.T):
            print(n)
            r = linear_model.LinearRegression(**kwargs)
            y_est = r.fit(np.c_[x],Y).predict(np.c_[x])
            sse = spt.stats.sse(Y, y_est)
            sst = spt.stats.sst(Y, y_est)
            R.append(1-(sse/sst))
        R = np.array(R).T
    else:
        R = 'Not computed'

    fit_intercept = kwargs.get('fit_intercept')
    if fit_intercept:
        coef = np.c_[regr.intercept_, regr.coef_]
        X = np.c_[np.ones((len(X),)), X]
    else:
        coef = regr.coef_
    se_ = spt.stats.standardError(Y, Y_est, X)
    T_ = coef/se_
    regr.X_ = X
    regr.Y_ = Y
    regr.sse_ = spt.stats.sse(Y, Y_est)
    regr.mse_ = regr.sse_/(X.shape[0]-X.shape[1]-1)
    regr.sst_ = spt.stats.sst(Y)
    regr.Rsq_ = 1-(regr.sse_/regr.sst_)
    regr.Rsq_adj_ = spt.stats.rSq_adj(Y,Y_est,X.shape[1])
    regr.Rsq_ind_ = R
    regr.pred_ = Y_est
    regr.se_ = se_
    regr.T_ = T_
    regr.regress_ols = regress_ols
    #regr.getCoefWeightedClrMap_posNeg = getCoefWeightedClrMap_posNeg
    #regr.getClrMapsForEachRegressor = getClrMapsForEachRegressor

    return regr

def regress_ols(X,Y):
    """
    Get stats like T values and P values using statsmodels.api.OLS
    Parameters
    ----------
    X: array, (nSamples, nFeatures)
        Regressor/predictor variable array.
    Y: array, (nSamples, nTargets)
        Reponse variable array
    Returns
    --------
    R: list, (nTargets,)
        Each element is the result of fitting statsmodels.api.OLS on (Y_i, and X)
        where i = 1,2,...,nTargets.
    """
    import statsmodels.api as sm
    def fit_ols(y,X):
        res = sm.OLS(y,X).fit()
        return res
    X = sm.add_constant(X)
    R = [dask.delayed(fit_ols)(y,X) for y in Y.T]
    R = dask.compute(*R)
#    R = []
#    for y in Y.T:
#        res = fit_ols(y,X)
#        R.append(res)
    return R
