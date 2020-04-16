# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 02:43:37 2019

@author: pujalaa
"""

from apCode.FileTools import scanImageTifInfo


def basToDf(bas, n_pre_bas = 200e-3, n_post_bas = 1.5):
    import apCode.SignalProcessingTools as spt
    import pandas as pd
    import numpy as np
    import apCode.util as util
    df = {}
    dt = bas['t'][1]-bas['t'][0]
    Fs = int(1/dt)
    inds_stim, amps_stim, ch_stim = getStimInfo(bas)
    keys = list(bas.keys())
    ind_mtr  = util.findStrInList('den', keys)[-1]
    x = np.squeeze(np.array(bas[keys[ind_mtr]]))
    if x.shape[0]==2:
        x = x.T
    motor_trl = spt.segmentByEvents(spt.zscore(x,axis = 0),inds_stim, int(n_pre_bas*Fs), int(n_post_bas*Fs))
    motorTime_trl = spt.segmentByEvents(bas['t'],inds_stim, int(n_pre_bas*Fs), int(n_post_bas*Fs))
    trlNum = np.arange(len(inds_stim))+1
    df['trlNum'] = trlNum
    df['stimInd'] = inds_stim
    df['stimAmp'] = amps_stim
    df['stimHT'] = ch_stim
    df['motorActivity'] = motor_trl
    df['motorTime'] = motorTime_trl
    return pd.DataFrame(df)

def concatenateBas(basList):
    """
    Concatenates files written by BehavAndScan
    """
    import numpy as np
    bas_conc = dict(filename = '')
    filenames = []
    for iBas, bas in enumerate(basList):
        if iBas == 0:
            bas_conc.update(bas)
            filenames.append(bas_conc['filename'])
        else:
            for key in bas.keys():
                if key == 'filename':
                    filenames.append(bas[key])
                else:
                    bas_conc[key] = np.concatenate((bas_conc[key], bas[key]))
    bas_conc['filename'] = filenames
    return bas_conc

def mergeRois(roiNames, roiNames_old, roiName_new, roiMat, axis = 0):
    """
    Consolidates a list of provided roi names into a single roi name
    sums their timeries and returns new roi matrix
    Parameters
    ----------
    roiNames: list/array of strings
        All the roi names
    roiNames_old: list/array of strings
        Roi names to consolidate
    roiName_new : string
        New roi name after consolidation
    roiMat: array
        Array where one axis corresponds to rois
    axis: int
        Axis corresponding to the Rois

    Returns
    -------
    d: dictionary
        Dictionary with new roi names and relevant array
    """
    from apCode import util
    import numpy as np
    inds = []
    for rn in roiNames_old:
        inds.extend(util.findStrInList(rn,roiNames))
    inds = np.unique(np.array(inds))
    roiMat_keep = np.delete(roiMat,inds,axis = 0)
    roiNames_keep = np.delete(roiNames,inds)
    inds_diff = np.setdiff1d(np.arange(len(roiNames)),inds)
    roiMat_diff = np.delete(roiMat,inds_diff,axis = axis)
    roiNames_diff = np.delete(roiNames,inds_diff)

    preDotStrings = []
    for rn in roiNames_diff:
        preDotStrings.append(rn.split('.')[0])
    preDotStrings = np.unique(preDotStrings)

    d = {}
    sigs, names =[],[]
    for ps in preDotStrings:
        inds = util.findStrInList(ps,roiNames_diff)
        sig = np.take(roiMat_diff,inds, axis = axis)
        sig = sig.mean(axis = axis)*np.sqrt(sig.shape[axis])
        sigs.append(sig)
        name=f'{ps}.{roiName_new}'
        names.append(name)

    if not isinstance(roiNames_keep, list):
        roiNames_keep = np.array(roiNames_keep).tolist()
    roiNames_keep.extend(names)

    d['roiNames'] = np.array(roiNames_keep)
    d['roiMat'] = np.concatenate((roiMat_keep,np.array(sigs)),axis =axis)
    d['rois_consolidated'] = roiNames_diff
    return d

def consolidateNMlfRois(roiNames, roi_ts):
    """
    Assumes all nMLF neurons have names starting with 'Me'. Then, inds
    roi indices for consolidation of both sides. Only consolidates along the first
    axis, so make sure roi_ts has ROIs as first axis
    Parameters
    ----------
    roi_ts: array, (nRois, ...)
    """
    import numpy as np
    from apCode import util
    sideInds = [[],[]]
    roiNames_unique = np.unique(roiNames)
    inds_mlf_left = util.findStrInList('L.Me',roiNames_unique)
    inds_mlf_right = util.findStrInList('R.Me', roiNames_unique)
    inds_mlf_both = [inds_mlf_left, inds_mlf_right]
    for iSide in range(2):
        inds_side = inds_mlf_both[iSide]
        for iRoi in inds_side:
            roi = roiNames_unique[iRoi]
            ind = util.findStrInList(roi, roiNames)
            sideInds[iSide].extend(ind)
    si = []
    roiNames_new = np.array(roiNames.copy())
    for iSide, side in enumerate(sideInds):
        if side == 0:
            prefix = 'L'
        else:
            prefix = 'R'
        roiNames_new[side] = f'{prefix}.nMLF'
        si.append(np.array(side))

    roi_ts_new = []
    roiNames_new_unique = np.unique(roiNames_new)
    for rn in roiNames_new_unique:
        inds = util.findStrInList(rn, roiNames_new)
        roi_ts_new.append(roi_ts[inds].mean(axis = 0))
    roi_ts_new = np.array(roi_ts_new)
#    dic = dict(roiNames = roiNames_new_unique, roi_ts = roi_ts_new)
    return roiNames_new_unique, roi_ts_new


def dataFrameOfMatchedMtrAndCaTrls_singleFish(bas, ca = None, ch_camTrig = 'camTrigger', ch_stim = 'patch3',
                                   ch_switch = 'patch2', thr_camTrig= 4, Fs_bas = 6000, t_pre_bas = 0.2,\
                                   t_post_bas = 1.5, t_pre_ca = 1, t_post_ca = 10, n_jobs = 20):
    """
    Parameters
    ----------
    bas: dict
        Dictionary resulting from reading of BAS file
    ca: array, (nRois, nSamples)
        nSamples is expected to match the number of Camera triggers
    Returns
    -------
    df: Pandas dataframe
    """
    import numpy as np
    from scipy.stats import mode
    import apCode.SignalProcessingTools as spt
    from apCode import util
    import pandas as pd
    def motorFromBas(bas):
        keys = list(bas.keys())
        ind_mtr  = util.findStrInList('den', keys)
        if len(ind_mtr)>0:
            ind_mtr = ind_mtr[-1]
            x = np.squeeze(np.array(bas[keys[ind_mtr]]))
        else:
            if 'ch3' in bas:
                x = np.array([bas['ch3'], bas['ch4']])
            else:
                x = np.array([bas['ch1'], bas['ch2']])
        if x.shape[0]< x.shape[1]:
            x = x.T
        return x
    mtr = motorFromBas(bas)
    stimInds, stimAmps, stimHt = getStimInfo(bas, ch_stim = ch_stim,\
                                             ch_camTrig= ch_camTrig, ch_switch= ch_switch)
    camTrigInds  = getCamTrigInds(bas[ch_camTrig], thr = thr_camTrig)

#    dt_bas = 1/Fs_bas
    dt_ca = np.mean(np.diff(bas['t'][camTrigInds]))
    Fs_ca = int(np.round(1/dt_ca))
    if np.any(ca==None):
        nFrames = len(camTrigInds)
#        t_ca = np.arange(nFrames)*dt_ca
    else:
        nFrames = ca.shape[1]
#        t_ca = np.arange(nFrames)*dt_ca
        d = len(camTrigInds)-ca.shape[1]
        if d>0:
            print(f'Check threshold, {d} more camera triggers than expected')
        elif d<0:
            print('Check threshold, {d} fewer camera triggers than expected')
    n_pre_bas = int(np.round(t_pre_bas*Fs_bas))
    n_post_bas = int(np.round(t_post_bas*Fs_bas))
    n_pre_ca = int(np.round(t_pre_ca*Fs_ca))
    n_post_ca = int(np.round(t_post_ca*Fs_ca))

    stimFrames = spt.nearestMatchingInds(stimInds, camTrigInds)-2
    mtr_trl = spt.segmentByEvents(mtr,stimInds,n_pre_bas, n_post_bas, n_jobs = n_jobs)
    indsVec_ca = np.arange(nFrames)
    indsVec_ca_trl = spt.segmentByEvents(indsVec_ca, stimFrames,n_pre_ca, n_post_ca, n_jobs = n_jobs)
    trlNum_actual = np.arange(len(mtr_trl))
    lens_mtr = np.array([len(mtr_) for mtr_ in mtr_trl])
    lens_ca = np.array([len(inds_) for inds_ in indsVec_ca_trl])
    inds_del_mtr = np.where(lens_mtr!=mode(lens_mtr)[0])[0]
    inds_del_ca = np.where(lens_ca!= mode(lens_ca)[0])[0]
    inds_del_trl =np.union1d(inds_del_mtr, inds_del_ca)
    trlNum_actual = np.delete(trlNum_actual, inds_del_trl)
    mtr_trl = list(np.delete(mtr_trl, inds_del_trl,axis = 0)) # Converting to list so that can later put in dataframe
    indsVec_ca_trl = list(np.delete(indsVec_ca_trl, inds_del_trl, axis = 0))
    dic = dict(mtr_trl = mtr_trl, caInds_trl = indsVec_ca_trl)
    if not np.any(ca==None):
        if np.ndim(ca)==1:
            ca = ca[np.newaxis,...]
        ca_trl = [ca[:,inds_] for inds_ in indsVec_ca_trl]
        dic['ca_trl'] = ca_trl
    trlNum = np.arange(len(mtr_trl))
    dic['trlNum'] = trlNum
    dic['trlNum_actual'] = trlNum_actual
    dic['stimAmp'] = np.delete(stimAmps,inds_del_trl)
    dic['stimLoc'] = np.delete(stimHt, inds_del_trl)
    return pd.DataFrame(dic,columns = dic.keys())

def fish_dataframe_from_xls(xls, ch_stim = 'patch3', ch_switch = 'patch2'):
    import numpy as np
    import time
    df = []
    for i in np.arange(xls.shape[0]):
        xls_now = xls.iloc[i]
        path_now = xls_now.path
        tic = time.time()
        try:
            print('Reading bas...')
            bas = readMotorActivityFromPath(path_now)
            print('Getting stim info..')
            stimInds, stimAmps, stimHt = getStimInfo(bas, ch_stim= ch_stim,\
                                                          ch_switch= ch_switch,\
                                                          ch_camTrig=xls_now.camTrigCh)
            print('Reading roi timeseries...')
            roi_ts = readRoiTsFromPath(path_now)

            print('Organizing in a dataframe...')
            df_fish = dataFrameOfMatchedMtrAndCaTrls_singleFish(bas, roi_ts)

            print('Reading roi names')
            roiNames = readRoiNamesFromPath(path_now)
            df_fish = df_fish.assign(roiName = [roiNames]*len(df_fish))

            print('Consolidating ROIs and other adjustments...')
            df_fish = mNormDffAndConsolidateRois(df_fish)
            df_fish = df_fish.assign(fishIdx = xls_now.idxFish)
            df_fish = df_fish.assign(path = path_now)
            df_fish = df_fish.assign(stimHt = stimHt)
            df.append(df_fish)
        except:
            print(f'Failed for path = {path_now}')
        print(f'{int(time.time()-tic)}, s')
    return df

def getCamTrigInds(camTrig,thr = 4):
    from apCode.SignalProcessingTools import levelCrossings
    ons, offs = levelCrossings(camTrig, thr)
    return offs

def getImgIndsInTifs(tifInfo):
    import numpy as np
    nImgsInFile_cum = np.cumsum(tifInfo['nImagesInFile']*tifInfo['nChannelsInFile'])
    imgIndsInTifs = []
    for i in range(len(nImgsInFile_cum)):
        if i ==0:
            inds_ = np.arange(0,nImgsInFile_cum[i])
        else:
            inds_ = np.arange(nImgsInFile_cum[i-1], nImgsInFile_cum[i])
        imgIndsInTifs.append(inds_)
    return imgIndsInTifs

def getStimInfo(bas,ch_stim:str = 'patch3',ch_switch:str = 'patch2',\
                ch_camTrig = 'camTrigger', minPkDist =5*6000):
    import numpy as np
    import apCode.SignalProcessingTools as spt
    pks, amps = spt.findPeaks(np.array(bas[ch_stim]).flatten(),thr = 0.5, pol =1, minPkDist=minPkDist)
    inds_keep = np.where(amps>0)[0]
    pks = pks[inds_keep]
    amps = amps[inds_keep]
    foo = np.array(bas[ch_switch]).flatten()[pks]
    ht = np.array(['T']*len(pks))
    ht[np.where(foo>1)]='H'
    return pks, amps, ht

def mapToIpsiContra(roiNames, ipsi = 'R'):
    """
    A convenience function to replace side designation ('L' or 'R')
    in roi names with 'ipsi' or 'contra'.
    Parameters
    ----------
    roiNames: list/array of strings
        Collection of roi names
    ipsi: str, 'R'(default) or 'L'
        String indicating the real ipsilateral side
    Returns
    -------
    roiNames_new: List/array
        Collection of new roi names with appropriate string replacements
    """
    import numpy as np
    sides = np.array(['L','R'])
    contra = np.setdiff1d(sides,ipsi)[0]
    roiNames_new =[]
    for rn in list(roiNames):
        ind_ipsi = rn.find(ipsi)
        ind_contra = rn.find(contra)
        if ind_ipsi ==0:
            roiNames_new.append(rn.replace(ipsi,'ipsi',1))
        elif ind_contra == 0:
            roiNames_new.append(rn.replace(contra, 'contra', 1))
        else:
            roiNames_new.append(rn)
    return np.array(roiNames_new)

def matchRoiNamesAndSignalsToCore(roiNames, roi_ts, roiNames_core =\
                                  ['Mauthner', 'MiD2c','MiD2i','MiD3c', 'MiD3i',\
                                   'CaD', 'RoL1', 'RoL2', 'RoM1', 'RoM2M','RoM2L','RoM3',\
                                   'MiM1','MiM1MiV1','MiV1','MiV2','RoV3','CaV',\
                                   'nMLF','ves', 'MiR1', 'MiR2']):
    """
    Parameters
    ----------
    roi_ts: array, (nRois,...)
        ROI timeseries
    Returns
    -------
    rois_full, roi_ts_full: Full set of ROI names and timseries.
    """
    import numpy as np
    from apCode import util
    rois_left = np.array(['L.'+ rn for rn in roiNames_core])
    rois_right = np.array(['R.'+ rn for rn in roiNames_core])
    rois_lr = np.insert(rois_right, 0, rois_left) # Don't use np.union1d, because can mess with order
    dims= (len(rois_lr), *roi_ts.shape[1:])
    roi_ts_full = np.zeros(dims).copy()
    for iRoi, r in enumerate(roiNames):
        ind = np.argmax(util.sequenceMatch(r, rois_lr, case_sensitive = False))
        roi_ts_full[ind] = roi_ts_full[ind] + roi_ts[iRoi]
    return rois_lr, roi_ts_full

def mNormDffAndConsolidateRois(df_fish, n_preStim = 30, perc = 20,
                                 roiNames_core = ['Mauthner', 'MiD2c','MiD2i','MiD3c', 'MiD3i',\
                                   'CaD', 'RoL1', 'RoL2', 'RoM1', 'RoM2M','RoM2L','RoM3',\
                                   'MiM1','MiM1MiV1','MiV1','MiV2','RoV3','CaV',\
                                   'nMLF','ves', 'MiR1', 'MiR2']):
    import apCode.util as util
    import numpy as np
    ca_trl = np.array([np.array(ca_) for ca_ in df_fish.ca_trl])
    roiNames = df_fish.iloc[0].roiName
    ind_back = util.findStrInList('background', roiNames)[0]
    back = np.expand_dims(ca_trl[:,ind_back,:],axis = 1)
    ca_trl = ca_trl-back
    F = np.expand_dims(np.percentile(ca_trl[...,:n_preStim],perc, axis = 2),axis = 2)
    ind_m = util.findStrInList('R.Mauthner', roiNames)[0]
    m_norm = np.expand_dims(np.expand_dims(ca_trl[:,ind_m,:],axis = 1).max(axis = 2), axis = 2).mean(axis = 0)[np.newaxis,...]
    df_trl = (ca_trl-F)
    ca_trl_mNorm = df_trl/m_norm
    roiNames_strip = stripRoiNames(roiNames)
    lRois = np.array([len(_) for _ in roiNames_strip])
    inds_del = np.where(lRois == 0)
    if len(inds_del)>0:
        inds_del = inds_del[0]
    ind_unknown = util.findStrInList('unknown', roiNames)
    if len(ind_unknown)>0:
        ind_unknown = ind_unknown[0]
    inds_del = np.union1d(inds_del, ind_unknown)
    roiNames_new = np.delete(roiNames_strip, inds_del, axis = 0)
    ca_trl_new =  list(np.delete(ca_trl, inds_del, axis = 1))
    ca_trl_mNorm = np.delete(ca_trl_mNorm, inds_del, axis = 1)

    x = np.swapaxes(ca_trl_mNorm,0,1)
    roiNames_con, x = consolidateNMlfRois(roiNames_new, x)
    roiNames_match, x = matchRoiNamesAndSignalsToCore(roiNames_con, x,\
                                                    roiNames_core = roiNames_core)
    roiNames_split, x = splitRoisToLR(roiNames_match,x)
    ca_trl_mNorm = list(np.transpose(x,(2,0,1,3)))

    x = np.swapaxes(ca_trl_new,0,1)
    _, x = consolidateNMlfRois(roiNames_new, x)
    _, x = matchRoiNamesAndSignalsToCore(_, x,\
                                         roiNames_core = roiNames_core)
    _, x = splitRoisToLR(_,x)
    ca_trl_short = list(np.transpose(x,(2,0,1,3)))

    df_fish_new = df_fish.copy()
    nRows = len(df_fish_new)
    df_fish_new = df_fish_new.assign(ca_trl = ca_trl_new)
    df_fish_new = df_fish_new.assign(ca_trl_mNorm = ca_trl_mNorm)
    df_fish_new = df_fish_new.assign(ca_trl_short = ca_trl_short)
    df_fish_new = df_fish_new.assign(roiName_new = [roiNames_split]*nRows)
    return df_fish_new


def omitRois(roiNames,omitList):
    import apCode.util as util
    import numpy as np
    inds_del = []
    for o in omitList:
        inds = util.findStrInList(o,roiNames)
        if len(inds)>0:
            inds_del.extend(inds)
    inds_del = np.array(inds_del)
    inds_all = np.arange(len(roiNames))
    inds_keep = np.setdiff1d(inds_all,inds_del)
    return np.unique(inds_keep)


def readMotorActivityFromPath(path, ch_camTrig:str = 'camTrigger', \
                              ch_stim:str = 'patch3', ch_switch = 'patch2'):
#    from apCode import util
    from apCode.FileTools import findAndSortFilesInDir
#    import pandas as pd
    import os
    from scipy.io import loadmat
    import apCode.hdf as hdf
    import apCode.behavior.FreeSwimBehavior as fsb

    readFlag = False
    bas = None
    fn = findAndSortFilesInDir(path, ext = 'mat')
    if 'bas.mat' in fn:
        try:
            bas = loadmat(os.path.join(path,'bas.mat'), struct_as_record = False,\
                          squeeze_me = True)
            bas = hdf.toDict(bas['bas'])
            readFlag = True
        except:
            readFlag = False
            print(f'Could not read from "bas.mat" in {path}')
    if (not readFlag) & ('bas_mf.mat' in fn):
        try:
            bas = fsb.openMatFile(path, name_str='bas_mf', mode = 'r')
            bas = hdf.h5pyToDict(bas)
            readFlag = True
        except:
            print(f'Could not read "bas_mf.mat" in {path}')
    if (not readFlag) & ('procData.mat' in fn):
        dic = None
        with fsb.openMatFile(path, name_str = 'procData', mode = 'r') as procData:
            try:
#                bas = hdf.recursively_load_dict_contents_from_group(procData,'/bas')
                bas  = procData['bas']
                dic = procBasToDic(bas)
            except:
                print('Could not read "bas" from "procData.mat" file')
                pass
        bas = dic
    return bas

def procBasToDic(bas):
    import numpy as np
    dic = {}
    for key in bas.keys():
        try:
            dic[key] = np.squeeze(np.array(bas[key]))
        except:
            pass
    return dic

def readPeriStimulusTifImages(tifPaths, basPaths, nBasCh = 16, ch_camTrig = 'patch4', ch_stim = 'patch3',\
                  tifNameStr = '', time_preStim = 1, time_postStim = 10, thr_stim = 1.5,\
                  thr_camTrig = 1, maxAllowedTimeBetweenStimAndCamTrig = 0.5, n_jobs = 1):
    """
    Given the directory to .tif files stored by ScanImage (Bessel beam image settings) and the full
    path to the accompanying bas files returns a dictionary with values holding peri-stimulus
    image data (Ca activity) in trialized format along with some other pertinent info.

    Parameters
    ----------
    tifDir: string
        Path to directory holding .tif files written by ScanImage. In the current setting, each .tif
        file holds nCh*3000 images, where nCh = number of channels.
    basPath: string
        Full path to the bas (BehavAndScan) file accompanying the imaging session.
    nBasCh: scalar
        Number of signal channels in the bas file.
    ch_camTrig: string
        Name of the channel in bas corresponding to the camera trigger signal
    ch_stim: string
        Name of the stimulus signal channel in bas.
    tifNameStr: string
        Only .tif files containing this will be read.
    time_preStim: scalar
        The length of the pre-stimulus time to include when reading images
    time_postStim: scalar
        The length of the post-stimulus time.
    thr_stim: scalar
        Threshold to use for detection of stimuli in the stimulus channel of bas.
    thr_camTrig: scalar
        Threshold to use for detection of camera trigger onsets in the camera trigger channel of bas.
    maxAllowedTimeBetweenStimAndCamTrig: scalar
        If a camera trigger is separated in time by the nearest stimulus by longer than this time
        interval, then ignore this stimulus trial.
    Returns
    -------
    D: dict
        Dictionary contaning the following keys:
        'I': array, (nTrials, nTime, nImageChannels, imageWidth, imageHeight)
            Image hyperstack arranged in conveniently-accessible trialized format.
        'tifInfo': dict
            Dictionary holding useful image metadata. Has following keys:
            'filePaths': list of strings
                Paths to .tif files
            'nImagesInfile': scalar int
                Number of images in each .tif file after accounting of number of
                image channels
            'nChannelsInFile': scalar int
                Number of image channels
        'inds_stim': array of integers, (nStim,)
            Indices in bas coordinates where stimuli occurred.
        'inds_stim_img': array of integers, (nStim,)
            Indices in image coordinates where stimuli occurred
        'inds_camTrig': array of integers, (nCameraTriggers,)
            Indices in bas coordinates corresponding to the onsets of camera triggers.
        'bas': dict
            BehavAndScan data

    """
    import tifffile as tff
    import numpy as np
    import apCode.FileTools as ft
    import apCode.ephys as ephys
    import apCode.SignalProcessingTools as spt
    import apCode.util as util
#    import os
    def getImgIndsInTifs(tifInfo):
        nImgsInFile_cum = np.cumsum(tifInfo['nImagesInFile']*tifInfo['nChannelsInFile'])
        imgIndsInTifs = []
        for i in range(len(nImgsInFile_cum)):
            if i ==0:
                inds_ = np.arange(0,nImgsInFile_cum[i])
            else:
                inds_ = np.arange(nImgsInFile_cum[i-1], nImgsInFile_cum[i])
            imgIndsInTifs.append(inds_)
        return imgIndsInTifs


    ### Read relevant metadata from tif files in directory
    print('Reading ScanImage metadata from tif files...')

    tifInfo = ft.scanImageTifInfo(tifPaths)
    nCaImgs = np.sum(tifInfo['nImagesInFile'])

    ### Check for consistency in the number of image channels in all files.
    if len(np.unique(tifInfo['nChannelsInFile']))>1:
        print('Different number of image channels across files, check files!')
        return None
    nImgCh = tifInfo['nChannelsInFile'][0]
    print(f'{nCaImgs} {nImgCh}-channel images from all tif files')

    ### Get a list of indices corresponding to images in each of the tif files
    inds_imgsInTifs = getImgIndsInTifs(tifInfo)

    ### Read bas file to get stimulus and camera trigger indices required to align images and behavior
    print('Reading and joining bas files, detecting stimuli and camera triggers...')
    basList = [ephys.importCh(bp,nCh = nBasCh) for bp in basPaths]
    bas = concatenateBas(basList)
    inds_stim = spt.levelCrossings(bas[ch_stim], thr = thr_stim)[0]
    if len(inds_stim)==0:
        print(f'Only {len(inds_stim)} stims detected, check channel specification or threshold')
        return dict(bas = bas)
    inds_camTrig = spt.levelCrossings(bas[ch_camTrig], thr = thr_camTrig)[0]
    if len(inds_camTrig)==0:
        print(f'Only {len(inds_camTrig)} cam trigs detected, check channel specification or threshold')
        return dict(bas = bas)
    dt_vec = np.diff(bas['t'][inds_camTrig])
    dt_ca = np.round(np.mean(dt_vec)*100)/100
    print('Ca sampling rate = {}'.format(1/dt_ca))
    inds_del = np.where(dt_vec<=(0.5*dt_ca))[0]+1
    inds_camTrig = np.delete(inds_camTrig, inds_del)

    ### Deal with possible mismatch in number of camera trigger indices and number of images in tif files
    if nCaImgs < len(inds_camTrig):
        inds_camTrig = inds_camTrig[:nCaImgs]
        nCaImgs_extra = 0
    elif nCaImgs > len(inds_camTrig):
        nCaImgs_extra = nCaImgs-len(inds_camTrig)
    else:
        nCaImgs_extra = 0
        print('{} extra Ca2+ images'.format(nCaImgs_extra))
    print('{} stimuli and {} camera triggers'.format(len(inds_stim), len(inds_camTrig)))

    ### Indices of ca images closest to stimulus
    inds_stim_img = spt.nearestMatchingInds(inds_stim, inds_camTrig)

    ### Find trials where the nearest cam trigger is farther than the stimulus by a certain amount
    inds_camTrigNearStim = inds_camTrig[inds_stim_img]
    t_stim = bas['t'][inds_stim]
    t_camTrigNearStim = bas['t'][inds_camTrigNearStim]
    inds_tooFar = np.where(np.abs(t_stim-t_camTrigNearStim)>maxAllowedTimeBetweenStimAndCamTrig)[0]
    inds_ca_all = np.arange(nCaImgs)
    nPreStim = int(time_preStim/dt_ca)
    nPostStim = int(time_postStim/dt_ca)
    print("{} pre-stim points, and {} post-stim points".format(nPreStim, nPostStim))
    inds_ca_trl = np.array(spt.segmentByEvents(inds_ca_all, inds_stim_img+nCaImgs_extra,nPreStim,nPostStim))
    ### Find trials that are too short to include the pre- or post-stimulus period
    trlLens = np.array([len(trl_) for trl_ in inds_ca_trl])
    inds_tooShort = np.where(trlLens < np.max(trlLens))[0]
    inds_trl_del = np.union1d(inds_tooFar, inds_tooShort)
    inds_trl_keep = np.setdiff1d(np.arange(len(inds_ca_trl)),inds_trl_del)

    ### Exclude the above 2 types of trials from consideration
    if len(inds_trl_del)>0:
        print('Excluding the trials {}'.format(inds_trl_del))
        inds_ca_trl = inds_ca_trl[inds_trl_keep]

    I = []
    print('Reading trial-related images from tif files...')
    nTrls = len(inds_ca_trl)
    def trlImages(inds_ca_trl,inds_imgsInTifs,nImgCh,tifInfo,trl):
        trl_ = np.arange(trl.min()*nImgCh, (trl.max()+1)*nImgCh)
        loc = util.locateItemsInSetsOfItems(trl_, inds_imgsInTifs)
        I_ = []
        for subInds, supInd in zip(loc['subInds'], loc['supInds']):
            with tff.TiffFile(tifInfo['filePaths'][supInd]) as tif:
                img = tif.asarray(key = subInds)
            I_.extend(img.reshape(-1,nImgCh,*img.shape[1:]))
        I_ = np.array(I_)
        return I_

    if n_jobs < 2:
        chunkSize = int(nTrls/5)
        for iTrl, trl in enumerate(inds_ca_trl):
            if np.mod(iTrl,chunkSize)==0:
                print('Trl # {}/{}'.format(iTrl+1,nTrls))
            I_ = trlImages(inds_ca_trl,inds_imgsInTifs,nImgCh,tifInfo,trl)
            I.append(I_)
    else:
        print('Processing with dask')
        import dask
        from dask.diagnostics import ProgressBar
        for trl in inds_ca_trl:
            I_ = dask.delayed(trlImages)(inds_ca_trl,inds_imgsInTifs,nImgCh,tifInfo,trl)
            I.append(I_)
        with ProgressBar():
            I = dask.compute(*I)

    D = dict(I = np.squeeze(np.array(I)), tifInfo = tifInfo, inds_stim = inds_stim, inds_stim_img = inds_stim_img,\
             inds_camTrig = inds_camTrig,bas = bas, inds_trl_excluded = inds_trl_del)
    return D

def readProcTableInPath(path):
    from apCode import util
    from apCode.FileTools import findAndSortFilesInDir
    import pandas as pd
    import os
    fn = findAndSortFilesInDir(path)
    ind = util.findStrInList('procTable', fn)
    procTable = None
    if len(ind)>0:
        ind = ind[-1]
        try:
            procTable = pd.read_excel(os.path.join(path,fn[ind]))
        except:
            print('Could not read procTable. Make sure it is not open')
    else:
        print(f'No procTable in {path}')
    return procTable

def readRoiNamesFromPath(path):
    from read_roi import read_roi_zip
    import numpy as np
    import os
    from apCode import util
    from apCode.FileTools import findAndSortFilesInDir
    fn = findAndSortFilesInDir(path, ext = '.zip')
    if len(fn)>0:
        ind_best = np.argmax(util.sequenceMatch('RoiSet', fn))
    zipName = fn[ind_best]
    rois = read_roi_zip(os.path.join(path, zipName))
    roiNames = np.array(list(rois.keys()))
    return roiNames

def readRoiTsFromPath(path):
    from apCode.FileTools import findAndSortFilesInDir
    from scipy.io import loadmat
    import numpy as np
    import os
    fn = findAndSortFilesInDir(path)
    roi_ts = None
    readFlag = False
    if 'roi_ts.mat' in fn:
        try:
            roi_ts = loadmat(os.path.join(path, 'roi_ts.mat'), struct_as_record=False,\
                             squeeze_me=True)['roi_ts']
            readFlag = True
        except:
            print(f'Could not read roi_ts in {path}')
    if not readFlag:
        import h5py
        try:
            fn = findAndSortFilesInDir(path, search_str='procData', ext = 'mat')
            if len(fn)>0:
                fn = fn[-1]
                procData = h5py.File(os.path.join(path, fn), mode = 'r')
                try:
                    roi_ts = np.array(procData['roi_ts'])
                    readFlag = True
                except:
                    pass
                if not  readFlag:
                    try:
                        roi_ts = np.array(procData['roi_ts_den'])
                    except:
                        pass
        except:
            print('Could not read roi_ts from HDF file either')
    return roi_ts

def relevant_from_xls(xls, check_procTable:bool =True):
    import pandas as pd
    import os
    import numpy as np
    from apCode.FileTools import findAndSortFilesInDir
    inds_data = xls.idxData.values
    inds_data = np.unique(np.delete(inds_data, np.where(np.isnan(inds_data))))
    D = []
    for count, iData in enumerate(inds_data):
#        print(f'Data set # {count + 1}/{len(inds_data)}')
        xls_now = xls.loc[xls.idxData == iData]
        xls_now = xls_now.rename(columns = {'chImg': 'camTrigCh', 'Quality': 'quality'})
        path_now = os.path.join(np.unique(xls_now.Path.values)[0].replace('X:','T:'), 'AnalyCont')
        if check_procTable:
            fn = findAndSortFilesInDir(path_now, search_str = 'procTable')
            if len(fn)>0:
                xls_now = xls_now.assign(path = path_now)
                D.append(xls_now)
        else:
            xls_now = xls_now.assign(path = path_now)
            D.append(xls_now)
    if len(D)>0:
        D = pd.concat(D)
    return D

def smoothenSwimEmgs(x, nKer:int = 80, nWaves = None):
    """
    Returns swim EMG (or VR activity) after convolving
    with a semi-gaussian causal kernel of specified length.
    Parameters
    ----------
    x: array, (nSignals,nSamples)
       Array of signals to smooth
    nKer: int
        Kernel length
    nWaves: int or None
        If int, then wavelet denoises signals with nWaves wavelet scales. If None,
        then skips denoising
    Returns
    -------
    y: array, (nSignals, nSamples)
        Smoothed signal array.
    """
    from dask import delayed, compute
    from apCode.SignalProcessingTools import causalConvWithSemiGauss1d
    import numpy as np
    from apCode.spectral.WDen import wden

    if nWaves == None:
        fun = lambda x, nKer:causalConvWithSemiGauss1d(x**2,nKer)**0.5
    else:

        fun = lambda x, nKer: causalConvWithSemiGauss1d(wden(x, n = nWaves)**2, nKer)**0.5

    if np.ndim(x)==1:
        x = x[np.newaxis,:]
    y = np.array(compute(*[delayed(fun)(x_, nKer) for x_ in x], scheduler = 'processes'))
    return np.squeeze(y)

def splitRoisToLR(roiNames, roi_ts):
    import numpy as np
#    from apCode import util
    sidePrefixes = ('L.', 'R.')
    roi_ts_lr = []
    roiNames_lr = []
    for sp in sidePrefixes:
#        inds = util.findStrInList(sp, roiNames, case_sensitive = True)
        inds = [i for i in range(len(roiNames)) if roiNames[i].startswith(sp)]
        roi_ts_lr.append(roi_ts[inds])
        roiNames_lr.append(roiNames[inds])
    roi_ts_lr = np.array(roi_ts_lr)
    return roiNames_lr, roi_ts_lr

def stripRoiNames(roiNames):
    """
    Given a set of roiNames, strips that last part connected by dot and returns
    """
    import numpy as np

    def joinWithDot(iterable):
        a = ''
        for s in iterable:
            a = a + '.' + s
        return a[1:]
    foo = [joinWithDot(rn.split('.')[:-1]) for rn in roiNames]
    return np.array(foo)


def swimParamsFromEmgs_to_dataframe(x, to_zscore:bool = True, smooth_kwargs = dict(nWaves = 2, nKer = 80), ep_kwargs = dict(thr = 1, minOnDur = 30,\
                      minOffDur = 100, use_emd = True), burst_kwargs = dict(minOnDur = 5, minOffDur = 5)):
    """
    Parameters
    ----------
    x: array, (nSignals, nChannels, nSamples)
        nSignals: E.g., number of trials of swim episode containing timeseries
        nChannels: E.g., number of EGM/VR channels (if 2, then typically left and right).
        code assumes that the signals in both the channels indicate the same episode.
        nSamples: Number of time points
    nWaves, nKer: See apCode.tanimotom.besselImaging.smoothenEmgs
    thr, minOnDur, minOffDur, use_emd: See headFixed.swimOnAndOffsets
    """
    from apCode.behavior.headFixed import swimOnAndOffsets
    from pandas import DataFrame as DF
    from apCode.tanimotom.besselImaging import smoothenSwimEmgs
    import numpy as np
    if np.ndim(x) <3:
        raise IOError('Input must be 2D (nTrials, nChannels, nSamples). Expand dimenions accordingly and try again!')
    if to_zscore:
        print('Data units to zscores...')
        from apCode.SignalProcessingTools import zscore
        x = zscore(x,axis = -1)

    print('Denoising and smoothing...')
    x_smooth = []
    for iCh in range(x.shape[1]):
        ch_now = x[:,iCh,:]
        x_smooth.append(smoothenSwimEmgs(ch_now, **smooth_kwargs))
    x_smooth = np.swapaxes(np.array(x_smooth),0,1)
    x_ep = np.max(x_smooth, axis = 1)
    dic = {'trialIdx':[], 'episodeDur':[], 'episodeIdx':[], 'episodeOnsetIdx':[], 'nEpisodes':[], 'chIdx':[],\
          'burstIdx':[], 'burstAmp':[], 'burstDur':[], 'nBursts':[]}
    for iTrl, x_ in enumerate(x_ep):
        onOffs  =  swimOnAndOffsets(x_, **ep_kwargs)
        if not np.any(onOffs == None):
            try:
                for iEp, onOff in enumerate(onOffs):
                    ep = x_smooth[iTrl][:,onOff[0]:onOff[1]]
                    for iCh, ch in enumerate(ep):
                        boofs = swimOnAndOffsets(ch, ep_kwargs['thr'], use_emd = False, **burst_kwargs)
                        if not np.any(boofs ==None):
                            for iBurst, boof in enumerate(boofs):
                                burst = ch[boof[0]:boof[1]]
                                dic['burstIdx'].append(iBurst)
                                dic['burstDur'].append(boof[1]-boof[0])
                                dic['burstAmp'].append(burst.max())
                                dic['chIdx'].append(iCh)
                                dic['nBursts'].append(len(boofs))
                                dic['episodeIdx'].append(iEp)
                                dic['nEpisodes'].append(len(onOffs))
                                dic['episodeOnsetIdx'].append(onOff[0])
                                dic['episodeDur'].append(onOff[1]-onOff[0])
                                dic['trialIdx'].append(iTrl)
            except:
                print(f'Code failed for {iTrl+1}/{len(x_ep)}')
        else:
            print(f'No episode in {iTrl+1}/{len(x_ep)}')

    return DF(dic)



