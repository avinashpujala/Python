# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:36:14 2018

@author: pujalaa
"""

import numpy as np
import dask, os, sys
import h5py
sys.path.append(r'v:/code/python/code')
import apCode.volTools as volt
from apCode import util
import apCode.FileTools as ft

def cleanTailAngles(ta,svd = None, nComps:int = 3, nWaves = 5, dt = 1/1000, lpf = 60):
    """
    Given the array of tail angles along the fish across time, and optionally,
    an svd object (Truncated Singular Value Decomposition from sklearn) fit across
    presumably multiple fish and trials, returns a clean minimal version of the
    tail angles first fit with the minimal components svd object, which is then 
    wavelet denoised and low pass filtered before the original tail angles array
    is reconstructed.
    Parameters
    ----------
    ta: array, (num_points_along_fish, num_time_points)
        Tail angles array
    svd: object
        Truncated SVD object from sklearn generated with "n_comps" and fit
        to a larger dataset. If None, computes using the provided dataset
    nComps: scalar
        Number of SVD components. Typically, 3 suffice.
    nWaves: int or None
        Number of wavelet levels to use for denoising. If None (default), automatically
        estimates
    dt: scalar
        Sampling interval for the data.
    lpf: scalar
        Low pass filter
    Returns
    -------
    ta_clean: array, (num_points_along_fish, num_time_points)
        Cleaned tail angles array
    ta_svd: array, (nComps, num_time_points)
        Timeseries of the 'weights' of the 3 SVD components
    svd: object
        SVD object. If None was provided as input returns the fit object, else returns
        as is.
    """
    from apCode.SignalProcessingTools import chebFilt
    from apCode.spectral.WDen import wden
    if svd == None:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=nComps, random_state = 143).fit(ta.T)
    ta_svd = svd.transform(ta.T).T
        
    ta_svd = np.apply_along_axis(chebFilt,1, np.apply_along_axis(wden,1,ta_svd, n= nWaves),\
                                 dt,lpf, btype = 'lowpass')
    ta_clean =  svd.inverse_transform(ta_svd.T).T
    return ta_clean, ta_svd, svd

def comBoot(h_trl,t_trl, target = 25):
    """
    Parameters
    ----------
    h_trl, t_trl: array, (nRois, nTrls, nTimePoints)
    target: int
        Number of target trials
    Returns
    -------
    h_trl_boot, t_trl_boot: array, (nRois, nTrls_boot, nTimePoints), where nTrls_boot>= target
    """
    h_trl, t_trl = np.transpose(h_trl,(1,0,2)), np.transpose(t_trl,(1,0,2))
    nTrls_head = h_trl.shape[0]
    nTrls_tail = t_trl.shape[0]
    nTrls = np.min([nTrls_head, nTrls_tail])
    h_trl = h_trl[:nTrls]
    t_trl = t_trl[:nTrls]    
    count = 0
    while nTrls <25:
        count = count + 1
        comb = util.CombineItems().fit(h_trl)
        t_trl = np.concatenate((t_trl,comb.transform(t_trl)[1]),axis = 0)        
        h_trl = np.concatenate((h_trl,comb.transform(h_trl)[1]),axis = 0)  
        nTrls = h_trl.shape[0]
        print(f'Iter # {count}')
    if nTrls > target:
        h_trl, t_trl = h_trl[:target], t_trl[:target]
    h_trl = np.transpose(h_trl,(1,0,2))
    t_trl = np.transpose(t_trl,(1,0,2))
    return h_trl, t_trl
    
def copyFishImgsForNNTraining(exptDir, prefFrameRangeInTrl = (115,160),\
                              nImgsForTraining:int = 50, overWrite:bool = False):
    """ A convenient function for randomly selecting fish images from within a range
    of frames (typically, peri-stimulus to maximize postural diversity) in each trial
    directory of images, and then writing those images in a directory labeled "imgs_train"
    Parameters
    ----------
    exptDir: string
        Path to the root directory where all the fish images (in subfolders whose names
        end with "behav" by my convention) from a particular experimental are located.
    prefFrameRangeInTrl: 2-tuple
        Preferred range of frames for selecting images from within a trial directory.
        Typically, peri-stimulus range.
    nImgsForTraining: integer
        Number of images to select and copy
    overWrite:bool
        If True, overwites existing directory and saves images afresh.
    Returns
    -------
    selectedFiles: list, (nImgsForTraining,)
        List of paths from whence the selected images
    dst: string
        Path to directory of stored images
    """    
    import apCode.FileTools as ft
    import shutil as sh
    from apCode.util import timestamp
    
    inds_sel = np.arange(*prefFrameRangeInTrl)
    behavDirs = [x[0] for x in os.walk(exptDir) if x[0].endswith('behav')]
    trlDirs = []
    for bd in behavDirs:
        trlDirs.extend(os.path.join(bd, td) for td in ft.subDirsInDir(bd))
    join = lambda p, x: [os.path.join(p,_) for _ in x]
    files_sel =[]
    for td in trlDirs:
        filesInDir = ft.findAndSortFilesInDir(td,ext = 'bmp')
        if len(filesInDir)> np.max(inds_sel):            
            files_sel.extend(join(td, filesInDir[inds_sel]))
        else:
            print(f'{len(filesInDir)} images in {td}, skipped')
    files_sel = np.random.choice(np.array(files_sel), size = nImgsForTraining, replace = False)
    ext = files_sel[0].split('.')[-1]
    rename = lambda path_, n_, ext: os.path.join(path_, 'img_{:05}.{}'.format(n_, ext))    
    dst = os.path.join(exptDir,f'imgs_train_{timestamp("min")}')
    dsts = [dask.delayed(rename)(dst,n,ext) for n, path_ in enumerate(files_sel)]
    if not os.path.exists(dst):
        os.mkdir(dst)
        dask.compute([dask.delayed(sh.copy)(src,dst_) for src, dst_ in zip(files_sel,dsts)],\
                      scheduler = 'threads')
        np.save(os.path.join(dst,f'sourceImagePaths_{timestamp("min")}.npy'), files_sel)
    else:
        if overWrite:
            from shutil import rmtree
            rmtree(dst)
            os.mkdir(dst)
            dask.compute([dask.delayed(sh.copy)(src,dst_) for src, dst_ in zip(files_sel,dsts)],
                         scheduler = 'threads')
            np.save(os.path.join(dst,f'sourceImagePaths_{timestamp("min")}.npy'), files_sel)
        else:
            print(f'{dst} already exists! Delete directory to re-select images or set overWrite = True')
    return files_sel, dst 

def deletion_inds_for_stimLoc(stimLoc_behav, stimLoc_ca):
    """ 
    Looks at mismatch in stimulus location vectors (eg., ['h', 'h','h', 't','t'])
    by examining continuous blocks of stimulus of a type an returns indices for
    deletion in each stimulus location vector so as to produce a match. If for
    each continuous block, either the behavior or ca block is longer then assumes
    that the extra entries to be deleated will be at the end, becuase these mismatches
    are typically because the data acquisition software can crash (usually ScanImage)
    or be interrupted.
    Parameters
    ----------
    stimLoc_behav: array-like, (N,)
        Vector of strings indicating stimulus locations ('h' or 't') obtained
        for behavior trials
    stimLoca_ca: array-like, (M,)
        " " "... for ca trials
    Returns
    -------
    inds_del_behav, inds_del_ca: array-like, (n,), (m,). If originally, stimLoc_behav,
        and stimLoc_ca were of the same length, then these will be empty vectors. If,
        one of these vectors is longer than the other by say 2 entries (for example), 
        then inds_del_behav and inds_del_ca can each be of length 0,1, oe 2, but 
        len(inds_del_behav) + len(inds_del_ca) = 2.
    """
    inds_behav = util.getBlocksOfRepeats(stimLoc_behav)[1]
    inds_ca = util.getBlocksOfRepeats(stimLoc_ca)[1]
    inds_del_ca, inds_del_behav = [],[]
    count_ca, count_behav = 0,0
    for ic, ib in zip(inds_ca, inds_behav):
        d = len(ic)-len(ib)
        if d>0:
            foo = np.arange(len(ic)-d,len(ic))
            inds_del_ca.append(count_ca+foo)
        elif d<0:
            foo = np.arange(len(ib)+d,len(ib))
            inds_del_behav.append(count_behav+foo)
        count_ca += len(ic)
        count_behav += len(ib)
    return np.array(inds_del_behav), np.array(inds_del_ca)

def estimateHeadSideInFixed(img):
    """
    A very quick and dirty way of estimating a location close to the head in image
    of fish's tail only (i.e. no head)
    Parameters
    ----------
    img: array, (M,N)
        Image with tail, but not head pixels. Assumes the fish is fixed.
    Returns
    -------
    out: 2 tuple
        x,y coordinates of head side.
        This is useful for sorting midline indices    
    """
    img_max = np.multiply(np.c_[np.sum(img,axis = 1)], 
                          np.r_[np.sum(img, axis = 0)])
    y,x = np.unravel_index(np.argmax(img_max), np.shape(img_max))
    return np.array([x,y])

def extractAndStoreBehaviorData_singleFish(fishPath, uNet = None, hFilePath=None,\
                                           regex=r'\d{1,5}_[ht]', imgExt='bmp'):
    """
    Parameters
    ----------
    fishPath: str
        Path to a fish data.
    uNet: U-net object (HDF file with stored parameters) or None
        If None, then searches for U-net in the fish path by looking for
        files with ".h5" extension and name containing with "trainedU".
    hFilePath: str or None
        Full path to HDF file where data is to be stored. If None, looks for an
        existing HDF file in fishPath. Assumes the file will have ".h5" extension
        and name that contains "procData".
    regex: str
        Regular expression for finding head and tail stimulation folders. For e.g.,
        "001_h", "002_t", etc.
    imgExt: str
        Extension of the behavior image files.
    Returns
    -------
    hFilePath: Full path to the HDF file object where behavior data is stored under 
        the group name "/behav".
        The following datasets can be found in the group.
        "/behav/images_prob": array, (nTimePoints, nRows, nCols)
            Probability images generated by the U-net
        "/behav/midlines_interp": array, (nTimePoints, nPointsAlongFish, 2)
            Interpolated midlines.
        "/behav/tailAngles": array, (nTimePoints, nPointsAlongFish, 2)
            Tangent angles (cumulative curvatures) along the length of the fish.    
    """
    import re    
    from apCode.machineLearning import ml as mlearn
    import apCode.behavior.headFixed as hf
    from apCode import hdf
    subDirs = [os.path.join(fishPath, sd) for sd in ft.subDirsInDir(fishPath) 
               if re.match(regex, sd)]
#    nDirs = len(subDirs)
    if np.any(hFilePath == None):
        fn = ft.findAndSortFilesInDir(fishPath,ext = 'h5', search_str='procData')
        if len(fn)>0:
            fn = fn[-1]
            hFileName = fn
        else:
            hFileName = f'procData_{util.timestamp()}.h5'
        hFilePath = os.path.join(fishPath, hFileName)
    if np.any(uNet == None):
        uNet = mlearn.loadPreTrainedUnet(None, search_dir=fishPath)
    stimLoc = []
#    expr = '_[h|t]$'    
    with h5py.File(hFilePath, mode = 'a') as hFile:
        for iSub, sd in enumerate(subDirs):
            sd_now = os.path.split(sd)[-1]
#            hort = re.findall(expr, sd_now)[0][-1]
            roots, dirs, files = zip(*[out for out in os.walk(sd)])
            inds = util.findStrInList('Autosave', roots)
            behavDirs = np.array(roots)[inds]               
            nTrls = len(inds)
            print(f'{nTrls} behavior directories found in {sd}')
#            stimLoc.extend([hort]*nTrls)
            stimLoc.extend([sd_now]*nTrls)
            for iTrl, bd in enumerate(behavDirs):
                print(f'Trl # {iTrl+1}/{nTrls}')
                out = hf.tailAnglesFromRawImagesUsingUnet(behavDirs[iTrl], uNet)
                if (iSub==0) & (iTrl == 0) & ('behav' in hFile):
                    del hFile['behav']                    
                keyName = 'behav/images_prob'
                hFile = hdf.createOrAppendToHdf(hFile,keyName,out['I_prob'])
                keyName = 'behav/tailAngles'
                hFile = hdf.createOrAppendToHdf(hFile,keyName,out['tailAngles'])
                keyName = 'behav/midlines_interp'
                hFile = hdf.createOrAppendToHdf(hFile,keyName, out['midlines']['interp'])     
        hFile.create_dataset('behav/stimLoc', data = util.to_ascii(stimLoc))                                    
    return hFilePath      

def fetchAllFishDataFromXls(xlsPath, dt_behav = 1/1000, lpf = 60, nWaves = 3, recompute_dff = True):
    from apCode.FileTools import openPickleFile, findAndSortFilesInDir
    import pandas as pd
    def to_dataframe(data):    
        D = []
        for hort in data.tailAngles_flt_den.keys():
            ta, dff = data.tailAngles_flt_den[hort], data.dff_trl[hort]
            ta_tot = ta[:,-1,:]
            behav = dict(trial = np.arange(ta.shape[0]), tailAngles = list(ta),tailAngles_total = list(ta_tot))
            behav = pd.DataFrame(behav)
            trls_= np.tile(np.arange(dff.shape[0]).reshape(-1,1),(dff.shape[1],1)).flatten()
            dff_ = np.concatenate(np.transpose(dff,(1,0,2)),axis = 0)
            roiNames_ = np.tile(data.roiNames[:,np.newaxis], (1,dff.shape[0])).flatten(order= 'C')
            roiIndex = np.tile(np.arange(len(data.roiNames))[:,np.newaxis],(1,dff.shape[0])).flatten(order = 'C')
            ca = pd.DataFrame(dict(trial = list(trls_),roiIndex = roiIndex, \
                                   roiName = list(roiNames_), ca_dff = list(dff_)))
            df = pd.merge(ca,behav, how = 'outer')
            df = df.assign(stimulus = hort)
            df = df.assign(fishNum = data.fishId)
            df = df.assign(path_hdf = data.path_hdf)
            df = df.assign(path_midlines = data.path_midlines)
#            df = df.assign(path_unet = data.path_unet)
            D.append(df)
        return pd.concat(D, ignore_index = True)    
    xls = pd.read_excel(xlsPath)
    xls = xls.loc[xls.Processed == 1]
    D = []
    for iFish in range(xls.shape[0]):
        xls_ = xls.iloc[iFish]
        print(f'Reading fish =  {iFish+1}/{xls.shape[0]} from {xls_.Date}-{xls_.FishID}...')
        f = findAndSortFilesInDir(xls_.Path, ext = 'pickle', search_str = 'data')        
        if len(f)>0:
            print(f'Reading from pickle file...')
            pathToFile = os.path.join(xls_.Path, f[-1])
            data = openPickleFile(pathToFile)
            if not hasattr(data, 'tailAngles_flt_den'):
                data = singleFishDataFromXls(xlsPath, xls_.Date, xls_.FishID, sessionId=xls_.SessionID,\
                                        dt_behav= dt_behav, lpf = lpf, nWaves= nWaves, recompute_dff = recompute_dff)
        else:
            print('Recomputing some data parameters...')
            data = singleFishDataFromXls(xlsPath, xls_.Date, xls_.FishID, sessionId=xls_.SessionID,\
                                    dt_behav= dt_behav, lpf = lpf, nWaves= nWaves)    
        df = to_dataframe(data)
        df = df.assign(fishID = iFish, exptDate = xls_.Date)
        D.append(df)
    return pd.concat(D)   

class FishData(object):    
    def __init__(self, xlsPath, exptDate = None, fishId = None, sessionId = 1):
        from apCode.util import to_ascii, to_utf
        self.to_ascii = to_ascii
        self.to_utf = to_utf
        self.preprocess_for_midline = fishImgsForMidline
        self.xlsPath = xlsPath
        if exptDate == None:
            exptDate = int(input('Enter expt date :'))
        if fishId == None:
            fishId = int(input('Enter fish ID (within experimental session): '))        
        self.exptDate = exptDate
        self.fishId = fishId
        self.sessionId = sessionId        
    
    def assess_unet(self, path_to_unet = None, nImgs:int = 50, 
                    intraTrlFrameRange = (115,160),verbose = 1,\
                    single_blob:bool = False, plot:bool = True):
        """ Randomly pulls specified number of images from within specified range (see
        fetch_fish_images), predict on them using the stored unet (self.unet) and return
        a dictionary holding the raw images and their probability maps yielded by the u-net.
        Parameters
        ----------
        path_to_unet: str
            Full path to the stored unet file (.h5, hdf). If None, uses the stored unet or
            looks for u-net file within self.path_session (session directory)
        nImgs, intraTrlFrameRange: see fetch_fish_imgs
        single_blob: bool
            If true, then applies preprocess_for_midlines to the images to identify
            filter for fish blobs on the assumption that a single fish exists in each image
        Returns
        -------
        out: dict
            Dictionary containing raw images, probability images, and fish images resulting from
            the application of preprocess_for_midlines on probability images
        """
        print('Reading images...')
        images = selectFishImgsForUTest(self.path_session, nImgsForTraining = nImgs,
                                        prefFrameRangeInTrl = intraTrlFrameRange)
        if not hasattr(self, 'unet'):
            self = self.load_unet(path_to_unet = path_to_unet)
        
        print('Predicting on images...')
        images_prob = self.predict_on_images(images, verbose = verbose)
        
        out = dict(images = images, images_prob = images_prob)
        if single_blob:
            out['images_fish']= self.preprocess_for_midline(images_prob)
        if plot:
            self.plot_montage(out['images'], out['images_prob'])                                     
        return out
    
    def copy_imgs_for_training_unet(self,nImgs:int = 30, intraTrlFrameRange = (115,160)):
        """
        Copies specified number of images by randomly pulling from within specified
        frame range within each trial directory of behavior images
        Parameters
        -----------
        See help for fetch_fish_images
        Returns
        -------
        path_images: str
            Path to directory where images where copied
        """        
        path_images = copyFishImgsForNNTraining(self.path_session,
                                                 prefFrameRangeInTrl=intraTrlFrameRange,
                                                 nImgsForTraining=nImgs)[1]
        self.path_images_train = path_images
        return path_images        
    
    def correct_tailAngles(self, ds_ratio = (0.5,0.25),**kwargs_griddata):
        """
        Corrects tail angles for slightly bent fish shapes and for noisy fluctuations
        along the spatial and temporal dimensions
        Parameters
        ----------
        see apCode.SignalProcessingTools.interp.downsample_gradients_and_interp2d
        **kwargs_griddata: Keyword arguments for scipy.interpolate.griddata
        Returns
        -------
        self
        """
        import numpy as np
        from dask import delayed, compute
        from apCode.SignalProcessingTools import interp
        if not hasattr(self, 'tailAngles'):
            print('Tail angles not found')
        else:
            ta = {}
            for hort in self.tailAngles.keys():
                ta[hort] = []
                for ta_trl in self.tailAngles[hort]:
                    ta_trl = ta_trl - np.expand_dims(np.median(ta_trl,axis = 1),axis = 1)
                    ta_trl = delayed(interp.downsample_gradients_and_interp2d)(ta_trl, \
                                    ds_ratio = ds_ratio,**kwargs_griddata)
                    ta[hort].append(ta_trl)
                ta[hort]= np.array(compute(*ta[hort]))
            self.tailAngles_corr = ta
        return self          
    
    def extract_and_store_tailAngles(self, filtSize = 2.5, otsuMult = 1, 
                                     smooth:int = 20, n:int = 50):
        """
        Extract midlines from all behavior images in all trial directories located within
        the fish directory
        """
        import numpy as np
        import apCode.FileTools as ft
        import os
        import re as regex
        from apCode.util import timestamp
        midlines = dict(h = [], t = [])
        tailAngles = dict(h = [], t = [])
        with self.open_hdf() as hFile:
            iTrl_h, iTrl_t = -1,-1
            for iSub,sd in enumerate(self.paths_ht):
                sd_ = os.path.split(sd)[-1]
                print(f'Processing from {sd}, folder {iSub+1}/{len(self.paths_ht)}')
                htStr = regex.findall(r'h|t', sd_)[0]        
                dir_behav = os.path.join(sd,'behav')
                trlDirs = ft.subDirsInDir(dir_behav)
                excludedTrls = np.squeeze(hFile['excludedTrls'][sd_][()])
                print(f'{len(trlDirs)} behavior trials in directory, {np.size(excludedTrls)} being excluded')
                trlDirs = np.delete(trlDirs,excludedTrls)
                print('Extracting midlines...')                
                for iTrl, td in enumerate(trlDirs):
                    if htStr == 'h':
                        iTrl_h += 1
                    elif htStr == 't':
                        iTrl_t += 1
                    print('Trl # {}/{}'.format(iTrl+1, len(trlDirs)))
                    print('Reading images...')
                    images = volt.img.readImagesInDir(os.path.join(dir_behav, td))
                    if images.size > 0:
                        ml = self.raw_images_to_midlines(images, filtSize= filtSize, 
                                                         otsuMult=otsuMult, smooth=smooth)
                        ta = self.tangent_angles_along_midlines(ml,n = n)
                        midlines[htStr].append(ml)
                        tailAngles[htStr].append(ta)
                        if iTrl == 0:
                            nImgsInTrl_behav = len(ft.findAndSortFilesInDir(os.path.join(dir_behav,td), ext ='bmp'))
                            if not 'nImgsInTrl_behav' in hFile:
                                hFile.create_dataset('nImgsInTrl_behav', data = nImgsInTrl_behav)
                            else:
                                hFile['nImgsInTrl_behav'][()] = nImgsInTrl_behav
                        key = f'{htStr}/tailAngles'
                        if (htStr == 'h') & (iTrl_h == 0) & (key in hFile):
                            del hFile[key]
                            print(f'Deleted {key} for iTrl_h = {iTrl_h}')
                        if (htStr == 't') & (iTrl_t == 0) & (key in hFile):
                            del hFile[key]
                            print(f'Deleted {key} for iTrl_t = {iTrl_t}')
                        ta = ta[np.newaxis,...] # To write in easily accessible trial format
                        if not key in hFile:
                            print(f'Creating {key} in hFile')
                            hFile.create_dataset(key, data = ta, maxshape = (None, *ta.shape[1:]))
                        else:
                            print(f'Appending to {key} in hFile')
                            hFile[key].resize((hFile[key].shape[0] + ta.shape[0]),axis = 0)
                            hFile[key][-ta.shape[0]:] = ta
                    else:
                        print(f'No images found in {td}, skipping!')
        print('Converting tail angles to arrays...')                
        for key in tailAngles.keys():
            tailAngles[key] = np.asarray(tailAngles[key])     
        self.midlines = midlines
        self.tailAngles= tailAngles
        self.nImgsInTrl_behav = len(images)
        path_midlines = os.path.join(self.path_fish, f'midlines_{timestamp("min")}.npy')
        np.save(path_midlines, midlines, allow_pickle = True)
        self.path_midlines = path_midlines
        return self
        
    def fetch(self, q = 20, updateHdf = False, overwrite_roi_ts = True, recompute_dff = False):
        """
        Fetch basic processed data from hdf file
        Parameters
        ----------
        *args:
        **kwargs:
        q: scalar
            Bottom percentile for computing ratiometric \delta F/F
        """ 
        import pandas as pd
        import apCode.behavior.headFixed as hf        
        from apCode import hdf     
        try:
            xls = pd.read_excel(self.xlsPath)
        except:
            raise IOError('Cannot read excel sheet, check path!')            
        iFish = np.where((xls.Date == self.exptDate) & (xls.FishID == self.fishId) &
                         (xls.SessionID == self.sessionId))[0]
        if len(iFish)>0:
            iFish = iFish[0]
            path_fish = xls.Path[iFish]
            self.path_fish = path_fish
            file_hdf = ft.findAndSortFilesInDir(path_fish, ext = 'h5', search_str = 'procData')
            if len(file_hdf)>0:
                file_hdf = file_hdf[-1]
                path_hdf = os.path.join(path_fish, file_hdf)
                self.path_hdf = path_hdf
                with h5py.File(path_hdf, mode = 'r+') as hFile:
                    if 'nImgsInTrl':
                        self.nImgsInTrl = np.array(hFile['nImgsInTrl'])
                    if 'img_avg' in hFile:
                        self.img_avg = np.array(hFile['img_avg'])
                    if ('roi_ts' in hFile) & (recompute_dff == False):
                        print('Copying roi timeseries...')
                        roi_ts = hdf.recursively_load_dict_contents_from_group(hFile, '/roi_ts')
                        if overwrite_roi_ts:
                            print('Overwriting roi timeseries in hdf file...')
                            del hFile['roi_ts']
                            if 'roiNames' in roi_ts.keys():
                                del roi_ts['roiNames']
                            hdf.recursively_save_dict_contents_to_group(hFile,'/roi_ts/',roi_ts)
                    else:
                        if 'rois_cell' in hFile:
                            print('Recomputing df/f...')
                            roi_ts = hf.roiTs2Dff(hf.roiTimeseriesAsMat(hFile), q = q)
                            if updateHdf:
                                print('Saving roi timeseries to hdf file...')
                                if 'roiNames' in roi_ts.keys():
                                    del roi_ts['roiNames']
                                hdf.recursively_save_dict_contents_to_group(hFile,'/roi_ts/',roi_ts)
                    roiNames = self.to_utf(roi_ts['roiNames_ascii'])
                    self.roiNames = roiNames
                    self.roi_ts = roi_ts                    
                    self.paths_ht = self.get_head_tail_paths()
                    self.path_session = os.path.split(self.path_fish)[0]
            else:
                print('No HDF file found')
        else:
            print('No matching fish found in excel file. Check Date (expt), FishID, SessionID')        
        return self
    
    def fetch_fish_images(self, nImgs:int = 30, intraTrlFrameRange = (115,160)):
        """ 
        Fetches the specified number of behavior images at random by pulling
        from within the specified frame range (typically, peri-stimulus range to maximize
        postural diversity) within each behavior trial directory
        Parameters
        ----------
        nImgs: int
            Number of images to fetch
        intraTrlFrameRange: 2-tuple or list
            Starting and ending value of the frame range within in each trial directory 
            from whence to fetch
        Returns
        -------
        images: array, ([nImgs,], M, N)
            Fetched images
        """        
        images = selectFishImgsForUTest(self.path_session, nImgsForTraining = nImgs,
                                        prefFrameRangeInTrl = intraTrlFrameRange)
        return images
    
    def filter_denoise_tailAngles(self, dt = 1/1000,
                                  lpf = 100, nWaves = None):
        """
        Lowpass filters and or denoises tailAngles
        Parameters
        ----------
        tailAngles: array, (nTrls, nPointsAlongFish, nTimePointsInTrl)
            Collection of tail angles to filter and/or wavelet denoise
        dt: scalar
            Sampling interval (used for low-pass filtering)
        lpf: scalar or None
            Lowpass value. If None then skips lowpass filtering
        nWaves: int
            Parameter "n" in wavelet denoising (wden); wavelet scales.
            If None, skips.
        Returns
        -------
        tailAngles_new: array, same shape as tailAngles
            Filtered and/or denoised tail angles array.
        """
        from apCode.SignalProcessingTools import chebFilt
        from apCode.spectral.WDen import wden
        if hasattr(self, 'tailAngles_corr'):
            tailAngles = self.tailAngles_corr
        else:
            tailAngles = self.tailAngles
        tailAngles_flt_den = {}
        for hort in tailAngles.keys():
            ta = tailAngles[hort]
            dims = ta.shape
            if np.ndim(ta) == 3:
                ta = ta.reshape(-1, *dims[2:])
            if not nWaves == None:
                ta = np.asarray(dask.compute(*[dask.delayed(wden)(ta_,n = nWaves)\
                                               for ta_ in ta]))
            if not lpf == None:            
                ta = dask.compute(*[dask.delayed(chebFilt)(ta_,dt,lpf, btype = 'lowpass') for ta_ in ta])
            ta = np.asarray(ta)        
            ta = ta.reshape(dims)
            tailAngles_flt_den[hort]= ta
        self.tailAngles_flt_den = tailAngles_flt_den
        self.tailAngles_lpf = lpf
        self.tailAngles_nWaves = nWaves
        return self
        
    def get_head_tail_paths(self, re_ht = r'\d+_h|\d+_t'):
        """Return a list of all head and tail trial subdirectories
        Parameters
        ----------
        re_ht: Regex string pattern 
            This is the regex string pattern to match when searching
            for the names of the head and tail stimulation subdirectories within the
            fish directory
        Returns
        -------
        paths_ht: list-like
            Collection of paths to the head and tail stimulation trial directories within
            the fish directory
        """
        import re
        import apCode.FileTools as ft
        import numpy as np
        import os
                
        paths_ht  = np.array([os.path.join(self.path_fish, sd) for sd in ft.subDirsInDir(self.path_fish) if re.match(re_ht,sd)!=None])
        
        self.paths_ht = paths_ht
        self.re_ht = re_ht
        return paths_ht
    
    def load_unet(self, path_to_unet = None, search_dir = None, name_prefix = 'trainedU'):
        """
        Automatically searches for save uNet file in relevant path or specified path (overrides)
        and returns the loaded unet model object
        Parameters
        ----------
        path_to_unet: string or None
            If specified, returns the unet at the end of this path. If None, searches of unet file
            in search_dir.
        search_dir: string or None
            If specified ant path_to_unet == None, searches for unet file in this directory
        name_prefix: string
            The string used in filtering files that are possibly the unet files
        Returns
        -------
        unet: model object
            The pre-trained unet model object
        """
        from apCode.machineLearning.unet import model
        import apCode.FileTools as ft
        if not path_to_unet == None:
            print('Loading u net...')
            self.unet = model.load_model(self.path_unet, 
                                         custom_objects=dict(dice_coef = model.dice_coef))
            self.path_unet = path_to_unet
        else:
            if search_dir ==None:
                search_dir = self.path_session
            file_u = ft.findAndSortFilesInDir(search_dir, ext = 'h5', search_str = name_prefix)
            if len(file_u)>0:
                file_u = file_u[-1]
                self.path_unet = os.path.join(search_dir,file_u)
                print('Loading unet...')
                self.unet = model.load_model(self.path_unet,
                                             custom_objects=dict(dice_coef = model.dice_coef))                
            else:
                print('No uNet found in search path, explicitly specify path')            
        return self
            
    def open_hdf(self, mode = 'a'):
        """Returns opened hdf file associated with data"""        
        if hasattr(self,'path_hdf'):
            hFile = h5py.File(self.path_hdf, mode = mode)
        else:
            print('No path to associated hFile found')
            hFile = None
        return hFile
    
    def plot_midlines(self, midlines, shift_x = 1,shift_y = 10, nPre:int = 100, 
                      figSize = (20,5), x = None, alpha = 0.2):
        """ Convenience function for plotting a set of midlines next to each other
        in a way that makes it easy to visually assess midline tracking or movement
        Parameters
        ---------
        midlines: List of len (T,), or array of shape (T, K, 2)
            Midlines to plot
        shift_x, shift_y: scalars
            Horizontal and vertical shifts when plotting successive midlines. 
            Note: Shift along vertical axis is sinusoidal and shift_y is the amplitude of this.
        nPre: int
            Number of pre-stimulus points. Used to plot midline coincident with stimulus
            in red, while all others are plotted in black
        figSize: tuple
            Figure size
        x: array, (T,)
            Time axis; Not yet implemented
        alpha: scalar
            Alpha value for plotted midlines
        Returns
        -------
        fh: Figure handle
            Handle to the plotted figure
        """
        import matplotlib.pyplot as plt
#        from dask import delayed, compute
#        getLens = lambda ml: np.sum((np.gradient(ml)[0])**2,axis = 1)**0.5
#        lMids = compute(*[delayed(getLens)(ml) for ml in midlines])
        yShifts = shift_y*np.sin(np.linspace(0,2*np.pi,len(midlines)))
        if np.any(x==None):
            x = np.arange(len(midlines))
        fh = plt.figure(figsize = figSize)
        for count, _ in enumerate(midlines):
            _ = np.array(_)
            _[:,0] = _[:,0] + (shift_x*count)
            _[:,1] = _[:,1] + yShifts[count]
            if count == nPre:
                plt.plot(*_.T,'r')
            else:
                plt.plot(*_.T,'k', alpha = alpha)
        plt.gca().invert_yaxis()
        return fh
    
    def plot_montage(self, images, images_prob):
        from skimage.util import montage
        import matplotlib.pyplot as plt
        nCols = 4
        if len(images)/nCols == 0:
            nRows = len(images)//nCols
        else:
            nRows = len(images)//nCols + 1
        ar = 0.5*images[0].shape[0]/images[0].shape[1]
        figLen = int(20*ar*nRows/nCols)
        m = [montage((img, img_prob), grid_shape=(1,2), rescale_intensity=True)\
                      for img, img_prob in zip(images, images_prob)]
        m = montage(m, grid_shape=(nRows,nCols))
        plt.figure(figsize=(20, figLen))
        plt.imshow(m)
        
    def predict_on_images(self, images, verbose:int = 0):
        """
        Retuns probability images resulting predicting on raw images using stored u-net.
        Parameters
        ----------
        images: array, ([T,], M, N)
        verbose: 0,1 or 2
            Verbosity while predicting, see unet.predict
        Returns
        -------
        images_prob: array, ([T,], M, N)
            Probability images yielded from u-net prediction
        """
        from apCode.behavior.FreeSwimBehavior import prepareForUnet_1ch
        from apCode.volTools import img as img_
        print('Predicting ...')            
        images_prob = np.squeeze(self.unet.predict(prepareForUnet_1ch(images,sz = self.unet.input_shape[1:3]), verbose=verbose))
        images_prob = img_.resize(images_prob, images.shape[1:], 
                                  preserve_dtype = True, preserve_range = True)        
        return images_prob
    
    def raw_images_to_midlines(self, images, filtSize = 2.5, otsuMult =1, smooth:int = 20,
                               verbose:int = 0):
        """
        Parameters
        -----------
        images: array-like, ([T,], M, N)
            Raw images to extract fish midlines from
        filtSize: Scalar
            Size of gaussian filter used to smooth images during processing
        otsuMult: scalar
            Multiplication factor by which to alter the otsu threshold during segmenting
        smooth: int
            The amount by which to smooth the midlines
        Returns
        -------
        midlines: list
            List of midlines of varyingn lengths
        """
        from apCode.behavior.FreeSwimBehavior import track
        images_pred = self.predict_on_images(images, verbose = verbose)
        images_fish = self.preprocess_for_midline(images_pred, 
                                                  filtSize= filtSize, otsuMult = otsuMult)
        midlines = track.midlinesFromImages(images_fish)[0]
        return midlines
    
    def read_tail_angles(self):
        """
        Read stored tail angles from associated hdf file
        Parameters
        ----------
        dt, lpf, nWaves: see self.filter_denoise_tailAngles
        Returns
        -------
        self with attribute "tailAngles", which is a dictionary
            with keys "h" and/or "t", each of which is an array of
            shape (nTrls, nPointsAlongFish, nPointsInTrl)
        """
        print('Reading tail angles from associated HDF file...')
        tailAngles = {}
        with self.open_hdf() as hFile:
            if 'h/tailAngles' in hFile:
                tailAngles['h'] = np.array(hFile['h/tailAngles'])                
            if 't/tailAngles' in hFile:
                tailAngles['t'] = np.array(hFile['t/tailAngles'])                
        self.tailAngles = tailAngles
        return self
            
    def read_midlines(self):
        """ Read raw midlines from saved .npy file
        Returns
        -------
        self
        """
        import numpy as np
        import apCode.FileTools as ft
        import os
        print('Reading midlines...')
        if hasattr(self, 'path_midlines'):
            midlines = np.load(self.path_midlines, allow_pickle = True)[()]
            self.midlines = midlines
        else:
            file_midlines = ft.findAndSortFilesInDir(self.path_fish, search_str = 'midlines')
            if len(file_midlines)>0:
                file_midlines = file_midlines[-1]
                path_midlines = os.path.join(self.path_fish, file_midlines)
                print(f'... from {path_midlines}')
                self.path_midlines = path_midlines
                self.midlines = np.load(path_midlines, allow_pickle = True)[()]
            else:
                print('No midlines file found in path!')
        return self
        
    def retrain_unet(self, upSample = 10, imgExt = 'bmp',epochs = 50,
                     saveModel = True, verbose = 0, use_newer_images:bool = False):
        from apCode.machineLearning.ml import retrainU
        import os
        from apCode.FileTools import findAndSortFilesInDir
        if (not hasattr(self, 'path_images_train')) | (use_newer_images):
            p = findAndSortFilesInDir(self.path_session, search_str = 'imgs_train')
            if len(p)>0:
                p = p[-1]
                self.path_images_train = os.path.join(self.path_session, p)                 
            else:
                print('No training images found, run ".copy_imgs_for_training_unet" first!')
                return None
        p = os.path.split(self.path_images_train)[-1]
        search_str = p.replace('train','mask')
        fldr_mask = findAndSortFilesInDir(self.path_session, search_str = search_str)
        if len(fldr_mask)>0:
            self.path_masks_train = os.path.join(self.path_session,fldr_mask[-1])
        else:
            print('No path to image masks found. Check!!')
            return None
        print(f'Training from images in {self.path_images_train}')
        uHist = retrainU(self.unet, self.path_images_train, self.path_masks_train, 
                 upSample = upSample, imgExt = imgExt, epochs = epochs, 
                 saveModel = saveModel,verbose = verbose)
        file_unet = findAndSortFilesInDir(self.path_session, ext = 'h5', 
                                          search_str = 'trainedU')[-1]
        self.path_unet = os.path.join(self.path_session, file_unet)
        self.unet_training_history = uHist
    
    def save_images_for_training(self, images):
        from apCode.volTools import img as img_
        import os
        from apCode import util
        imgPath = self.path_session
        p = f'imgs_train_{util.timestamp("min")}'
        imgDir = os.path.join(imgPath,p)
        img_.saveImages(images, imgDir = imgDir)
    
    def svd_of_tailAngles(self, nComponents = 3):
        """
        Do SVD to determine fish eigenshapes, then get the timeseries of their weights.
        """
        from sklearn.decomposition import TruncatedSVD
        import numpy as np
        ### Firt concatenate all head and/or tail trials to determine eigenshapes
        if hasattr(self,'tailAngles_flt_den'):
            tailAngles = self.tailAngles_flt_den
        else:
            tailAngles = self.tailAngles_corr
        ta = []
        for hort in tailAngles.keys():
            ta.append(tailAngles[hort])
        ta = np.concatenate(np.concatenate(ta,axis = 0),axis = 1)
        svd = TruncatedSVD(n_components=3, random_state=123).fit(ta.T)
        ta_svd = {}
        for hort in tailAngles.keys():
            ta_svd[hort] = []
            for ta in tailAngles[hort]:
                ta_svd[hort].append(svd.transform(ta.T).T)
            ta_svd[hort] = np.array(ta_svd[hort])
        svd.tailAngles_ = ta_svd
        self.svd = svd
        return self
                
    def tangent_angles_along_midlines(self, midlines, n:int = 50):
        """
        Returns tangent angles (cumsum of curvatures) along midlines
        Parameters
        ----------
        midlines: list, (T,) or array (T,K,2)
            Collection of midlines
        n: int
            Number of tangent angles per midline
        dt: scalar
            Sampling interval at which behavior images were collected. 
            Used for lowpass filtering.
        lpf: scalar
            Low pass filter value
        nWaves: int
            Parameter n for wavelet denoising of timeseries using spectral.Wden.wden
        Returns
        -------
        ta: array, (T,n,2)
            Tangent angles along the midlines
        """
        import numpy as np
        from apCode.behavior.FreeSwimBehavior import track
        kappas = track.curvaturesAlongMidline(midlines, n = n)        
        return np.cumsum(kappas,axis =0)
    
    def match_ca_and_behav_trls(self):
        """
        Sometimes, because ScanImage can crash, I end up with more behavior trials than
        Ca trials. This removes excess trials from behavior (or Ca trials) to make
        subsequent trial-matching easier
        """
        import numpy as np
        dff = self.trialized_dff()
        for hort in self.tailAngles.keys():
            behav = self.tailAngles[hort]
            nTrls = np.min((dff[hort].shape[0],behav.shape[0]))
            dff[hort] = dff[hort][:nTrls]
            self.tailAngles[hort] = self.tailAngles[hort][:nTrls]
            if hasattr(self, 'tailAngles_corr'):
                self.tailAngles_corr[hort] = self.tailAngles_corr[hort][:nTrls]            
            
    def trialized_dff(self):
        """
        Returns the head and tail dff arrays in trialized format, where in the first
        axis is trial number. This makes it easy to access specific trials
        """
        import numpy as np
        dff_trl = {}
        for hort in self.tailAngles.keys():
            dff = self.roi_ts[f'{hort}_dff']
            dff_trl[hort] = np.transpose(dff.reshape(-1, dff.shape[-1]//self.nImgsInTrl, self.nImgsInTrl),(1,0,2))
        self.dff_trl = dff_trl
        return dff_trl

def fishImgsForMidline(I, filtSize = 2.5, otsuMult = 1):
    """
    Given a probability image generated by a U-net or a similar NN, 
    returns an image that is ready for midline estimation. Assumes
    that a given image only has a single fish in it.
    Parameters
    ----------
    I: array, ([T,] M, N)
        Input image stack
    filtSize: scalar
        Size of gaussian convolution filter for smoothing images 
        before thresholding
    otsuMult: scalar
        Multiplier for thresholding with otsu. Lower values lead to more lax
        thresholding, while higher values lead to more stringent thresholding.
    n_jobs, verbose: see Parallel, delayed from joblib
    Returns
    -------
    I_flt: array, ([T,] M, N)
        Output image array with single fish blob in each image.
    """
    from skimage.filters import gaussian
    from skimage.measure import regionprops, label
    from apCode.volTools import img as img_
#    from joblib import Parallel, delayed
    from dask import delayed, compute
    def fishImgForMidline(img, filtsize, otsuMult):
#        img_dtype = img.dtype
        img_flt = gaussian(img, filtSize, preserve_range=True)
        img_bool = img_.otsu(img_flt, mult = otsuMult, binary=True)
        img_flt = img_flt*img_bool
        rp = regionprops(label(img_bool), img_flt)
        if len(rp)==0:
            print('No fish blobs found, check otsu threshold!')
            return img_flt
        else:
            ap = np.zeros(len(rp),)
            for count,rp_ in enumerate(rp):
                ap[count] = rp_.area*rp_.perimeter
            rp = rp[np.argmax(ap)]
        img_bool = img_flt*0
        img_bool[rp.coords] = 1
        img_fish = img_flt*img_bool
        img_fish = np.asarray(img_fish>0)
        return img_fish.astype(int)
    n_workers = np.min((os.cpu_count(),32))
    if np.ndim(I) ==2:
        I = I[np.newaxis,...]
    I_fish = compute(*[delayed(fishImgForMidline)(img, filtSize, otsuMult) for img in I],\
                       scheduler = 'processes', num_workers = n_workers)
    I_fish = np.asarray(I_fish)
    return I_fish

def fixEyeOrientations(eyeOr):
    """
    Fixes eye-orientations for jumps and also demeans to show relative changes
    Parameters
    ----------
    eyeOr: array, (2,N)
        Eye orientations in RADIANS. Rows are eyes and columns are time
    Returns
    -------
    eyeOr_fixed: array, (2,N)
        Corrected eye orientations in DEGREES
    """
    def toNegAngles(x):
        x_neg = x.copy()
        inds= np.where(x_neg>180)
        x_neg[inds] = np.mod(x_neg[inds],-180)
        return x_neg
    eyeOr_fixed = []
    for eye in eyeOr:
        foo = toNegAngles(np.unwrap(np.mod(eye*(180/np.pi),360)))
        eyeOr_fixed.append(foo-foo.mean())
    return np.array(eyeOr_fixed)       

def midlinesFromImages(images, n_jobs = 32, orientMidlines = True):
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
        from dask import delayed, compute
        import os
        
        getDists = lambda point, points: np.sum((point.reshape(1,-1)-points)**2,axis = 1)**0.5
        
        def identifyPointTypesOnMidline(ml):
            dist_adj = np.sqrt(2)+0.01    
            L = np.array([len(np.where(getDists(ml_,ml) < dist_adj)[0]) for ml_ in ml])
            endInds = np.where(L==2)[0].astype(int)
            branchInds = np.where(L==4)[0].astype(int)
            middleInds = np.where(L==3)[0].astype(int)
            return middleInds, endInds, branchInds
        
        def midlineFromImg(img):
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
            pt_brightest = np.unravel_index(np.argmax(img), img.shape)
            d_one = np.sum((ml_sort[0,:]-pt_brightest)**2)
            d_end = np.sum((ml_sort[-1,:]-pt_brightest)**2)
            if d_one > d_end:#                
                ml_sort = np.flipud(ml_sort)
                d = np.flipud(d)
            d = np.cumsum(np.insert(d,0,0))
            ml_sort = np.fliplr(ml_sort)
            ml_dist = (ml_sort, d)                              
            return ml_sort, ml_dist
        
        def orientMidlines_(midlines):
            inds = np.arange(1,len(midlines))
            count = 0         
            for ind in inds:        
                d = np.sum((midlines[0][0]-midlines[ind][0])**2)
                d_flip = np.sum((midlines[0][0]-np.flipud(midlines[ind])[0])**2)
                if d> d_flip:
                    count = count + 1
                    midlines[ind] = np.flipud(midlines[ind])         
            return midlines
        
        if np.ndim(images) == 2:
            images = images[np.newaxis,...]
        n_workers = np.min((os.cpu_count(), 48))
        midlines, ml_dist = zip(*compute(*[delayed(midlineFromImg)(img) for img in images], scheduler = 'processes', num_workers = n_workers))
        midlines = np.array(midlines)
        if orientMidlines:
            print('Orienting midlines')
            midlines = orientMidlines_(midlines)
        return midlines, ml_dist       



def midlinesToCurvatures_lines(midlines, n_angles = 8, upSample = True,
                               parallel = True, verbose = 1):
    """ 
    Given a list of midline coordinates, returns the curvatures estimated by 
    fitting a set of lines.
    Parameters
    ----------
    midlines: list, (N,)
        List of midline coordinates, where each element, M[n] has shape (K,2)
        where K is number of midline points in 2D space
    n_angles: integer
        Number of angles to estimate by fitting n_angles + 1 lines to each set of
        midline curves
    upSample: Boolean
        If True, then up-samples curvature angles to number of midline points
    parallel: Boolean
        If True, runs parallel loops
    verbose: scalar
        Verbosity parameter for Parallel function
    Returns
    ------
    thetas: list (N,)
        List of curvature angles for the list of midlines
    """
    import numpy as np
    from apCode.geom import fitLines
    from apCode.SignalProcessingTools import timeseries as ts
    if np.ndim(midlines) !=3:
        midlines = midlines[np.newaxis,:,:]
    len_ml = len(midlines[0])
    if not parallel:
        thetas = []
        for count, ml in enumerate(midlines):
            thetas = np.array([fitLines(ml, n_angles = n_angles)[0] for ml in midlines])
    else:
        from sklearn.externals.joblib import Parallel, delayed
        import multiprocessing as mp
        n_jobs = np.min([22,mp.cpu_count()])
        thetas = Parallel(n_jobs= n_jobs, verbose = verbose)(delayed(fitLines)(ml, n_angles = n_angles) for ml in midlines)
        thetas = np.array([th[0] for th in thetas])
    
    if upSample:
        print('Up-sampling')
        if not parallel:
            thetas_up = np.array([ts.resample(th,len_ml) for th in thetas])
        else:
            thetas_up = Parallel(n_jobs=n_jobs,verbose = 0)(delayed(ts.resample)(th,len_ml) for th in thetas)
            thetas_up = np.array(thetas_up)
    else:
        thetas_up = np.array(thetas)
    return thetas_up.T    
    
def pMapToMidline_vMinusOne(imgName, filtSize = 5, kernel = 'gauss', headPos = [],
                      nhood = 4, len_midline = 60):
    """
    Function for returning the midline indices along with distances between the points on the
    midline and the curvatures along the midline.
    Parameters
    ----------
    imgName: string
        Path to the image
    filtSize: integer
            Size of filter used for smoothing images before midline estimation.
            Set to None, for no filtering
    kernel: string
        Type of kernel to use for smoothing images. See volTools.img.filtImgs
    headPos: array, (2,)
        The xy coordinates of a point close to the head. Used for orienting
        midline curves in the right direction.
    nhood: integer
        Radius of neighborhood. Used for snapping midline returned by thinning
        algorithm to the true center using local pixel intensities
    len_midline: integer
        Final desired length of the the midlines in number of pixels.
        
    Returns
    -------
    out: 4 element list
        midline, dMidline, kappa, sKappa
        midline: array, (M,2)
            The midline; M[m,0] is the x coordinate of m-th point on the midline.
        dMidline: array, (M,)
            The distances between succcessive points of the midline. Computed 
            using gradient instead of diff, so has the same number of points as 
            the midline
        kappa: array, (M,)
            Curvatures along the midline computed as the difference in the angle
            between a tangent vector at one point and the next point on the midline.
            Gives positive angle for counterclockwise rotation of vectors
        sKappa: array, (M,)
            Cumulative sums of curvatures along the midline. Equivalent to the
            angles of the tangents along the midline, but the angles are w.r.t 
            the angle of the first tangent on the midline, which is set to 0.            
    """      
    import apCode.geom as geom
    from skimage.io import imread
    import numpy as np
    dist = lambda u,v: np.sqrt(np.sum((v-u)**2))
    arcLens = lambda c: np.sqrt(np.sum((np.gradient(c)[0])**2,axis =1))
    img = imread(imgName)
    
    if np.size(headPos)==0:
        headPos = estimateHeadSideInFixed(img)  
    
    if not filtSize == None:
        img_conv = volt.img.filtImgs(img,filtSize= filtSize, kernel=kernel, process = 'serial')
    else:
        img_conv = img
        
    img_otsu = volt.img.otsu(img_conv)
    #midline = geom.sortCurvePts(volt.morphology.thin_weighted(img_otsu, nhood = nhood).T)[0]
    midline = geom.sortPoints(volt.morphology.thin_weighted(img_otsu, nhood = nhood).T, headPos)[0] 
    midline = geom.smoothenCurve(midline, N = len_midline)
    if dist(midline[0],headPos) > dist(midline[-1], headPos):
        midline = np.flipud(midline)
    dMidline = arcLens(midline)
    kappa = geom.dCurve(midline)
    sKappa = np.cumsum(kappa)
    sKappa = sKappa-sKappa[0]
    
    return midline, dMidline, kappa, sKappa

def plotMidlines(midlines, shift_x = 1,shift_y = 10, nPre:int = 100, 
                      figSize = (20,5), x = None, alpha = 0.2):
        """ Convenience function for plotting a set of midlines next to each other
        in a way that makes it easy to visually assess midline tracking or movement
        Parameters
        ---------
        midlines: List of len (T,), or array of shape (T, K, 2)
            Midlines to plot
        shift_x, shift_y: scalars
            Horizontal and vertical shifts when plotting successive midlines. 
            Note: Shift along vertical axis is sinusoidal and shift_y is the amplitude of this.
        nPre: int
            Number of pre-stimulus points. Used to plot midline coincident with stimulus
            in red, while all others are plotted in black
        figSize: tuple
            Figure size
        x: array, (T,)
            Time axis; Not yet implemented
        alpha: scalar
            Alpha value for plotted midlines
        Returns
        -------
        fh: Figure handle
            Handle to the plotted figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        yShifts = shift_y*np.sin(np.linspace(0,2*np.pi,len(midlines)))
        if np.any(x==None):
            x = np.arange(len(midlines))
        fh = plt.figure(figsize = figSize)
        for count, _ in enumerate(midlines):
            _ = np.array(_)
            _[:,0] = _[:,0] + (shift_x*count)
            _[:,1] = _[:,1] + yShifts[count]
            if count == nPre:
                plt.plot(*_.T,'r')
            else:
                plt.plot(*_.T,'k', alpha = alpha)
        plt.gca().invert_yaxis()
        return fh

def pMapToMidline(imgOrPath, filtSize = 5, kernel = 'gauss', headPos = [],
                      nhood_snap = 4, len_midline = 60, nhood_sort = 2):
    """
    Function for returning the midline indices along with distances between the points on the
    midline and the curvatures along the midline.
    Parameters
    ----------
    imgName: string
        Path to the image
    filtSize: integer
            Size of filter used for smoothing images before midline estimation.
            Set to None, for no filtering
    kernel: string
        Type of kernel to use for smoothing images. See volTools.img.filtImgs
    headPos: array, (2,)
        The xy coordinates of a point close to the head. Used for orienting
        midline curves in the right direction.
    nhood_snap: integer
        Radius of neighborhood. Used for snapping midline returned by thinning
        algorithm to the true center using local pixel intensities
    len_midline: integer
        Final desired length of the the midlines in number of pixels.
        
    Returns
    -------
    out: 4 element list
        midline, dMidline, kappa, sKappa
        midline: array, (M,2)
            The midline; M[m,0] is the x coordinate of m-th point on the midline.
        dMidline: array, (M,)
            The distances between succcessive points of the midline. Computed 
            using gradient instead of diff, so has the same number of points as 
            the midline
        kappa: array, (M,)
            Curvatures along the midline computed as the difference in the angle
            between a tangent vector at one point and the next point on the midline.
            Gives positive angle for counterclockwise rotation of vectors
        sKappa: array, (M,)
            Cumulative sums of curvatures along the midline. Equivalent to the
            angles of the tangents along the midline, but the angles are w.r.t 
            the angle of the first tangent on the midline, which is set to 0.            
    """      
    import apCode.geom as geom
    from skimage.io import imread
    from skimage.morphology import thin
    from skimage.measure import label
    from scipy.signal import convolve2d as conv
    from apCode.SignalProcessingTools import gausswin
    import os
    dist = lambda u,v: np.sqrt(np.sum((v-u)**2))
    arcLens = lambda c: np.sqrt(np.sum((np.gradient(c)[0])**2,axis =1))
    
    if isinstance(imgOrPath,str):
        if os.path.isfile(imgOrPath):
            img = imread(imgOrPath)
        else:
            print('Input must be path to an image')
    else:
        img = imgOrPath    
    
    if np.size(headPos)==0:
        headPos = estimateHeadSideInFixed(img) # Estimation of the point in the that is closer 
                                        ### to where the head would have than most, if not all of
                                        ### the midline points. This is used in the point sorting
                                        ### algorithm to follow
    if not filtSize == None:
        ker = gausswin(filtSize)[:,np.newaxis]*gausswin(filtSize)[np.newaxis,:]
        img_conv = conv(img,ker,mode = 'same')
    else:
        img_conv = img        
    
    img_otsu = volt.img.otsu(img_conv) # Threshold image using otsu from sklearn.filters such that
                    ### pixels above the otsu threshold are not zeroed out, while all other pixels are
    
    img_bool = (img_otsu >0).astype(int)
    lbls = np.unique(label(img_bool))
    if len(lbls)>2:
        img_otsu = geom.connectIslands(img_otsu,mult = np.max(img_otsu))[0]
        img_bool = (img_otsu >0).astype(int)
    ml_raw = np.fliplr(np.array(np.where(thin(img_bool))).T) # First thin traditionally. Better for sorting    
    srcInd = np.argmin(dist(headPos,ml_raw)) # Ref point to start sorting from. I am using the midline point 
            ### that is closest to the point that is approximately the closest in the image to where the head 
            ### would have been. Although I doubt that this is that crucial since the sorting algorithm seems
            ### robust enough for this to not matter.    
    opt_order = geom.sortPointsByDist(ml_raw, n_neighbors = nhood_sort, src = srcInd) #Sorting using NearestNeighbors
                ### from sklearn.neighbors
#     ml = geom.sortCurvePts(volt.morphology.thin_weighted(img_otsu, nhood = nhood).T)[0]
#     midline = geom.sortPoints(volt.morphology.thin_weighted(img_otsu, nhood = nhood_snap).T, headPos)[0]
    ml_sort = ml_raw[opt_order,:]
    ml_wt = volt.morphology.thin_weighted(img_otsu, nhood = nhood_snap,points = ml_sort) # Drift midline closer 
                ### to a true center. Inspired by the KH Huang paper.
    ml_wt = np.array(ml_wt).T
    
    ml_smooth = geom.smoothenCurve(ml_wt, N = len_midline) # Smoothen midline using spline interpolation.
    if dist(ml_smooth[0],headPos) > dist(ml_smooth[-1], headPos):
        ml_smooth = np.flipud(ml_smooth) #Flip if tail is closer than head
    dMidline = arcLens(ml_smooth) # Distances between successive points of the midline
    kappa = geom.dCurve(ml_smooth) # Curvatures along the midline
    sKappa = np.cumsum(kappa) # Cumulative sum of the curvatures
    sKappa = sKappa-sKappa[0]  # Angles w.r.t the first point  
    return (ml_smooth, dMidline, kappa, sKappa)

def rawImagesToEyeInfo(images, unet, baryCenter, filtSigma = None, otsuMult = 1):
    """
    Given a collection raw images, segments eyes using trained U net and returns
    eye orientations, centroids, coordinates, raw cropped images, raw probability
    images
    Parameters
    ----------
    images: array-like, ([T,], M, N)
        Raw images in which to segment eyes
    unet: object
        Trained Keras U net model object
    baryCenter: 2-tuple
        Barycenter of the fish eyes used for cropping images to the size accepted
        by the U net.
    filtSigma: scalar or None
        The sigma value for skimage.filters.gaussian. If not None, then will
        convolve probability images with this filter for improving detection.
    otsuMult: scalar
        Multiple of the otsu threshold used to get binary images from probability
        images. Larger values lead to more stringent thresholding.
    Returns
    -------
    out: dict
        Dictionary with useful eye info such as 'orientations', 'centroids', 'coordinates',
        'img_prob', 'img_raw'
    """
    
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian
    from dask import delayed, compute
    import os
    n_workers = np.min((os.cpu_count(), 32))    
    def largestTwo(rp):
        a = np.array([r.filled_area for r in rp])
        inds = np.argsort(a)[-2:]
        rp_flt = [rp[ind] for ind in inds]
        centroids = np.array([r['weighted_centroid'] for r in rp_flt])
        refPt = np.array([0,0]).reshape(1,2)
        dists = np.sum((centroids -refPt)**2,axis = 1)
        inds = np.argsort(dists)
        rp_flt = np.asarray(rp_flt)[inds]
        return rp_flt 
    
    def probImgToEyeInfo(img_pred, filtSigma= None, otsuMult = 1):
        if not filtSigma == None:
            img_pred = gaussian(img_pred, sigma = filtSigma)
        img_otsu = volt.img.otsu(img_pred, binary= True, mult = otsuMult).astype(int)
        img_lbl = label(img_otsu)
        rp = regionprops(img_lbl, intensity_image= img_pred, coordinates = 'rc')    
        rp_filt = largestTwo(rp)
        orientations, centroids, coords = [], [], []
        for r in rp_filt:
            orientations.append(r['orientation'])
            centroids.append(r['weighted_centroid'])
            coords.append(r['coords'])
        orientations, centroids, coords = np.asarray(orientations), np.asarray(centroids),np.asarray(coords)   
        return orientations, centroids, coords

    cropSize = unet.input_shape[1]
    if np.ndim(images)==2:
        images= images[np.newaxis,...]
    images_crop = volt.img.cropImgsAroundPoints(images,baryCenter, cropSize = cropSize)
    images_pred = np.squeeze(unet.predict(images_crop[...,np.newaxis], verbose = 0))
    orientations, centroids, coords = \
    zip(*compute(*[delayed(probImgToEyeInfo)(img, filtSigma= filtSigma,\
                   otsuMult = otsuMult) for img in images_pred], scheduler = 'processes', num_workers = n_workers))    
    orientations, centroids = np.array(orientations), np.array(centroids)
    out = dict(orientations = orientations, centroids = centroids,\
               coords = coords, images_raw = images_crop, images_prob = images_pred,\
               baryCenter = baryCenter, cropSize = cropSize)
    return out

def readPeriStimulusTifImages(tifDir, basPath, nBasCh = 16, ch_camTrig = 'patch1', ch_stim = 'patch3',\
                  tifNameStr = '', time_preStim = 1, time_postStim = 10, thr_stim = 0.5,\
                  thr_camTrig = 3, maxAllowedTimeBetweenStimAndCamTrig = 0.5, n_jobs = 2):
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
    import apCode.FileTools as ft
    import apCode.ephys as ephys
    import apCode.SignalProcessingTools as spt
    import apCode.util as util    
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
    tifInfo = ft.scanImageTifInfo(tifDir)
    nCaImgs = np.sum(tifInfo['nImagesInFile'])
    print('{} images from all tif files'.format(nCaImgs))
 
    ### Check for consistency in the number of image channels in all files.
    if len(np.unique(tifInfo['nChannelsInFile']))>1:
        print('Different number of image channels across files, check files!')
        return None
    nImgCh = tifInfo['nChannelsInFile'][0]
    
    ### Get a list of indices corresponding to images in each of the tif files
    inds_imgsInTifs = getImgIndsInTifs(tifInfo)
    
    ### Read bas file to get stimulus and camera trigger indices required to align images and behavior
    print('Reading bas file, detecting stimuli and camera triggers...')
    bas = ephys.importCh(basPath,nCh=nBasCh)
    inds_stim = spt.levelCrossings(bas[ch_stim], thr = thr_stim)[0]
    inds_camTrig = spt.levelCrossings(bas[ch_camTrig], thr = thr_camTrig)[0]
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
    nPreStim = int(np.round(time_preStim/dt_ca))
    nPostStim = int(np.round(time_postStim/dt_ca))
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

def readPeriStimulusTifImages_volumetric(tifDir, basPath, nBasCh = 16, ch_camTrig = 'patch1', ch_stim = 'patch3',\
                  tifNameStr = '', time_preStim = 2, time_postStim = 10, thr_stim = 0.5,\
                  thr_camTrig = 3, maxAllowedTimeBetweenStimAndCamTrig = 0.5, n_jobs = 2):
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
#    import tifffile as tff
#    import apCode.FileTools as ft
    import apCode.ephys as ephys
    import apCode.SignalProcessingTools as spt
    from dask import compute
    from dask.diagnostics import ProgressBar
    from scipy.stats import mode    
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
    print('Reading ScanImage tif files as dask array...')
    images, tifInfo = volt.dask_array_from_scanimage_tifs(tifDir)    
 
    ### Read bas file to get stimulus and camera trigger indices required to align images and behavior
#    print('Reading bas file, detecting stimuli and camera triggers...')
    bas = ephys.importCh(basPath,nCh=nBasCh)
    inds_stim = spt.levelCrossings(bas[ch_stim], thr = thr_stim)[0]
    inds_camTrig = spt.levelCrossings(bas[ch_camTrig], thr = thr_camTrig)[0]
    print(f'{len(inds_camTrig)} camera triggers detected!')
    images_ser = images.reshape(-1,*images.shape[2:])
    nFrames_total = images_ser.shape[0]
    print(f'{nFrames_total} total number of frames...')
    if len(inds_camTrig)> nFrames_total:
        inds_camTrig = inds_camTrig[:nFrames_total]
    elif len(inds_camTrig)<nFrames_total:
        print(f'{nFrames_total -len(inds_camTrig)} fewer camera triggers than images, check threshold!')
    
    nFramesPerVol = tifInfo['nFramesPerVolume'][0]
#    print(f'{nFramesPerVol} frames per volume')
    inds_stackInit = inds_camTrig[::nFramesPerVol]
    dt_vec = np.diff(bas['t'][inds_stackInit])
    dt_ca = np.round(np.mean(dt_vec)*100)/100
    print(f'Ca volume acquisition rate = {np.round((1/dt_ca)*10)/10} Hz')

    ### Indices of ca images closest to stimulus
    inds_stim_ca = spt.nearestMatchingInds(inds_stim, inds_stackInit)
    n_pre = int(np.round(time_preStim/dt_ca))
    n_post = int(np.round(time_postStim/dt_ca))
    print(f'{n_pre} pre stim images, {n_post} post-stim images')
    with ProgressBar():
        images_trl = compute(*spt.segmentByEvents(images,inds_stim_ca, n_pre, n_post))
    nImgs = np.array([img.shape[0] for img in images_trl])
    inds_del = np.where(nImgs< mode(nImgs)[0])
    images_trl = np.delete(images_trl,inds_del,axis = 0)    
    try:
#        images_trl = np.squeeze(np.array(images_trl))
        images_trl = np.array([np.array(img) for img in images_trl])
        images_trl= np.squeeze(images_trl)
    except:
        pass
    
    out = dict(stimInds = inds_stim, camTrigInds = inds_camTrig, stimInds_ca = inds_stim_ca,\
               tifInfo = tifInfo, images_trl = images_trl, inds_excluded = inds_del)
    return out


def readScanImageTif(tifPath):
    """ Given the full path to a tif file saved by ScanImage, returns the hyperstack of images
    Parameters
    ----------
    tifPath: string
        Full path to .tif file written by ScanImage
    Returns
    -------
    images: array (T,[C],M,N,[Z])
        Image hyperstack, where T is number of time points, C is number of channels, M, N, are image 
        height and width, and Z is number of slices in a single volume
    """
    import ScanImageTiffReader as sitr
    import tifffile as tff
    def tifInfo(filePath):
        with tff.TiffFile(filePath) as tf:
            nChannels = len(tf.scanimage_metadata['FrameData']['SI.hChannels.channelSave'])
            nImgs = int(len(tf.pages)/nChannels)
            return (nChannels, nImgs)
    nCh, nImgs = tifInfo(tifPath)
    with sitr.ScanImageTiffReader(tifPath) as reader:
        images = reader.data()
        if nCh >1 :                        
            images = images.reshape(-1,nCh,*images.shape[1:])
    return images

def registerTrializedImgStack_old(I, iCh_ref = 0, nImgs_ref = 40, filtSigma = 3, filtMode = 'wrap', 
                              denoise:bool = False, regMethod = 'st'):
    """
    Registers image stack returned by apCode.behavior.headFixed.readPeriStimulusTifImages
    Parameters
    ----------
    I: array, (K,T,C,M,N)
        Trialized image stack returned by headFixed.readPeriStimulusTifImages.
        K = # of trials
        T = # of time points in each trial
        C = # of image channels
        M, N = image dimensions
    iCh_ref: int
        If C (# of image channels) > 1, then iCh_ref specifies the channel to use for computing
        registration parameters (shifts), which are then applied to all other channels
    nImgs_ref: int
        Number of images to average from the beginning of the stack to generate the 
        reference image.
    filtSigma: scalar
        The standard deviation of the gaussian filter for filtering images prior to
        registration. If filtSigma is None, then skips filtering. See skimage.filters.gaussian
        for more info.
    filtMode: string
        See skimage.filters.gaussian for info
    denoise: bool
        If True, denoises images using skimage.restoration.denoise_wavelet before 
        filtering/registration. WARNING: Denoising will convert dtype of image to float
        and also change the range of values
    regMethod: string, 'st', 'cr', 'cpwr'
        Specifies registration method
        'st' (standard translation): Uses skimage.feature.register_translation with a fixed 
            reference image.
        'cr' (caiman rigid): Uses rigid registration implemented in CaImAn. Uses dynamically updated
            reference image.
        'cpwr' (caiman piecewise rigid): Uses piecewise rigid registration implemented in caiman.
        
    Returns
    -------
    I_reg: array, (K*T, C, M, N)
        Registered image stack
    regObj: object
        Registration object        
    """
    print('Serializing and filtering images prior to registration...')
    imgDims = I.shape
    I = I.reshape(-1,*imgDims[2:])
    if denoise:
        print('Denoising prior to registration...')
#        imgRange = I.max()-I.min()
#        I = volt.denoiseImages(I/imgRange)
        I = volt.denoiseImages(I)
    else:
        print('No denoising')
    
    if filtSigma != None:
        I_flt = volt.img.gaussFilt(I,sigma = filtSigma, mode = filtMode)
    else:
        I_flt = I.copy()

    ref = I_flt[:nImgs_ref,iCh_ref,...].mean(axis = 0)
    
    print('Computing registration parameters...')
    try:
        regObj = volt.Register(backend = 'joblib', regMethod = regMethod).fit(I_flt[:,iCh_ref,...], ref = ref)
    except:
        print('Parallel registration with joblib failed')
        try:
            print('Trying with "dask" backend instead of "joblib"...')
            regObj = volt.Register(backend = 'dask', scheduler = 'processes', regMethod = regMethod).fit(I_flt[:,iCh_ref,...], ref = ref)
        except:
            print('Trying in serial mode...')
            regObj = volt.Register(n_jobs = 1, regMethod = regMethod).fit(I_flt[:,iCh_ref,...], ref = ref)

    regObj.ref_ = ref
    I_flt_reg = I_flt.copy()
    I_reg = I.copy()
    print('Applying registraton...')
    for iCh in range(I_flt.shape[1]):
        print('Channel # {}'.format(iCh))
        I_flt_reg[:,iCh,...] = regObj.transform(I_flt[:,iCh,...])
        I_reg[:,iCh,...] = regObj.transform(I_reg[:,iCh,...])
    return I_reg, regObj

def registerTrializedImgStack(I, iCh_ref = 0, nImgs_ref = 40, filtSigma = 3, filtMode = 'wrap', 
                              denoise:bool = False, regMethod = 'st'):
    """
    Registers image stack returned by apCode.behavior.headFixed.readPeriStimulusTifImages
    Parameters
    ----------
    I: array, (K,T,C,M,N)
        Trialized image stack returned by headFixed.readPeriStimulusTifImages.
        K = # of trials
        T = # of time points in each trial
        C = # of image channels
        M, N = image dimensions
        Note: To work properly the input array must be 5-D. To use for 1 channel images,
        use np.expand_dims(I,2), where I is the image stack.
    iCh_ref: int
        If C (# of image channels) > 1, then iCh_ref specifies the channel to use for computing
        registration parameters (shifts), which are then applied to all other channels
    nImgs_ref: int
        Number of images to average from the beginning of the stack to generate the 
        reference image.
    filtSigma: scalar
        The standard deviation of the gaussian filter for filtering images prior to
        registration. If filtSigma is None, then skips filtering. See skimage.filters.gaussian
        for more info.
    filtMode: string
        See skimage.filters.gaussian for info
    denoise: bool
        If True, denoises images using skimage.restoration.denoise_wavelet before 
        filtering/registration. WARNING: Denoising will convert dtype of image to float
        and also change the range of values
    regMethod: string, 'st', 'cr', 'cpwr'
        Specifies registration method
        'st' (standard translation): Uses skimage.feature.register_translation with a fixed 
            reference image.
        'cr' (caiman rigid): Uses rigid registration implemented in CaImAn. Uses dynamically updated
            reference image.
        'cpwr' (caiman piecewise rigid): Uses piecewise rigid registration implemented in caiman.
        
    Returns
    -------
    I_reg: array, (K*T, C, M, N)
        Registered image stack
    regObj: object
        Registration object        
    """
#    import apCode.volTools as volt
    import numpy as np
    from skimage.exposure import rescale_intensity
    print('Serializing and filtering images prior to registration...')
    if np.ndim(I) !=5:
        raise IOError("Input must be 5 dimensional, see help for fix!")
    imgDims = I.shape
    I = I.reshape(-1,*imgDims[2:])
    I_norm = I.copy()
    for ch in range(I.shape[1]):
        print(f'Rescaling ch {ch}')
        I_norm[:,ch,...] = rescale_intensity(I_norm[:,ch,...])
    I_norm = I_norm.mean(axis = 1)      
    if denoise:
        print('Denoising prior to registration...')
        I_norm = volt.denoiseImages(I_norm)
    else:
        print('No denoising')
    
    if filtSigma != None:
        print(f'Gaussian filtering images with sigma = {filtSigma}')
        I_flt = volt.img.gaussFilt(I_norm,sigma = filtSigma, mode = filtMode)
    else:
        I_flt = I_norm    
    ref = I_flt[:nImgs_ref,...].mean(axis = 0)
    
    print('Computing registration parameters...')
    try:
        regObj = volt.Register(backend = 'joblib', regMethod = regMethod).fit(I_flt, ref = ref)
    except:
        print('Parallel registration with joblib failed')
        try:
            print('Trying with "dask" backend instead of "joblib"...')
            regObj = volt.Register(backend = 'dask', scheduler = 'processes', regMethod = regMethod).fit(I_flt, ref = ref)
        except:
            print('Trying in serial mode...')
            regObj = volt.Register(n_jobs = 1, regMethod = regMethod).fit(I_flt, ref = ref)
    regObj.ref_ = ref
    I_reg = []
    print('Applying registraton...')
    for iCh in range(I.shape[1]):
        print('Channel # {}'.format(iCh))
        I_reg.append(regObj.transform(I[:,iCh,...]))
    I_reg = np.asarray(I_reg)
    dims = np.arange(np.ndim(I_reg))
    I_reg = np.transpose(I_reg,(1,0,*dims[2:]))
    return I_reg, regObj

def register_trialized_volumes_by_slices(images, filtSize = 1, regMethod = 'cpwr',**kwargs):
    """
    Register image volumes slice-by-slice
    Parameters
    ----------
    images: array, (nTrials, nTimePoints, nSlices, nRows, nCols)
        Image hyperstack to perform registration on.
    filtSize: scalar
        Gaussian filter size (sigma) for smoothing of images prior to registration. The
        smoothing is only done to compute registration parameters, which are then applied
        to raw images.
    regMethod: str
        Method of registration to use.
        'cr': Caiman's rigid
        'cpwr': Caiman's piecewise rigid
        'st': skimage's register_translation
    **kwargs: Other keyword arguments for apCode.volTools.Register class
    Returns
    -------
    images_reg: array, same shape as images
        Registered image hyperstack
    regObj: list
        List of registration objects (one for each slice)
    """
    stackDims = images.shape
    images_ser = images.reshape(-1, *images.shape[2:])
    images_ser = np.swapaxes(images_ser,0,1)
    if len(kwargs)>0:
        reg = volt.Register(filtSize = filtSize, regMethod = regMethod, **kwargs)
    else:
        reg = volt.Register(filtSize = filtSize, regMethod = regMethod)
    
    regObj = []
    images_reg = []
    print('Registering...')
    for z, frame in enumerate(images_ser):
        print(f'Slice {z+1}/{images_ser.shape[0]}')
        ro = reg.fit(frame)
        regObj.append(ro)
        images_reg.append(ro.transform(frame))
    images_reg = np.swapaxes(np.array(images_reg),0,1)
    images_reg = images_reg.reshape(stackDims)
    return images_reg, regObj
    

def register_volumes_by_slices_and_trials(images, regMethod='cpwr',\
                                          backend='dask', upSample=1,\
                                              filtSize=None):
    """
    Register image volumes slice-by-slice and trial-by-trial
    Parameters
    ----------
    images: array, (nTrls, nTimePts, nSlices, nRows, nCols)
        Image array to register.
    regMethod: str
        Method of registration. Options are:
        'cpwr'- Caiman piecewise rigid registration
        'st' - skimage translation
        'cr' - Caiman rigid
    upSample: int
        Factor by which to upsample images prior to registration. Only valid
        if regMethod = 'st'.
    backend: str, 'dask' or 'joblib'
        Type of parallel backend to use.
    filtSize: int or None
        Gaussian filtersize to use if images are to be filtered prior to 
        registration. If None, no filtering.
    Returns
    -------
    images_reg: array, shape(images)
        Registered image stack.
    
    """
#    import apCode.volTools as volt
#    stackDims = images.shape    
    images = np.swapaxes(images,1,2)
    reg = volt.Register(backend = backend, regMethod = regMethod,\
                        upsample_factor = upSample)
    images_reg, regObj = [],[]
    print('Registering...')
    nTrls = images.shape[0]
    for iTrl, trl in enumerate(images):
        print(f'Trl# {iTrl+1}/{nTrls}')
        ro, imgs_reg = [],[]
        for iSlice, frame in enumerate(trl):
            if not filtSize is None:
                frame_flt = volt.img.gaussFilt(frame, sigma=filtSize,\
                                               preserve_range=True)
                ro_now = reg.fit(frame_flt)
            else:
                ro_now = reg.fit(frame)
            ro.append(ro_now)
            imgs_reg.append(ro_now.transform(frame))
        regObj.append(ro)
        images_reg.append(imgs_reg)
    images_reg = np.swapaxes(np.array(images_reg),1,2)
    return images_reg, regObj

def roiTimeseriesAsMat(hFile, subtractBack = True):
    """
    Read ROI timeseries from HDF file (roiTimeseriesFromImagesInHDF
    must be run first) and returns as matrices in a dictionary
    Parameters
    ----------
    hFile: HDF file containing the ROI timeseries
    subtractBack: bool
        If True, and 'background' ROI exists, subtract this value
        from all the other ROIs
    nWaves: int
        If nWaves > 0, then denoises signals with wavelet denoising (apCode.spectral.Wden.wden)
        by setting the parameter n = nWaves
    Returns
    -------
    roi_ts: dict
        Dictionary with the following key-value pairs
        'h': Raw roi timeseries for head stimulation trials
        't': ---     ---      ----  tail ----
        'h_dff': Roi timeseries for head trials as df/f
        't_dff'  ---                tail trials  ----
    """
    import apCode.util as util
    import numpy as np
    
    roiNames = list(hFile['rois_cell'].keys())
    ht_iter = []
    if 'h' in hFile:
        ht_iter.append('h')
    if 't' in hFile:
        ht_iter.append('t')
    roi_ts = {}   
    ind_back = util.findStrInList('background',roiNames)    
    if np.size(ind_back)==0:
        print('No background ROI found')
    for ht in ht_iter:
        tsMat = []
        for rn in roiNames:
            ts = np.array(hFile[f'rois_cell/{rn}/ts/{ht}'])
            tsMat.append(ts)
        tsMat = np.array(tsMat)       
        if (np.size(ind_back)>0) & (subtractBack):         
            tsMat = tsMat - tsMat[ind_back[0]][np.newaxis,...]
            tsMat = np.delete(tsMat,ind_back, axis = 0)            
            print('Subtracted background ROI')               
        roi_ts[ht] = np.array(tsMat)
    roiNames = np.delete(roiNames,ind_back)
    roiNames_ascii = np.array([rn.encode(encoding = 'ascii', errors = 'ignore') for rn in roiNames])
    roi_ts['roiNames'] = roiNames
    roi_ts['roiNames_ascii'] = roiNames_ascii
    return roi_ts    

def roiTimeseriesFromImagesInHDF(hFile, saveToHdf = True):
    """ 
    Given an hdf file that contains ROI info (hFile['rois_cell]') and registered 
    Ca2+ images (hFile[hOrT/I_reg], where hOrT = 'h' or 't'), extracts
    Ca2+ timeseries for the ROIs and appends these to the "rois_cell" group 
    in the hdf file
    Parameters
    ----------
    hFile: HDF file
    saveToHDF: bool
        If true (default), directly saves the timeseries to hdf file.
    Returns
    -------
    roi_ts: List of delayed dask operations, which when computed yields the ROI
        timeseries or a dictionary with roi timeseries as arrays
    """
#    from dask import delayed
    import dask.array as da
    import dask
    from dask.diagnostics import ProgressBar
    import numpy as np
    import psutil
    import time
    def roiTSFromHFile(mask,images):  
        M = images*(np.flipud(mask.T))
        ts = M.mean(axis = -1).mean(axis = -1).T        
        return ts
    def getRoiMasks(rois_cell):
        masks = []
        for rk in rois_cell.keys():
            masks.append(dask.array.from_array(rois_cell[rk]['mask']))
        return np.array(dask.compute(*masks))
    rois_cell = hFile['rois_cell']
    masks = da.from_array(getRoiMasks(rois_cell))
    print(rois_cell.keys())
    roi_ts = {}
    if 'h' in hFile:
        roi_ts['h'] = []
    if 't' in hFile:
        roi_ts['t'] = []    
    tic = time.time()
    for htk in roi_ts.keys():
        images = da.from_array(hFile[htk]['I_reg'])
#        inMemBool = False
        if int(images.nbytes/(1024**3)) < int(0.80*psutil.virtual_memory().available/(1024**3)):
            print('RAM available, loading image stack from hdf...')
#            inMemBool = True
            tic = time.time()
            images = da.from_array(np.array(images))
            print(int(time.time() -tic),'s')        
        print('Extracting and storing roi timeseries for "{}" trials'.format(htk))
        for count, rk in enumerate(rois_cell.keys()):
#            roi_ = rois_cell[rk]
            ts_arr = roiTSFromHFile(masks[count], images)
            roi_ts[htk].append(ts_arr) 
            hPath = '{}/ts/{}'.format(rk,htk)                       
            if hPath in rois_cell:
                del rois_cell[hPath]
            if saveToHdf:
                if isinstance(ts_arr, dask.array.core.Array):                
                    dset = rois_cell.create_dataset(hPath, shape = ts_arr.shape)
                    ts_arr.store(dset)
                else:        
                    rois_cell.create_dataset(hPath, data = ts_arr)
        toc = np.round(10*((time.time()-tic)/60))/10
        print(f'{toc} mins')
        with ProgressBar():
            roi_ts[htk] = np.array(dask.compute(*roi_ts[htk]))
    return roi_ts         

def roiTs2Dff(roi_ts, trlLen:int = 550, nPre:int = 50, q:float = 20, nWave = 4, fitFirst:bool = True):
    """
    Given the dictionary (returned by roiTimeseriesAsMat) containing roi timseries
    updates the dictionary the signals expressed in df/f units
    """
    from apCode.util import findStrInList
    import numpy as np
    import dask
    keys = list(roi_ts.keys())
    ht_iter = []
    if len(findStrInList('h',keys))>0:
        ht_iter.append('h')
    if len(findStrInList('t',keys))>0:
        ht_iter.append('t')
    
    for ht in ht_iter:
        dff,sigs_norm = zip(*dask.compute(*[dask.delayed(toRatioDff)
        (r, trlLen = trlLen, nPre = nPre, q = q, nWave = nWave) for r in roi_ts[ht]]))
        roi_ts[f'{ht}_dff'] = np.asarray(dff)
        roi_ts[f'{ht}_perc']= np.asarray(sigs_norm)
    return roi_ts

def seeBehavior(eye_imgs, tail_imgs, eye_ts, tail_ts, fps = 30, n_pts = 50, 
                display = True, save = False, savePath = None, yLim_eyes = (-30, 30),
                yLim_tail = (-200,200),**kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation
    plt.rcParams['animation.ffmpeg_path'] = r'V:\Code\Python\FFMPEG\bin\ffmpeg.exe'
    from IPython.display import HTML
    import time
    
    N = eye_imgs.shape[0]
    eye_ts, tail_ts = np.array(eye_ts), np.array(tail_ts)
    if np.ndim(eye_ts)==1:
        eye_ts = eye_ts.reshape((-1,1))
    if np.ndim(tail_ts)==1:
        tail_ts = tail_ts.reshape((-1,1))
    
    cmap = kwargs.get('cmap', 'gray')
#    interp = kwargs.get('interpolation', 'nearest')
    dpi = kwargs.get('dpi', 70)
    plt.style.use(('seaborn-poster', 'seaborn-white'))
    fh = plt.figure(dpi = dpi)
    ax = [[]]*4
    ax[0] = fh.add_axes([0, 0.61, 0.34, 0.38])
    ax[0].set_aspect('equal')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_frame_on(False)
    im_eyes = ax[0].imshow(eye_imgs[0], cmap= cmap, vmin = eye_imgs.min(), vmax = eye_imgs.max())
    
    ax[1]= fh.add_axes([0.38, 0.62, 0.6, 0.29])
    ax[1].set_frame_on(False)
    dummy= np.zeros((n_pts,))*np.nan
    ax[1].plot(dummy)
    ax[1].get_xaxis().set_visible(False)    
    
    
    ax[2] = fh.add_axes([0, 0.02, 0.34, 0.54])
    ax[2].set_aspect('equal')
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_frame_on(False)
    im_tail = ax[2].imshow(tail_imgs[0], cmap = cmap, vmin = tail_imgs.min(), vmax = tail_imgs.max())
    

    ax[3] = fh.add_axes([0.38,0.1,0.6,0.4])
    ax[3].set_frame_on(False)
    ax[3].plot(dummy)
    ax[3].set_ylabel('Tail angles')
    
#    fh.tight_layout()
    
    def update_img(n):        
        im_eyes.set_data(eye_imgs[n])   
        im_tail.set_data(tail_imgs[n])
        ax[2].set_title('Frame # {}'.format(n))
        
        start= np.max((0, n-n_pts))
        t = np.arange(start, n+1)
        
        ax[1].cla()
        ax[1].plot(t,eye_ts[start:n+1,:])
        if len(t)>0:
            ax[1].plot(t[-1],eye_ts[n,0].reshape((1,-1)),marker = '$\\triangledown$', label = 'Left', c= plt.cm.tab10(0))
            ax[1].plot(t[-1],eye_ts[n,1].reshape((1,-1)),marker = '$\\triangledown$', label = 'Right', c= plt.cm.tab10(1))
#        ax[1].axhline(y = 0, ls = '--', c = 'w', lw = 1.5, alpha = 0.7)
        ax[1].set_xlim(n-n_pts, n+5)
        ax[1].set_ylim(yLim_eyes)
        ax[1].set_title('Eye angles')
        ax[1].legend(fontsize = 15, loc = 'upper left')
        
        ax[3].cla()
        ax[3].plot(t,tail_ts[start:n+1,:])
        if len(t)>0:
            ax[3].plot(t[-1], tail_ts[n,:], marker = '$\\triangledown$', c= plt.cm.tab10(0))
        ax[3].axhline(y = 0, ls = '--', c = 'k', lw = 1.5, alpha = 0.5)
        ax[3].set_xlim(n-n_pts,n+5)
        ax[3].set_ylim(yLim_tail)
        ax[3].set_title('Tail curvature')
    
    ani = animation.FuncAnimation(fh,update_img,np.arange(N),interval= fps, 
                                  repeat = False)    
    plt.close(fh)
    
    if save:
        print('Saving...')
        writer = animation.writers['ffmpeg'](fps=fps)
        if savePath != None:
            ani.save(savePath, writer = writer, dpi = dpi)
            print('Saved to \n{}'.format(savePath))
        else:
            vidName = 'video_{}.mp4'.format(time.strftime('%Y%m%d'))
            ani.save(vidName, writer=writer, dpi=dpi)
            print('Saved in current drirve as \n{}'.format(vidName))
        
    if display:
        print('Displaying...')
        return HTML(ani.to_html5_video())
    else:
        return ani 

def see_behavior_with_labels(images, ts, labels = None, fps = 30, display = True,\
                             save = False, savePath = None, yl = (-150, 150),\
                                 ms = 20, cmap_lbls= 'nipy_spectral', **kwargs):
    """
    Parameters
    ----------
    images: array, (nFrames, nRows, nCols)
        Images to display
    ts: array, (nFrames[,1])
        Timeseries to display
    labels: dict or None
        'inds': array, (n,), where n <= nFrames
            Frame indices where to mark timeseries (ts) by label points.
    fps: int
        Frames per second for the movie
    display: bool
        If True, displays movie
    save: bool
        It True, saves movie
    savePath: str
        Path to where the movie should be saved
    yl: array-like, (2,)
        Y-limits for the timeseries
    **kwargs:
        cmap: str, [Default: 'gray']
            Name of python colormap to use for images.
        dpi: int, [Default: 70]
            Dots per inch; movie resolution
    Returns
    -------
    ani: HTML movie object
        Simply typing ani results in playing of the movie        
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    plt.rcParams['animation.ffmpeg_path'] = r'V:\Code\FFMPEG\bin\ffmpeg.exe'
    from IPython.display import HTML
       
    nFrames = images.shape[0]      
    cmap = kwargs.get('cmap', 'gray')
#    interp = kwargs.get('interpolation', 'nearest')
    dpi = kwargs.get('dpi', 70)
    plt.style.use(('seaborn-poster', 'seaborn-white'))
    fh = plt.figure(dpi = dpi)
    ax = [[]]*2
    ax[0] = fh.add_axes([0.1, 0.3, 0.65, 0.65])
    ax[0].set_aspect('equal')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_frame_on(False)
    im = ax[0].imshow(images[0], cmap= cmap, vmin = images.min(),\
                      vmax = images.max())
    
    t = np.arange(0,nFrames)
    ax[1]= fh.add_axes([0, 0, 0.95, 0.29])
    ax[1].set_frame_on(False)
    ax[1].plot(t,ts, c = 'k', lw = 0.5)
    
    if not labels is None:
        ax[1].scatter(t[labels['inds']], ts[labels['inds']], c = labels['labels'],\
                      s= ms, cmap = cmap_lbls)
    ax[1].get_xaxis().set_visible(False)    
    ax[1].set_ylabel('Tail angle')
    ax[1].set_ylim(yl) 
    ax[1].set_xlim(0, len(ts))
    
#    fh.tight_layout()
    
    def update_img(n):        
        im.set_data(images[n])   
        ax[0].set_title('Frame # {}'.format(n))
        ax[1].cla()
        ax[1].plot(t,ts, lw = 1)
        if not labels is None:
            ax[1].scatter(t[labels['inds']], ts[labels['inds']], c = labels['labels'],\
                          s= ms, cmap = cmap_lbls)        
        if n>0:
            ax[1].axvline(t[n], ls = '--', c = 'k', lw = 1.5, alpha = 0.5)
        ax[1].set_xlim(t.min(), t.max())
        
    ani = animation.FuncAnimation(fh, update_img,t, interval= fps, repeat = False)    
    plt.close(fh)
    
    if save:
        print('Saving...')
        writer = animation.writers['ffmpeg'](fps=fps)
        if savePath != None:
            ani.save(savePath, writer = writer, dpi = dpi)
            print('Saved to \n{}'.format(savePath))
        else:
            vidName = f'video_{util.timestamp("m")}.mp4'
            ani.save(vidName, writer=writer, dpi=dpi)
            print('Saved in current drive as \n{}'.format(vidName))
        
    if display:
        print('Displaying...')
        return HTML(ani.to_html5_video())
    else:
        return ani 


def selectFishImgsForUTest(exptDir, prefFrameRangeInTrl = (115,160),\
                              nImgsForTraining:int = 50):
    """ A convenient function for randomly selecting fish images from within a range
    of frames (typically, peri-stimulus to maximize postural diversity) in each trial
    directory of images, and then writing those images in a directory labeled "imgs_train"
    Parameters
    ----------
    exptDir: string
        Path to the root directory where all the fish images (in subfolders whose names
        end with "behav" by my convention) from a particular experimental are located.
    prefFrameRangeInTrl: 2-tuple
        Preferred range of frames for selecting images from within a trial directory.
        Typically, peri-stimulus range.
    nImgsForTraining: integer
        Number of images to select and copy
    overWrite:bool
        If True, overwites existing directory and saves images afresh.
    Returns
    -------
    selectedFiles: list, (nImgsForTraining,)
        List of paths from whence the selected images
    """
    import numpy as np
    import os
    import apCode.FileTools as ft
#    import shutil as sh
#    from apCode.util import timestamp
#    import apCode.volTools as volt
    
    inds_sel = np.arange(*prefFrameRangeInTrl)
    behavDirs = [x[0] for x in os.walk(exptDir) if x[0].endswith('behav')]
    trlDirs = []
    for bd in behavDirs:
        trlDirs.extend(os.path.join(bd, td) for td in ft.subDirsInDir(bd))
    join = lambda p, x: [os.path.join(p,_) for _ in x]
    files_sel =[]
    for td in trlDirs:
        filesInDir = ft.findAndSortFilesInDir(td,ext = 'bmp')
        if len(filesInDir)> np.max(inds_sel):            
            files_sel.extend(join(td, filesInDir[inds_sel]))
        else:
            print(f'{len(filesInDir)} images in {td}, skipped')
    files_sel = np.random.choice(np.array(files_sel), size = nImgsForTraining, replace = False)
    images= volt.img.readImagesInDir(imgPaths = files_sel)
    return images

def singleFishDataFromXls(xlsPath, exptDate, fishId, sessionId = 1, dt_behav = 1/1000,\
                          lpf = 60, nWaves = 3, recompute_dff:bool = True):
    data = FishData(xlsPath, exptDate= exptDate, fishId=fishId, sessionId=sessionId).fetch(recompute_dff = recompute_dff).read_tail_angles()
    data.trialized_dff();
    data = data.read_midlines()
    data = data.correct_tailAngles().filter_denoise_tailAngles(dt = dt_behav, lpf = lpf, nWaves = nWaves)
    data.match_ca_and_behav_trls()
    return data

def stimIDToStimVel(stimID):
    """ Given the stimulus ID timeseries recorded during OKR expts, returns stimulus velocity time series
    Parameters
    ----------
    stimID: array, (N,)
        Stimulus ID timeseries recorded during OKR experiments.
    Returns
    -------
    stimVel: array, (N,)
        Stimulus velocity
    """
    import numpy as np
    from apCode.util import getContiguousBlocks
    from apCode.SignalProcessingTools import standardize
    stimVel = standardize(stimID)-0.5
    zerVec = np.zeros(np.shape(stimVel))
    stimOnInds = np.where(stimVel<=0)[0]
    stimBlocks  = np.array(getContiguousBlocks(stimOnInds))
    stimOnInds_pos = stimBlocks[::2]
    stimOnds_neg = stimBlocks[1::2]
    stimVel = zerVec.copy()
    for inds in stimOnInds_pos:
        stimVel[inds] =1
    for inds in stimOnds_neg:
        stimVel[inds] = -1
    return stimVel

def swimEnvelopes(x, x_svd = None, interp_kind:str = 'slinear'):
    """ 
    Given a timeseries containing swim signals, and optionally,
    the timeseries of SVD coefficients, returns envelopes
    for the timeseries and their derivatives contained in a dictionary.
    Parameters
    ----------
    x: array, (nTimePoints,)
        Swim signals containing timeseries.
    x_svd: array, (nComponents, nTimePoints)
        SVD coefficients timeseries
    interp_kind: str
        Kind of interpolation to use when computing the envelopes. See scipy.interp1d
        for options.
    Returns
    -------
    dic: dict
        Dictionary containing the envelopes with self-explanatory names for the keys
    """
    from apCode.SignalProcessingTools import timeseries, emd
    import numpy as np
    from pandas import DataFrame
    dic = {}
    ### Total curvature envelopes
    xt = timeseries.triplicate(x)
    emd_ = emd.envelopesAndImf(xt, n_comps=1, interp_kind= interp_kind)        
    dic['env_max'] = timeseries.middleThird(emd_['env']['max'])
    dic['env_crests'] = timeseries.middleThird(emd_['env']['crests'])
    dic['env_troughs'] = timeseries.middleThird(emd_['env']['troughs'])
    
    ### Total curvature derivative envelopes
    gx = np.gradient(xt)
    emd_ = emd.envelopesAndImf(gx, n_comps=1, interp_kind=interp_kind)
    dic['env_der'] = timeseries.middleThird(emd_['env']['max'])
    
    if not np.any(x_svd == None):
        ### SVD components envelopes
        xt = np.apply_along_axis(timeseries.triplicate,1,x_svd)
        gxt = np.gradient(xt,axis = 1)
        i = 0
        for xt_, gxt_ in zip(xt, gxt):
            emd_ = emd.envelopesAndImf(xt_,n_comps=1,interp_kind = interp_kind)
            dic[f'env_max_svd{i}'] = timeseries.middleThird(emd_['env']['crests'])
            dic[f'env_min_svd{i}'] = timeseries.middleThird(emd_['env']['troughs'])        
            emd_ = emd.envelopesAndImf(gxt_,n_comps=1,interp_kind = interp_kind)
            dic[f'env_der_svd{i}'] = timeseries.middleThird(emd_['env']['max'])
            i += 1
    dic['idx_sample'] = np.arange(len(x))
    return DataFrame(dic)    

def swimEnvelopes_multiLoc(x, n_loc:int = 5, interp_kind:str = 'cubic',\
                           triplicate:bool = True):
    """
    Given the tail angles array, extracts envelopes for tail angle timeseries
    at specified number of points along the fish, takes their derivatives and
    their 2nd derivatives and puts all tis info in a datframe so that they can
    be used multidimensional features (at each time point) for classification,
    gaussian mixture modeling, dimensionality reduction, etc.
    Parameters
    ----------
    x: array, (nPointsAlongFish, nTimePoints)
        Tail angles array (prefereably cleaned with SVD before; see cleanTailAngles)
    n_loc: int
        Number of points along fish to use in computing features.
    interp_kind: str
        Type of interpolation to use when computing envelopes from tail angle
        timeseries.
    triplicate: bool
        If True, concatenates timeseries back to back to make a triplicate and
        uses this to compute envelope. Suggested when using "cubic" 
        interpolation so as to avoid ugly edge effects.
    Returns
    -------
    features: pandas dataframe, (nTimePoints, nFeatures)
        Pandas dataframe with features as columns.
    """
    import numpy as np
    from apCode.SignalProcessingTools import emd
    import pandas as pd
    from collections import OrderedDict
    if (interp_kind =='cubic') & (not triplicate):
        print('WARNING!: If using "cubic" interpolation, better to set "triplicate = True"')    
    posInds = np.linspace(0,x.shape[0]-1,n_loc+1).astype(int)
    x_sub = np.diff(x[posInds],axis = 0)
    features = []
    for iPos, x_ in enumerate(x_sub):
        e = emd.envelopesAndImf(x_, interp_kind = interp_kind, triplicate=triplicate)
        crests, troughs = e['env']['crests'], e['env']['troughs']        
        dx_ = np.gradient(x_)
        dx_crests = emd.envelopesAndImf(dx_, interp_kind= interp_kind, triplicate=triplicate)['env']['crests']
        dx_troughs = emd.envelopesAndImf(dx_, interp_kind= interp_kind, triplicate=triplicate)['env']['troughs']
        ddx_ = np.gradient(dx_)
        ddx_crests = emd.envelopesAndImf(ddx_, interp_kind=interp_kind, triplicate=triplicate)['env']['crests']
        ddx_troughs = emd.envelopesAndImf(ddx_, interp_kind=interp_kind, triplicate=triplicate)['env']['troughs']
        dic = OrderedDict()
        dic[f'env_crests_pos{iPos}'] =  crests
        dic[f'env_troughs_pos{iPos}']  = troughs
        dic[f'env_crests_der1_pos{iPos}'] =  dx_crests
        dic[f'env_troughs_der1_pos{iPos}']  = dx_troughs
        dic[f'env_crests_der2_pos{iPos}'] =  ddx_crests 
        dic[f'env_troughs_der2_pos{iPos}'] = ddx_troughs
        features.append(pd.DataFrame(data = dic, columns = dic.keys()))
    features = pd.concat(features, axis = 1, join = 'outer', sort = False)
    return features

def swimEnvelopes_multiLoc_polarized(x, n_loc:int = 5, interp_kind:str = 'cubic',\
                           triplicate:bool = True):
    """
    Given the tail angles array, extracts envelopes for tail angle timeseries
    at specified number of points along the fish, takes their derivatives and
    their 2nd derivatives and puts all tis info in a datframe so that they can
    be used multidimensional features (at each time point) for classification,
    gaussian mixture modeling, dimensionality reduction, etc.
    Parameters
    ----------
    x: array, (nPointsAlongFish, nTimePoints)
        Tail angles array (prefereably cleaned with SVD before; see cleanTailAngles)
    n_loc: int
        Number of points along fish to use in computing features.
    interp_kind: str
        Type of interpolation to use when computing envelopes from tail angle
        timeseries.
    triplicate: bool
        If True, concatenates timeseries back to back to make a triplicate and
        uses this to compute envelope. Suggested when using "cubic" 
        interpolation so as to avoid ugly edge effects.
    Returns
    -------
    features: pandas dataframe, (nTimePoints, nFeatures)
        Pandas dataframe with features as columns.
    """
    import numpy as np
    from apCode.SignalProcessingTools import emd
    import pandas as pd
    from collections import OrderedDict
    if (interp_kind =='cubic') & (not triplicate):
        print('WARNING!: If using "cubic" interpolation, better to set "triplicate = True"')    
    posInds = np.linspace(0,x.shape[0]-1,n_loc+1).astype(int)
    x_sub = np.diff(x[posInds],axis = 0)
    features = []
    for iPos, x_ in enumerate(x_sub):
        e = emd.envelopesAndImf(x_, interp_kind = interp_kind, triplicate=triplicate)
        crests, troughs = e['env']['crests'], e['env']['troughs']        
        dx_ = np.gradient(x_)
        dx_crests = emd.envelopesAndImf(dx_, interp_kind= interp_kind,\
                                        triplicate=triplicate)['env']['crests']
        dx_troughs = emd.envelopesAndImf(dx_, interp_kind= interp_kind,\
                                         triplicate=triplicate)['env']['troughs']
        ddx_ = np.gradient(dx_)
        ddx_crests = emd.envelopesAndImf(ddx_, interp_kind=interp_kind,\
                                         triplicate=triplicate)['env']['crests']
        ddx_troughs = emd.envelopesAndImf(ddx_, interp_kind=interp_kind,\
                                          triplicate=triplicate)['env']['troughs']
        dic = OrderedDict()
        dic[f'env_crests_pos{iPos}'] =  crests
        dic[f'env_troughs_pos{iPos}']  = troughs
        dic[f'env_crests_der1_pos{iPos}'] =  dx_crests
        dic[f'env_troughs_der1_pos{iPos}']  = dx_troughs
        dic[f'env_crests_der2_pos{iPos}'] =  ddx_crests 
        dic[f'env_troughs_der2_pos{iPos}'] = ddx_troughs
        features.append(pd.DataFrame(data = dic, columns = dic.keys()))
    features = pd.concat(features, axis = 1, join = 'outer', sort = False)
    return features

def swimOnAndOffsets(x, thr = 10, minOnDur:int =30, minOffDur:int = 100, 
                     use_emd:bool = True):
    """
    Given a timeseries that contains swim signals (total tail curvature, EMG, etc),
    uses EMD-based methods to extract relevant envelopes and estimate the
    onset and offset times of swims. The signal can contain multiple swims
    Parameters
    ----------
    x: timeseries, (nTimePoints,)
        Timeseries with swim signals.
    thr: scalar
        Threshold for detecting swims
    minOnDur: int
        Minimum duration of a swim episodes. This is used to filter out spurious
        transient events that exceed the threshold
    Returns
    -------
    onOffs: array-like, (nSwimEvents,)
        Each element of the array is a 2-tuple holding the swim onset and offset 
        index for a given swim
    """
    from apCode.SignalProcessingTools import emd, levelCrossings
    import numpy as np
    
    if use_emd:
        info = emd.envelopesAndImf(x, n_comps=1, interp_kind='slinear')
        maxEnv = info['env']['max']
#        maxEnv = maxEnv-maxEnv[0]
        ons, offs = levelCrossings(maxEnv, thr=thr)
    else:
        ons, offs = levelCrossings(x, thr = thr)
        
    if (len(ons)==0) | (len(offs)==0):
        return None
    offs = np.delete(offs, np.where(offs < ons[0]))
#    ons = np.delete(ons, np.where(ons > offs[-1]))
    if np.ndim(offs)<1:
        offs = [offs]
    if np.any(ons>offs[-1]):
        offs = np.insert(len(x),0,offs)
    if len(ons) < len(offs):
        ons = np.insert(ons,0,0)
    if len(offs) < len(ons):
        offs = np.insert(len(x),0, offs)    
    onOffs = []
    for on in ons:
        durs = offs-on
        inds = np.where(durs>0)[0]
        if len(inds)>0:
            off = offs[inds[0]]
        else:
            off = len(x)
        onOffs.append((on, off))
        
    onOffs_fin = []
    if len(onOffs)>1:
        curr = onOffs[0]
        i = 1
        for onOff in onOffs[1:]:
            if (onOff[0]-curr[1]) < minOffDur:
                curr = (curr[0], onOff[1])
            else:
                onOffs_fin.append(curr)
                curr = onOff
            i += 1
        onOffs_fin.append(curr)
    else:
        onOffs_fin = onOffs
        
    if len(onOffs_fin)>0:
        return onOffs_fin            
    else:
        return None         

def tailAngles_from_hdf_concatenated_by_trials(hFileDirs, hFileExt = 'h5',\
                                               hFileName_prefix = 'procData',\
                                               key = 'behav/tailAngles', nTailPts = 50):
    """Given a paths to directories for HDF files storing processed behavior data
    are located, extracts tail angles and returns in a dictionary
    Parameters
    ----------
    hFileDirs: list or str
        Path to directory where HDF file with processed data is located or a list
        of such paths
    hFileExt: str
        When locating HDF files in the relevant paths, searches for files with this
        extension.
    hFileName_prefix: str
        When locating HDF files in the relevant paths, searches for files with this
        prefix in their names.
    key: str 
        Path in HDF to tail angles dataset.
    nTailPts: int
        Number of points along the tail in each of the tail angle datasets.
    Returns
    -------
    dic: dictionary
        Dictionary with the following keys
            hFilePath: list
                Paths to the relevant HDF files
            tailAngles: list
                Tail angles extracted from HDF files and reshaped such that
                they are concatenated by trials. Each item in the list corresponds
                to one fish.
    """
    if not (isinstance(hFileDirs, list) | isinstance(hFileDirs,np.ndarray)):
        hFileDirs = [hFileDirs]
    dic = dict(hFilePath = [], tailAngles = [])
    for iDir, hfd in enumerate(hFileDirs):
        hFileName = ft.findAndSortFilesInDir(hfd, ext = hFileExt, search_str = hFileName_prefix)
        if len(hFileName)>0:
            hfp = os.path.join(hfd, hFileName[-1])
            with h5py.File(hfp, mode = 'r') as hFile:
                if key in hFile:
                    print(f'{iDir+1}/{len(hFileDirs)}')
                    ta = np.array(hFile[key])
                    trlLen = ta.shape[-1]
                    ta = ta.reshape(-1,nTailPts, ta.shape[-1])
                    ta_ser = np.concatenate(ta,axis = 1)
                    dic['hFilePath'].append(hfp)
                    dic['tailAngles'].append(ta_ser)
    return dic    

def tailAnglesFromRawImagesUsingUnet(I, uNet, imgExt = 'bmp', filtSize = 2.5, 
                                   smooth = 20, kind = 'cubic', n = 50,
                                   otsuMult = 1, verbose =0):
        """
        Given an array of images or the path to a directory of images (with 
        single fish per image), segments the fish using a trained U net, and
        extracts midlines from them.
        Parameters
        ----------
        I: array (T, M, N) or string
            Stack of raw images with a single fish per image or path string to 
            the directory of images.
        uNet: Keras neural network object
            Trained U net model for segmenting fish.
        imgExt: string
            Filters for images with this extension
        filtSize: scalar
            Size of convolution kernel used to smooth images for detecting and
            possibly coalescing fish blobs
        smooth: scalar
            Smoothing factor to apply to raw midlines extracting from image 
            using "thinning" procedure
        fill_value: scalar or string
            Value to fill with when extrapolating midlines. If None then fills
            with NaNs (although this could lead to error in the current 
            implementation). If, constant then uses this value, and if 
            "extrapolate" then extrapolates using scipy.interpolate.interp1d. 
            There is also a second step of interpolation across fish length and
            time points and that is done using scipy.interpolate.griddata
        kind: string
            Kind of interpolation, default is 'cubic'. 
            See scipy.interpolate.interp1d
        n: integer
            Number of points constituting the fish midline.
        verbose: bool
            If 1/True, then print progress messages
        Returns
        -------
        out: dict
            Dictionary with the follow key, values:
            midlines: array, (T, N, 2)
                Array of midlines. The 2 dimensions of the 3rd axis of midlines
                corresponds to the x- and y coordinates respectively.
            kappas: array, (N,T)
                Curvatures at N points along the midline.
            I_prob: array, (T, M, N)
                Probability images predicted by the U net
        """
        import os
        import numpy as np
        import apCode.behavior.FreeSwimBehavior as fsb
#        import apCode.volTools as volt
        from apCode import geom
        from dask import delayed, compute
        
        if isinstance(I, str):
            if os.path.exists(I):
                I = volt.img.readImagesInDir(I, ext= imgExt)
            else:
                print("Image path not found, check path!")
                return None
        if np.ndim(I)==2:
            I = I[np.newaxis,...]
    
        if verbose:
            print('Predicting on images...')
        I_prob = np.squeeze(uNet.predict(fsb.prepareForUnet_1ch(I,sz=uNet.input_shape[1:3])))
        I_prob = volt.img.resize(I_prob, I.shape[1:], preserve_dtype = True, preserve_range = True)
    
        if verbose:
            print('Processing images for midline detection...')
        I_fish = fishImgsForMidline(I_prob, filtSize = filtSize, otsuMult = otsuMult)        
    
        if verbose:
            print('Computing midlines...')
        midlines = midlinesFromImages(I_fish)[0]
    
        if verbose:
            print('Interpolating midlines to sample uniformly to the same length...')
        print('2D interpolation...')    
        midlines_interp = geom.interpolateCurvesND(midlines,mode = '2D', N= 50)
        print('Curve smoothening...')
        midlines_interp = np.asarray(compute(*[delayed(geom.smoothen_curve)(_, smooth = smooth) for _ in midlines_interp], scheduler = 'processes'))
        print('Length equalization')
        midlines_interp = geom.equalizeCurveLens(midlines_interp)
                
        if verbose:
            print('Computing curvatures...')
        kappas = fsb.track.curvaturesAlongMidline(midlines_interp, n = n)
        tailAngles = np.cumsum(kappas,axis = 0)
        midlines = dict(raw = midlines, interp = midlines_interp)
        out = dict(midlines = midlines, I_fish = I_fish, I_prob = I_prob,\
                   tailAngles = tailAngles)
        return out
    
def toRatioDff(x, trlLen:int = 550, nPre:int = 50, q:float = 20, nWave = 2, fitFirst:bool = True): 
    """
    Given the timeseries for the indicator and dye for a given ROI or such, returns
    the ratiometric dff
    Parameters
    ----------
    x: array, (2,T)
        Timeseries array of the dye (AF 405 in my case) and indicator (Cal 590) where
        the first row is the dye
    trlLen: int
        Length of a single trial in number of image frames
    nPre: int
        Number of points in the trial that occur before stim. These are used for calculating
        the specified percentile by which the signals are normalized
    q: scalar
        The percentile to use to compute baseline
    nWave: int or None
        The parameter n in my function spectral.Wden.wden, which determines the smoothing of
        the signals by wavelets
    Returns
    -------
    y: array
        Indicator signal as dF/F
    x_perc: percentile normalized indicator and dye signals
    """
    import numpy as np
    from apCode.spectral.WDen import wden
    from apCode import geom    
    
    trialize = lambda x, trlLen: x.reshape(len(x)//trlLen, trlLen)
    percNormalize = lambda x, axis, q, nPre :(x/np.expand_dims(np.apply_along_axis(np.percentile,axis,x[:,:nPre],q),axis)).flatten()
    x = x.copy()
    if fitFirst:
        fit = geom.fitLine(x.T)[0][:,1]
        x[0]= fit
#        p = np.expand_dims(np.apply_along_axis(np.percentile,1,x,q),1)
#        x = x/p
    
    x_perc = np.array([np.array([percNormalize(trialize(r,trlLen),1,q,nPre) for r in x])]).transpose(1,0,2)
    x_perc = np.squeeze(x_perc)
    if not nWave == None:
        x_perc[1] = wden(x_perc[1], n = nWave)
        x_perc[0] = wden(x_perc[0], n = nWave + 1)
    dff = (x_perc[1]/x_perc[0]) -1
    return dff, x_perc

    
    
    
    
