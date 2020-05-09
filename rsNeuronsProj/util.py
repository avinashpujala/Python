
import numpy as np
import os
import sys
from dask import delayed, compute
import pandas as pd

sys.path.append(r'\\dm11\koyamalab\code\python\code')
import apCode.behavior.FreeSwimBehavior as fsb # noqa E402
# from apCode.tanimotom import besselImaging as bing # noqa E402

def add_suffix_to_paths(paths, suffix='proc'):
    """
    Go through list of paths, adding the specified suffix ending if not
    already present
    Parameters
    ----------
    paths: str or list-like
        Path or list of paths
    """
    if (isinstance(paths, list) | isinstance(paths, np.ndarray)):
        paths_new = []
        for path_ in paths:
            paths_new.append(add_suffix_to_paths(path_))
    else:
        pre, post = os.path.split(paths)
        if post != suffix:
            paths_new = os.path.join(paths, suffix)
        else:
            paths_new = paths
    return np.array(paths_new)


def append_fishPos_to_xls(xls):
    """Given the xls dataframe, reads the HDF files at the end of the paths in
    the datrame, extracts fish position and other relevant information and
    appends this to the dataframe
    Parameters
    ----------
    xls: pandas dataframe
        The dataframe which contains paths (xls.Path) to the HDF files with
        fish position information
    Returns
    -------
    df: pandas dataframe
        Dataframe that still contains the paths to hdf files, but that also has
        additional information about fish position, etc
    """
    from apCode.FileTools import findAndSortFilesInDir

    def vel_from_fp(fp, dt):
        ds = (np.diff(fp, axis=1)**2).sum(axis=0)**0.5
        return ds/dt

    def cum_dist_from_fp(fp):
        dfp = np.sum(np.diff(fp, axis = 1)**2, axis=0)**0.5
        cfp = np.insert(np.cumsum(dfp), 0, 0)
        return cfp

    def cum_disp_from_fp(fp):
        dfp = (fp-fp[:, 0][:, np.newaxis])**2
        dfp = np.sqrt(dfp.sum(axis=0))
        return np.insert(dfp, 0, 0)

    def total_dist_from_cumDist(cfp):
        return np.round(cfp[-1]*10)/10

    def total_disp_from_cumDisp(cfp):
        return np.round(cfp[-1]*10)/10


    if 'Path_proc' not in xls.columns:
        paths = np.array(xls.Path)
        paths_deproc = remove_suffix_from_paths(paths, suffix='proc')
        paths_proc = add_suffix_to_paths(paths_deproc, suffix='proc')
        xls = xls.assign(Path_proc=paths_proc)
    df = []
    for iRow in range(xls.shape[0]):
        row = xls.iloc[iRow]
        fName = findAndSortFilesInDir(row.Path_proc, ext='mat')
        if len(fName)>0:
            with fsb.openMatFile(row.Path_proc, mode='r') as hFile:
                if 'fishPos' in hFile:
                    fp = np.array(hFile['fishPos'])
                else:
                    print('"fishpos" key not found')
                    fp = None
                if 'fps' in hFile:
                    fps = int(np.array(hFile['fps'])[0, 0])
                else:
                    print('"fps" key not found, setting to 500')
                    fps = 500
                if 'nFramesInTrl' in hFile:
                    nFramesInTrl = int(np.array(hFile['nFramesInTrl'])[0, 0])
                else:
                    print('"nFramesIntrl" key not found, setting to 750')
                    nFramesInTrl = 750
                if fp is not None:
                    pxlSize = row.pxlSize
                    dt = 1/fps
                    if np.mod(fp.shape[1], nFramesInTrl) != 0:
                        N = int((fp.shape[1]//nFramesInTrl)*nFramesInTrl)
                        fp = fp[:, N]
                    fp_trl = fp.reshape(fp.shape[0], -1, nFramesInTrl)
                    fp_trl = np.transpose(fp_trl, (1, 0, 2))
                    # --- Set first points of fish trajectory to zero
                    fp_start = fp_trl[...,0]
                    fp_trl = fp_trl - fp_start[:, :, np.newaxis]
                    # --- Change from pixel to physical units
                    fp_trl = fp_trl*pxlSize
                    cumDist_trl = [delayed(cum_dist_from_fp)(_) for _ in
                                   fp_trl]
                    cumDist_trl = compute(*cumDist_trl)
                    cumDisp_trl = [delayed(cum_disp_from_fp)(_) for _ in
                                   fp_trl]
                    cumDisp_trl = compute(*cumDisp_trl)
                    totDist_trl = [delayed(total_dist_from_cumDist)(_)
                                   for _ in cumDist_trl]
                    totDist_trl = compute(*totDist_trl)
                    totDisp_trl = [delayed(total_disp_from_cumDisp)(_)
                                   for _ in cumDisp_trl]
                    totDisp_trl = compute(*totDisp_trl)
                    swimVel_trl = [delayed(vel_from_fp)(_, dt) for _ in fp_trl]
                    swimVel_trl = compute(*swimVel_trl)
                    swimVel_trl_max = np.array(swimVel_trl).max(axis=1)
                    trlNums = np.arange(len(fp_trl))
                    fp_trl = list(fp_trl)
                    fp_start = list(fp_start)
                    dic = dict(fishPos_mm=fp_trl, fishPos_start=fp_start,
                               swimDist=cumDist_trl, swimDisp=cumDisp_trl,
                               swimDist_total=totDist_trl,
                               swimDisp_total=totDisp_trl, trlNum=trlNums,
                               fps=fps, nFramesInTrl=nFramesInTrl,
                               swimVel=swimVel_trl,
                               swimVel_max=swimVel_trl_max)
                    df_fish = pd.DataFrame(dic, columns=dic.keys())
                    df_fish = df_fish.assign(FishIdx=row.FishIdx)
                    df_fish = pd.merge(row.to_frame().T, df_fish, how='outer')
                    df.append(df_fish)
        else:
            pass
    return pd.concat(df)

def append_latency_to_df(df, var='swimVel', nKer=100, ind_start=50, zThr=1,
                         dt=1/500):
    """
    Parameters
    ----------
    df: pandas dataframe
        The number of rows of df must correspond to the total number of trials
        because the program assumes this when computing onset latencies
    """
    from apCode.machineLearning.preprocessing import Scaler
    from apCode.SignalProcessingTools import causalConvWithSemiGauss1d
    from apCode.SignalProcessingTools import levelCrossings

    swimVel = np.array([np.array(_) for _ in df[var]])
    nTrls = len(swimVel)
    swimVel_ser = causalConvWithSemiGauss1d(swimVel.flatten(), n=nKer)
    scaler = Scaler(with_mean=False)
    swimVel_trl = scaler.fit_transform(swimVel_ser[:, None]).reshape(nTrls, -1)
    lats = np.ones((nTrls, ))*np.nan
    # latsToFirstPk = lats.copy()
    for iTrl, trl in enumerate(swimVel_trl):
        onInds = levelCrossings(trl, thr=zThr)[0]
        if len(onInds)>0:
            # inds_keep = np.where(onInds >= ind_start)[0]
            # if len(inds_keep) > 0:
            #     onInd = onInds[inds_keep][0]
            #     lats[iTrl] = onInd-ind_start
            lats[iTrl] = 1000*(onInds[0]-ind_start)*dt # In milliseconds
    df = df.assign(onsetLatency=lats)
    return df

def apply_to_imgs_and_save(imgDir, func='fliplr', splitStr='Fish'):
    """
    Apply some function to images in specified directory and save.
    After function is applied the output must still be 2D/3D.
    Parameters
    ----------
    imgDir: str
        Image directory
    func: str or function
        Function to apply to images. If str, then must be a function in
        numpy
    splitStr: str
        Substring where to split path. The images will be saved at the level
        above where the path is split.
    Returns
    -------
    saveDir: str
        Directory where images were saved
    """
    from apCode import util
    import apCode.FileTools as ft
    import apCode.volTools as volt

    if isinstance(func, str):
        func = eval(f'np.{func}')
    strList = os.path.abspath(imgDir).split("\\")
    ind = util.findStrInList(splitStr, strList, case_sensitive=False)[0]
    strRep = strList[ind]
    rootDir = os.path.join(*np.array(strList)[:ind])
    fn = ft.subDirsInDir(rootDir)
    nDir = len(fn)
    sfx = strRep.replace(splitStr, f'{splitStr}{nDir+1}')
    saveDir = os.path.join(rootDir, sfx)
    os.makedirs(saveDir, exist_ok=True)
    print('Reading images into dask array...')
    imgs = volt.dask_array_from_image_sequence(imgDir)
    print(f'{imgs.shape[0]} images!')
    blockSize= np.minimum(750, (imgs.shape[0]//10)+1)
    print(f'Block size = {blockSize}')
    print(f'Transforming and saving images to\n{saveDir}')
    inds = np.arange(len(imgs))
    inds_list = ft.sublistsFromList(inds, blockSize)
    nBlocks = len(inds_list)
    for iBlock, inds_ in enumerate(inds_list):
        print(f'Block # {iBlock+1}/{nBlocks}')
        inds_now = np.array(inds_)
        imgs_ = imgs[inds_now].compute()
        imgs_ = np.array([func(img) for img in imgs_])
        imgNames = [r'f{}_{:06d}.bmp'.format(nDir+1, ind) for ind in inds_now]
        volt.img.saveImages(imgs_, imgDir=saveDir, imgNames=imgNames)
    return saveDir



def bootstrap_df(df, keys, vals, mult=2):
    """Bootstrap subset of dataframe filtered by the specified
    keys and values
    Parameters
    ----------
    df: pandas datframe
    keys, vals: lists of key and values
        These are used to filter the dataframe and boostrap only that subset.
        For e.g., if keys = ['ablationGroup', 'Treatment'] and vals = ['mHom', 'ctrl']
        then df_sub = df.loc[(df.ablationGroup=='mHom') & (df.Treatment=='ctrl')].
        Then, df_sub is bootstrap so that length is multiplied by mult and this
        is placed back into the original datframe, which is returned
    mult: scalar
        Multiplication factor for bootstrapping, i.e.after ootstrap df_sub
        will have mult times as many rows as before bootstrapping.
    Returns
    -------
    df_bs: pandas dataframe
        New longer dataframe including the boostrapped subset
    """
    df_now = df.copy()
    inds=np.arange(df_now.shape[0])
    for key, val in zip(keys, vals):
        inds_ = np.where(df_now[key]==val)[0]
        inds=np.intersect1d(inds, inds_)
    n = len(inds)*mult
    d = n-len(inds)
    inds_bs = np.random.choice(inds, size=d, replace=True)
    inds_bs = np.r_[inds, inds_bs]
    df_bs = df.iloc[inds_bs]
    inds_rest = np.setdiff1d(np.arange(len(df)), inds)
    df_rest = df.iloc[inds_rest]
    df_bs = pd.concat((df_rest, df_bs), ignore_index=True)
    return df_bs


def detect_noisy_trials(df, nKer : int = 100,
                        var : str = 'swimVel',
                        convolve : bool = True):
    """Use a Gaussian Mixture Model to identify noisy points in trials
    and eliminate noisy trials from the dataframe
    Parameters
    ----------
    df: pandas dataframe
        Each row of the dataframe must correspond to a single trial
    nKer: int
        Kernel length for smoothing the relevant timeseries before fitting a
        GM model
    Returns
    -------
    df_new: pandas dataframe
        New dataframe with additional column f"noisyTrlInds_{var}" which is
        a boolean with values of 1 corresponding to trials identified as noisy.
    """
    from apCode.machineLearning.ml import GMM
    from apCode.SignalProcessingTools import causalConvWithSemiGauss1d
    swimVel = np.array([np.array(_) for _ in df[var]])
    nTrls = len(swimVel)
    swimVel_ser = swimVel.flatten()
    if convolve:
        swimVel_ser = causalConvWithSemiGauss1d(swimVel_ser, n=nKer)
    swimVel_trl = swimVel_ser.reshape(nTrls, -1)
    print('Fitting GMM model for detection of noisy trials...')
    gmm = GMM(n_components=3).fit(swimVel_ser[:, None])
    noiseLbl = np.argmax(gmm.means_)
    noisyTrlInds = np.zeros((nTrls, ))
    for iTrl, trl in enumerate(swimVel_trl):
        lbls = gmm.predict(trl[:, None])
        if len(np.intersect1d([noiseLbl], lbls)) > 0:
            noisyTrlInds[iTrl] = 1
    kwargs = {f'noisyTrlInds_{var}' : noisyTrlInds}
    df = df.assign(**kwargs)
    return df

def expand_on_trls(df_fish, trlLen=750):
    """
    Takes dataframe whose each row contains a single fish data
    and expands into a larger dataframe wherein each row contains
    single trial data
    Parameters
    ----------
    df_fish: pandas dataframe, (nFish, nVariables)
    trlLen: int
        Number of samples in a single trial
    Returns
    -------
    df_trl: pandas dataframe, (nTrlsInTotal, nVariables)
        Expanded dataframe
    """
    df_trl = []
    for fi in np.unique(df_fish.FishIdx):
        df_now = df_fish.loc[df_fish.FishIdx==fi]
        ta_ = np.array(df_now.iloc[0].tailAngles)
        nTrls = ta_.shape[1]//trlLen
        ta_trl = np.hsplit(ta_, nTrls)
        dic = dict(trlIdx = np.arange(nTrls), FishIdx=[fi]*nTrls, tailAngles=ta_trl)
        df_trl.append(pd.DataFrame(dic))
    df_trl = pd.concat(df_trl, ignore_index=True)
    df_trl = df_trl.assign(trlIdx_glob=np.arange(df_trl.shape[0]))
    foo = df_fish.drop(columns=['tailAngles', 'tailAngles_tot'])
    return pd.merge(foo, df_trl, on='FishIdx')


def expand_on_bends(df_trl, Fs=500, tPre_ms=100, bendThr=10, minLat_ms=5,
                    maxGap_ms=100):
    """Takes dataframe where each row contains single trial information and
    expands such that each row contains single bend information
    Parameters
    ----------
    df_trl: pandas dataframe, (nTrlsInTotal, nVariables)
    Fs: int
        Sampling frequency when collecting data(images)
    nPre_ms: scalar

    """
    import apCode.SignalProcessingTools as spt
    minPkDist = int((10e-3)*Fs)
    nPre = tPre_ms*1e-3*Fs
    minLat = minLat_ms*1e-3*Fs
    maxGap = maxGap_ms*1e-3*Fs
    df_bend=[]
    for iTrl in np.unique(df_trl.trlIdx_glob):
        df_now = df_trl.loc[df_trl.trlIdx_glob==iTrl]
        y = df_now.iloc[0]['tailAngles'][-1]
        y = spt.chebFilt(y, 1/Fs, (5, 60), btype='bandpass')
        pks = spt.findPeaks(y, thr=bendThr, thrType='rel', pol=0,
                            minPkDist=minPkDist)[0]
        if len(pks)>3:
            dpks = np.diff(pks)
            tooSoon = np.where(pks<(nPre+minLat))[0]
            tooSparse = np.where(dpks>maxGap)[0]+1
            inds_del = np.union1d(tooSoon, tooSparse)
            pks = np.delete(pks, inds_del, axis=0)
        if len(pks)>3:
            nBends = len(pks)
            bendIdx = np.arange(nBends)
            bendSampleIdxInTrl=pks
            bendAmp = y[pks]
            bendAmp_abs = np.abs(bendAmp)
            bendAmp_rel = np.insert(np.abs(np.diff(bendAmp)), 0, bendAmp[0])
            bendInt_ms = np.gradient(pks)*(1/Fs)*1000
            onset_ms = (pks[0]-nPre+1)*(1/Fs)*1000
        else:
            nBends=0
            bendIdx, bendAmp, bendAmp_abs, bendAmp_rel, bendInt_ms =\
                [np.nan for _ in range(5)]
            bendsampleIdxInTrl, onset_ms = [np.nan for _ in range(2)]
        dic = dict(trlIdx_glob=iTrl, nBends=nBends, bendIdx=bendIdx,
                   bendSampleIdxInTrl=bendSampleIdxInTrl, bendAmp=bendAmp,
                   bendAmp_abs=bendAmp, bendAmp_rel=bendAmp_rel,
                   bendInt_ms=bendInt_ms, onset_ms=onset_ms)
        df_now = pd.DataFrame(dic)
        df_bend.append(df_now)
    df_bend = pd.concat(df_bend, ignore_index=True)
    return pd.merge(df_trl, df_bend, on='trlIdx_glob')



def get_img_props(paths_to_imgs, diam=50, plotBool: bool = False,
                  verbose: bool = True, inParallel: bool = True,
                  override_and_save: bool = True, **kwargs):
    """
    Estimate pixel sizes from background images computed from the images in
    the given paths.
    Parameters
    ----------
    paths_to_imgs: str or list of string
        Path or List of paths to images
    diam: scalar
        Actual diameter of arena edge within the images in any units of
        interest
    plotBool: bool
        If True, then plots points on the detected arena edge
    verbose: bool
        If True, then displays a progress bar
    override_and_save: bool
        If True, recomputes background image regardless of whether or
        not one already exists in the path (slower)
    **kwargs: dict
        Keyword args for FreeSwimBehavior.getArenaEdge
        nIter, tol
    Returns
    -------
    pxlSizes: (nPaths_to_images,)
        Pixel sizes in given units.
    imgDims: tuple, (2,)
        Image dimensions
    """
    if isinstance(paths_to_imgs, str):
        bgd = fsb.track.computeBackground(paths_to_imgs,
                                          override_and_save=override_and_save)
        pxlSizes = fsb.getPxlSize(bgd, diam=50, plotBool=plotBool, **kwargs)[0]
        img_props = (pxlSizes, bgd.shape)
    else:
        if inParallel:
            del_func = delayed(get_img_props)
            img_props = [del_func(path_, diam=50, plotBool=plotBool, **kwargs)
                         for path_ in paths_to_imgs]
            if verbose:
                from dask.diagnostics import ProgressBar
                with ProgressBar():
                    img_props = compute(*img_props, scheduler='processes')
            else:
                img_props = np.array(compute(*img_props,
                                             scheduler='processes'))
        else:
            img_props = []
            for iPath, path_ in enumerate(paths_to_imgs):
                print(f'Path # {iPath+1}/{len(paths_to_imgs)}')
                try:
                    img_props.append(get_img_props(path_, diam=50,
                                                   plotBool=plotBool,
                                                   **kwargs))
                except RuntimeError:
                    print(f'Failed for path:\n {path_}')
    return img_props


def remove_suffix_from_paths(paths, suffix='proc'):
    """
    Filter a list of paths, removing the specified suffix ending if present
    Parameters
    ----------
    paths: str or list-like
        Path or list of paths
    """
    if (isinstance(paths, list) | isinstance(paths, np.ndarray)):
        paths_new = []
        for path_ in paths:
            paths_new.append(remove_suffix_from_paths(path_))
    else:
        pre, post = os.path.split(paths)
        if post == suffix:
            paths_new = pre
        else:
            paths_new = paths
    return np.array(paths_new)

def shuffle_trls(df_trl, grpNames):
    """
    Shuffle trials for given group names
    Parameters
    ----------
    df_trl: pandas dataframe, (nTrlsInTotal, nVariables)
        Dataframe where each row corresponds to a single trial
    grpNames: list of strings
        Names of ablation groups whose trials are to be shuffled
    """
    df_grp = df_trl[df_trl.AblationGroup.isin(grpNames)]
    df_notGrp = df_trl[df_trl.AblationGroup.isin(grpNames)==False]
    inds = np.arange(df_grp.shape[0])
    ta_ = np.array(df_grp.tailAngles)
    np.random.shuffle(inds)
    ta_shuf = ta_[inds]
    df_grp = df_grp.assign(tailAngles=ta_shuf)
    df_shuf = pd.concat((df_notGrp, df_grp))
    return df_shuf


