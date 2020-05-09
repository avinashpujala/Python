# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:00:48 2015

@author: pujalaa
"""
import tifffile as tff
import os
import sys
import shutil as sh
import numpy as np
import glob
sys.path.append(r'v:/code/python/code')
from apCode import util  # noqa: E402


def deleteFilesWithoutExtension(inputDir, ext):
    '''
    Given the path to a directory and an extension, deletes all files without
    the specified extension
    '''
    filesInDir = findAndSortFilesInDir(inputDir)
    filesInDir_ext = findAndSortFilesInDir(inputDir, ext=ext)
    filesInDir_woExt = np.setdiff1d(filesInDir, filesInDir_ext)
    for file in filesInDir_woExt:
        os.remove(os.path.join(inputDir, file))


def dissolveFolders(supraDir, ext=''):
    '''
    Given a supradirectory containing subdirectories, dissolves all the
    subdirectories such that the files within them are now in the
    supradirectory. Specifying image extension, only results in moving out of
    all files with the specified extension.
    On 20190316 removed parallel processing input option because this won't
     work and leads to errors.
    '''
    fldrs = subDirsInDir(supraDir)

    [moveFilesUpOneLevel(os.path.join(supraDir, fldr),
                         ext=ext, processing='serial') for fldr in fldrs]

    [os.rmdir(os.path.join(supraDir, fldr))
     for fldr in fldrs if len(os.listdir(os.path.join(supraDir, fldr))) == 0]


def evenlyDiviseDir(inputDir, chunkSize, ext='', remove=False):
    '''
    Given the path to a directory, removes extra files after sorting and evenly
    dividing total # of files in the directory with specified chunkSize. Can
    specify an extension to only consider files with the given extension in
    the first place
    Parameters:
    inputDir - String, path to directory containing files
    chunkSize - Scalar, the size of the chunks into which the total number of files
        of interest in the input directory must evenly divide into. The extras are
        either removed or moved
    ext - String, the extension of the files of interest. E.g. 'bmp' or 'tif' for
        images. Only checks these files and moves or removes them.
    remove - Boolean. If true, deletes the extra files, or else moves them into a
        subfolder with the name "extra files".

    '''
    filesInDir = findAndSortFilesInDir(inputDir, ext=ext)
    nFilesInDir = len(filesInDir)
    nFilesToKeep = np.floor(len(filesInDir)/chunkSize)*chunkSize
    filesInDir_del = filesInDir[np.arange(nFilesToKeep+1, nFilesInDir+1).astype('int')-1]
    if remove == 0:
        srcDir = os.path.join(inputDir, 'extra files')
        if os.path.exists(srcDir) == 0:
            os.mkdir(srcDir)

    for file in filesInDir_del:
        if remove:
            os.remove(os.path.join(inputDir, file))
        else:
            src = os.path.join(inputDir, file)
            dst = os.path.join(srcDir, file)
            sh.move(src, dst)
    if remove:
        print('Deleted', len(filesInDir_del), 'files!')
    else:
        print('Moved', len(filesInDir_del), 'files!')


def findAndSortFilesInDir(fileDir, ext=None, search_str=None):
    '''
    Finds files in a specified directory with specified extension and/or
    string in name.
    '''
    if (ext == None) & (search_str == None):
        filesInDir = np.sort(os.listdir(fileDir))
    elif (ext != None) & (search_str == None):
        filesInDir = np.sort([f for f in os.listdir(fileDir) if
                              f.endswith(ext)])
    elif (ext == None) & (search_str != None):
        filesInDir = np.sort([f for f in os.listdir(fileDir) if
                              f.find(search_str) != -1])
    else:
        filesInDir = np.sort([f for f in os.listdir(fileDir) if
                              (f.endswith(ext) & (f.find(search_str) != -1))])
    return filesInDir


def getFilteredPathsFromXls(xlsDir, xlsName, sheetNum=1):
    """
    Works similar to the eponymous MATLAB script. When given the path to the
        excel file containing ablation data locations and other info, the name
        of the file and the sheet number, then asks user for parameters of
        interest, and then returns a list of paths
    """
    import pandas as pd
    xl = pd.ExcelFile(os.path.join(xlsDir, xlsName))
    xl = xl.parse(xl.sheet_names[sheetNum])
    # print(xl.head(5))
    print(xl.columns)
    filtInds = eval(input("Enter the col numbers to filter by, e.g., [0, 2, 4] \n"))

    valInds, vals = [], []
    if isinstance(filtInds, list) == 0:
        filtInds = [filtInds]
    for ind in filtInds:
        colName = xl.columns[ind]
        vals_unique = np.unique(xl[colName])
        if len(vals_unique) > 1:
            print(colName, vals_unique, '\n')
            val_inds = eval(input('Enter index of variable of interest: \n'))
            valInds.append(val_inds)
            vals.append(vals_unique[val_inds])
        else:
            valInds.append(0)
            vals.append(vals_unique[0])
    colNames = xl.columns[filtInds]
    if isinstance(colNames, str):
        colNames = [colNames]
    inds = np.arange(len(xl)+1)
    for ccNum, cc in enumerate(colNames):
        foo = xl[cc]
        iterable = vals[ccNum]
        if np.sum(np.shape(iterable)) == 0:
            iterable = list(np.reshape(iterable, 1))
        for vv in iterable:
            inds_now = np.where(foo == vv)[0].ravel()
            inds = np.intersect1d(inds, inds_now.ravel())
    pathList = np.array(xl.iloc[inds, -1])
    return pathList


def getNumStamps(fileNames):
    """
    Given a list of file names with number stamps on them, returns the number
    stamps so that they can be checked for continuity, or some such thing.
    Assumes that the largest contiguous block of numbers in the file name is
    the number stamp
    """

    from apCode import util
    fn = fileNames[0]
    digInds = np.zeros(len(fn))
    for lNum, let in enumerate(fn):
        if let.isdigit():
            digInds[lNum] = 1
    digInds = np.where(digInds)[0]
    blocks = util.getContiguousBlocks(digInds)
    blockLens = [len(block) for block in blocks]
    block = blocks[np.argmax(blockLens)]

    numStamps = [int(fn[block[0]:block[-1]+1]) for fn in fileNames]
    return tuple(numStamps)


def moveFilesInSubDirsToRoot(rootDir, ext: str = 'tif', copy: bool = False):
    """
    Move or copy files with matching extension in all of the subdirectories
    of the root directory to the root directory. Prefixes are added to the
    files to prevent name clash and to indicate their original locations.
    Parameters
    ----------
    rootDir: str
        Root directory in which to search for and move files.
    ext: str
        Extension of the files to search for.
    copy: boolean
        If True, copies files instead of moving.
    """
    from apCode import util

    def getRootsAndFiles(rootDir, ext):
        r, f = [], []
        for root, dirs, files in os.walk(rootDir):
            inds = util.findStrInList('.'+ext, files)
            if (len(inds) == len(files)) & (len(inds) > 0):
                r.append(root)
                f.append(files)
        return r, f

    def getPrefix(path_now, rootDir):
        prfx = ''
        while path_now != rootDir:
            path_now, sfx = os.path.split(path_now)
            prfx = sfx + '_' + prfx
        return prfx

    ext = ext.split('.')[-1]
    roots, files = getRootsAndFiles(rootDir, ext)
    prefixes = [getPrefix(root, rootDir) for root in roots]
    src, dst = [], []
    for p, f, r in zip(prefixes, files, roots):
        foo = [os.path.join(r, f_) for f_ in f]
        src.extend(foo)
        foo = [os.path.join(rootDir, p + f_) for f_ in f]
        dst.extend(foo)
    for src_, dst_ in zip(src, dst):
        if copy:
            sh.copy(src_, dst_)
        else:
            sh.move(src_, dst_)
    return src, dst


def moveFilesUpOneLevel(srcDir, ext='', processing='parallel'):
    '''
    Moves files in the srcDir with specified extension up one folder level
    Parameters:
    srcDir - String; source directory in which to files and move up one level
    ext - String; extension of the files to find in source directory
    processing - String, 'parallel' or 'serial'. If parallel, processes in
    parallel
    '''
    import time

    numCores = 32
    dst = os.path.split(srcDir)[0]
    tic = time.time()
    imgsInDir = findAndSortFilesInDir(srcDir, ext=ext)
    if processing.lower() == 'parallel':
        from joblib import Parallel, delayed
        import multiprocessing as mp
        numCores = min([mp.cpu_count(), numCores])
        Parallel(n_jobs=numCores, verbose=5)(delayed(sh.move)(src, dst) for src in imgsInDir)
    else:
        for img in imgsInDir:
            sh.move(os.path.join(srcDir, img), dst)
    print(int(time.time()-tic), 'sec')


def openPickleFile(pathToFile):
    import pickle
    with open(pathToFile, mode='rb') as f:
        data = pickle.load(f)
    return data


def subDirsInDir(inDir):
    allInDir = os.listdir(inDir)
    subDirs = list(np.array(allInDir)[np.where([os.path.isdir(os.path.join(inDir, f))
                                                for f in allInDir])[0]])
    return subDirs


def recursively_find_paths_with_searchStr(searchDir, searchStr):
    """ Walks down the directory tree for a specified
    search directory and returns the paths to all files or folders
    with specified search string.
    Parameters
    ----------
    searchDir: str
        Path to search directory
    searchStr: str
        Search string to use in looking for files/folders
    Returns
    -------
    paths: list or str
        List of paths returned by the search
    """
    roots, dirs, files = zip(*[out for out in os.walk(searchDir)])
    inds = util.findStrInList(searchStr, roots)
    return np.array(roots)[inds]

def rename_files(fileDir, pre_str, post_str):
    """
    Renames files in specified directory by replacing the substring pre_str
    in the file name with the substring post_str
    """
    import dask
    from dask.diagnostics import ProgressBar
    def replaceAndMove(src, pre_str, post_str):
        dst = src.replace(pre_str, post_str)
        sh.move(src, dst)
    filePaths = glob.glob(os.path.join(fileDir, f'*{pre_str}*.*' ))
    foo = [dask.delayed(replaceAndMove)(fp, pre_str, post_str) for
           fp in filePaths]
    with ProgressBar():
        dask.compute(*foo)
    return fileDir

def scanImageTifInfo(tifDirOrPaths, searchStr='', n_jobs=32, verbose=0):
    """
    Returns some useful info about .tif files in a folder collected with
    ScanImage. To do this quickly, reads scanImage metadata without loading
    images.
    Parameters
    ----------
    tifDirOrPaths: string or list/array of strings
        Path to the directory containing the .tif files or a list/array of
        paths to the .tif files
    searchStr: string
        Filter the .tif files by looking for this string token in file names.
        Only valid when tifDirOrPath is a directory.
    n_jobs, verbose: see Parallel, delayed from joblib
    Returns
    -------
    tifInfo: dict
        Dictionary with the following keys:
        filePaths: array
            Paths to the files
        nChannelsInFile: array
            Number of image channels in each tif file
        nImagesInFile: array
            Number of images in each tif file
    """
    from apCode import util

    def tifInfo(filePath):
        """
        Given the path to a tif file recorded returns the number of image
        frames within as well as number of image channels
        """
        import numpy as np
        with tff.TiffFile(filePath) as tf:
            nChannels = np.size(tf.scanimage_metadata['FrameData']['SI.hChannels.channelSave'])
            nImgs = int(np.size(tf.pages)/nChannels)
            nImgs = np.size(tf.pages)
            keys = list(tf.scanimage_metadata['FrameData'].keys())
            idx = util.findStrInList('numFramesPerVolume', keys)
            if len(idx) > 0:
                idx = idx[0]
                numFramesPerVol = tf.scanimage_metadata['FrameData'][keys[idx]]
                if np.size(numFramesPerVol) == 0:
                    numFramesPerVol = 1
            else:
                numFramesPerVol = 1
            idx = util.findStrInList('numVolumes', keys)
            if len(idx) > 0:
                idx = idx[0]
                numVols = tf.scanimage_metadata['FrameData'][keys[idx]]
            else:
                numVols = 1
        return (nChannels, nImgs, numFramesPerVol, numVols)

    if isinstance(tifDirOrPaths, str):
        files_tif = findAndSortFilesInDir(tifDirOrPaths, ext='tif', search_str=searchStr)
        filePaths = np.array([os.path.join(tifDirOrPaths, file_) for file_ in files_tif])
    else:
        filePaths = tifDirOrPaths

    if (len(filePaths) > 5) & (n_jobs > 1):
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs, verbose=0)(delayed(tifInfo)(fp) for fp in filePaths)
        nChannels, nImagesInFile, nFramesPerVol, nVols = zip(*out)
    else:
        nChannels, nImagesInFile, nFramesPerVol, nVols = zip(*[tifInfo(fp) for fp in filePaths])
    tifInfo = dict(filePaths=np.array(filePaths), nChannelsInFile=np.array(nChannels),
                   nImagesInFile=np.array(nImagesInFile), nFramesPerVolume=np.array(nFramesPerVol),
                   nVolumes=np.array(nVols))
    return tifInfo


def splitFolders(srcDir, nParts):
    '''
    Given a source directory, srcDir, splits all the folders in this dir
    into nParts by distributing all the files within each folder into nParts
    sub-folders.
    '''
    import time

    tic = time.time()
    for subDir in subDirsInDir(srcDir):
        subDir_path = os.path.join(srcDir, subDir)
        fileList = findAndSortFilesInDir(subDir_path, ext='')
        fileLists = sublistsFromList(fileList, np.floor(len(fileList)/nParts))
        dstList = [os.path.join(subDir_path, subDir + str(partNum))
                   for partNum in range(1, len(fileLists)+1)]
        print('Moving files to ', dstList)
        for flNum, fl in enumerate(fileLists):
            for file in fl:
                src = os.path.join(srcDir, os.path.join(subDir, file))
                dst = dstList[flNum]
                if not os.path.exists(dst):
                    os.mkdir(dst)
                sh.move(src, dst)
    print(int(time.time()-tic), 'sec')


def split_files_into_subs(fileDir, n_sub: int = 4, div=750, ext='bmp',
                          subPrefix='sub'):
    """ Split files in a directory into 'n_sub' subdirectories within that
    directory such that the # of files in each subdirectory is divisible by
    'div'. If the total # of files does not evenly divide into 'div', the does
    not move the remainder of the files into a subdirectory.
    Parameters
    ----------
    fileDir: str
        Path to the directory of files to be moved
    n_sub: int
        Number of subdirectories into which the files are to be moved
    div: int or None
        The # of files in each subfolder wil be divisible by this number
    ext: str
        File extension filter
    subPrefix: str
        Name prefix of the created subdirectories
    Returns
    -------
    subDirs: List-like
        Subdirectory paths
    """
    import dask
    if div is None:
        div = 1
    filePaths = glob.glob(os.path.join(fileDir, f'*.{ext}'))
    filePaths = np.array(filePaths)
    N = len(filePaths)
    N_div = (N//div)*div
    inds = np.arange(N_div)
    subList = sublistsFromList(inds, div)
    chunkSize = len(subList)//n_sub
    supList = sublistsFromList(np.arange(len(subList)), chunkSize)
    if len(supList)>n_sub:
        supList[-2].extend(supList[-1])
        supList.pop(-1)
    inds_sup = []
    for sl in supList:
        sub_now = np.array(subList)[sl]
        inds_=[]
        for sn in sub_now:
            inds_.extend(sn)
        inds_sup.append(inds_)

    subDirs = []
    for iSub, inds_ in enumerate(inds_sup):
        sn = f'{subPrefix}_{iSub+1}'
        dst = os.path.join(fileDir, sn)
        os.makedirs(dst, exist_ok=True)
        subDirs.append(dst)
        print(f'Moving into {sn}, {iSub+1}/{len(inds_sup)}')
        foo = [dask.delayed(sh.move)(fp, dst) for fp in filePaths[inds_]]
        dask.compute(*foo)
    return subDirs


def strFromHDF(hFile, refList):
    """
    When given an .h5 file (hdf; h5py) and a list of reference objects holding strings,
    returns an array of those strings
    Parameters
    ----------
    hFile: .h5 file object (opened with h5py.File(path_to_file))
    refList: List of reference objects within the input .h5 file

    Returns
    -------
    strList: array of strings
    """
    strList = [u''.join(chr(c) for c in hFile[ref]) for ref in refList]
    return np.array(strList)


def sublistsFromList(inputList, chunkSize):
    '''
    Given a list, chunks it into sizes specified and returns the chunks as items
        in a new list
    '''
    subList, supList = [], []
    for itemNum, item in enumerate(inputList):
        if np.mod(itemNum+1, chunkSize) == 0:
            subList.append(item)
            supList.append(subList)
            subList = []
        else:
            subList.append(item)
    supList.append(subList)
    supList = list(filter(lambda x: len(x) != 0, supList))  # Remove zero-length lists
    return supList
