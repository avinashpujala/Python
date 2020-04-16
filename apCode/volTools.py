
import numpy as np
import psutil
import dask
import os
import sys


def angleBetween2DVectors(v1, v2):
    '''
    Given a list or array of 2D vectors, returns the angle (in radians) between
        each of the vectors such that a sweep from from the 1st vec to the
        2nd vec in the counterclockwise direction returns negative angles
        whereas a sweep in the clockwise direction results in positive angles
    Inputs:
    v1, v2 - The 2 input vectors of size N x 2
    '''
    v1, v2 = np.array(v1), np.array(v2)
    if len(np.shape(v1)) > 1:
        if np.shape(v1)[1] != 2:
            v1 = np.transpose(v1)
        if np.shape(v2)[1] != 2:
            v2 = np.transpose(v2)
        v1 = v1[:, 0] + v1[:, 1]*1j
        v2 = v2[:, 0] + v2[:, 1]*1j
    else:
        v1 = v1[0] + v1[1]*1j
        v2 = v2[0] + v2[1]*1j
    angle = np.angle(v1*np.conj(v2))
    return angle


def animate_images(images, fps=30, display=True, save=False, savePath=None,
                   path_to_ffmpeg = r'V:\Code\FFMPEG\bin\ffmpeg.exe',
                   fig_size=(10, 10), **kwargs):
    """
    Movie from an image stack
    Parameters
    ----------
    images: array, (T,M,N)
        Image stack. Animate along first dimension (typically time)
    fps: scalar
        Frames per second
    display: boolean
        If True, displays movie.
    save: boolean
        IF True, save movie to path specified in savePath.
    savePath: path string
        Full path (includes movie name) where to save movie.
    **kwargs: Key, value pairs for plt.imshow (cmap and interpolation)
        and plt.figure (dpi)
    Returns
    -------
    ani: animation object.
        If this is returned, then video is not displayed until ani is
        typed and run
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    plt.rcParams['animation.ffmpeg_path'] = path_to_ffmpeg
    from IPython.display import HTML
    import time

    N = images.shape[0]
    cmap = kwargs.get('cmap', 'gray')
    interp = kwargs.get('interpolation', 'nearest')
    dpi = kwargs.get('dpi', 30)
    plt.style.use(('seaborn-poster', 'dark_background'))
    fh = plt.figure(dpi=dpi, facecolor='k', figsize=fig_size)
    ax = fh.add_subplot(111, frameon=False)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = ax.imshow(images[0], cmap=cmap, interpolation=interp,
                   vmin=images.min(), vmax=images.max())

    def update_img(n):
        im.set_data(images[n])
        ax.set_title('Frame # {}'.format(n))

    ani = animation.FuncAnimation(fh, update_img, np.arange(N),
                                  interval=1000/fps, repeat=False)
    plt.close(fh)

    if save:
        print('Saving...')
        writer = animation.writers['ffmpeg'](fps=fps)
        if savePath is not None:
            ani.save(savePath, writer=writer, dpi=dpi)
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


def calculateTimePtsFromTPlane(imgDir, fileExt='.stack'):
    import time
    sys.path.insert(0,
                    'C:/Users/pujalaa/Documents/Code/Python/code/codeFromNV')
    sys.path.insert(0, 'C:/Users/pujalaa/Documents/Code/Python/code/util')
    import tifffile as tf
    fileName = 'Plane01' + fileExt
    fp = os.path.join(imgDir, fileName)
    print('Reading plane data...')
    startTime = time.time()
    with open(fp, 'rb') as file:
        A = file.read()
    print(int(time.time()-startTime), 'sec')
    images = tf.TiffFile(os.path.join(imgDir, 'ave.tif')).asarray()
    nPxlsInSlice = np.shape(images)[1]*np.shape(images)[2]
    nTimePts = len(A)/nPxlsInSlice
    nTimePts = int(nTimePts/2)
    return nTimePts


def cart2pol(x, y):
    """
    When give cartesian coordinates, returns polar coordinates
    Parameters
    ----------
    x, y: 1D arrays
        x, and y coordinates respectively
    Returns
    -------
    rho, theta : 1D arrays
        Polar coordinates corresponding vector length and angle respectively
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.array([theta, rho])


def cropND(arr, bounding):
    """
    Crop an N-dimensional volume using the specified bounding
    Parameters
    ----------
    arr: array, (M, N, [,....])
        Array to crop
    bounding: tuple/list/array, (D,), where D is the dimensionaly of the array
        This specifies the shape of the array after cropping.
    Returns
    -------
    arr_crop: array, (bounding.shape)
        Cropped array

    From the good sam "Losses Don" at
    https://stackoverflow.com/users/3931936/losses-don
    """
    from operator import add
    start = tuple(map(lambda a, da: a//2-da//2, arr.shape, bounding))
    end = tuple(map(add, start, bounding))
    slices = tuple(map(slice, start, end))
    return arr[slices]


class cv(object):
    '''
    A set of routines used in computer vision. Most of the code is from the
    book "Programming Computer Vision with Python" by Jan Erik Solem (2012)
    '''
    def pca(X):
        """
        Principal Components Analysis
        Parameters
        X: array-like
            Matrix of size M x N,  M = # of observations, and
            N = dimensionality of the training data.

        Returns
        V : projection matrix with important dimensions first
        S: variance
        mean_X : mean
        """
        import scipy as sp
        # Get dimensions
        nObs, dim = X.shape

        # Center data
        mean_X = X.mean(axis=0)
        X = X - mean_X
        if dim > nObs:
            # PCA - compact trick used
            M = np.dot(X, X.T)  # covariance matrix
            e, EV = np.linalg.eigh(M)  # eigenvalues and eigenvectors
            tmp = np.dot(X.T, EV).T  # this is the compact trick
            V = tmp[::-1]  # reverse, since last eigenvectors are the ones we
            # want
            S = sp.sqrt(e)[::-1]  # reverse, since eigenvalues are in
            # increasing order
            V = V + 0j  # This is a hack that AP implemented to prevent error
            # with complex numbers

            for i in range(V.shape[1]):
                V[:, i] /= S
            V = sp.absolute(V)
            S = sp.absolute(S)
        else:
            # PCA - SVD used
            U, S, V = np.linalg.svd(X)
            V = V[:nObs]  # only makes sense to return the first nObs vectors

        # Return the projection matrix, the variance and the mean
        return V, S, mean_X


def dask_array_from_image_sequence(imgDir, ext: str = 'bmp', verbose=0):
    """
    Returns a lazy dask array of dimensions corresponding to the image
    stack that would have resulted if a sequence of image files within the
    specified directory had been really been loaded. This makes it easy to
    access select images from within a directory.
    Parameters
    ----------
    imgDir: str
        Path to a directory of images
    ext: str
        File extension of the image files to be mapped to the dask array.
    verbose: bool
        If true, prints image dimensions
    Returns
    -------
    imgStack: dask array, ([T,] M, N)
        Dask array corresponding to an image stack, where T is the number of
        images of dimensions (M, N)
    """
    from skimage.io import imread
    import dask.array as da
    from dask import delayed
    from os.path import join
    from apCode.FileTools import findAndSortFilesInDir
    files = findAndSortFilesInDir(imgDir, ext=ext)
    imgPath = join(imgDir, files[0])
    img = imread(imgPath)
    imgDims = img.shape
    if verbose:
        print(f'Image dimensions: {imgDims}')
    images_delayed = [delayed(imread)(join(imgDir, f)) for f in files]
    arr = [da.from_delayed(d, shape=imgDims, dtype=img.dtype)
           for d in images_delayed]
    arr = da.stack(arr, axis=0)
    return arr


def dask_array_from_scanimage_tifs(imgDir):
    """
    Returns a lazy dask array of dimensions corresponding to the image
    stack that would have resulted if a sequence of image files within the
    specified directory had been really been loaded. This makes it easy to
    access select images from within a directory. Assumes that the tif files
    files were written by ScanImage software.
    Parameters
    ----------
    imgDir: str
        Path to a directory of images
    ext: str
        File extension of the image files to be mapped to the dask array.
    Returns
    -------
    imgStack: dask array, ([T,C,Z,] M, N)
        Dask array corresponding to an image stack, where T is the number of
        time points, C = # of channels, Z = # of slices, and (M, N) are the
        image height and width respectively.
    metaData: dict
        Metadata dictionary with the following keys:
        filePaths: list or array
            Paths to each of the .tif file in the path
        nChannelsInFile: list/array
            Number of channels in each of .tif file
        nImagesInFile: list/array
            Number of images in each .tif file
        nFramesPerVolume: list/array
            Number of frames per volume (there can be > 1 vols in each file)
        nVolumes: list/array
            Number of volumes in each .tif file
    """
    from skimage.io import imread
    import dask.array as da
    from dask import delayed
    from os.path import join
    from apCode.FileTools import findAndSortFilesInDir, scanImageTifInfo
    from scipy.stats import mode
    import numpy as np
    ext = 'tif'
    tifInfo = scanImageTifInfo(imgDir)
    nCh = mode(tifInfo['nChannelsInFile'])[0][0]
    nFrames = mode(tifInfo['nFramesPerVolume'])[0][0]

    inds_del1 = np.where(tifInfo['nChannelsInFile'] != nCh)
    inds_del2 = np.where(tifInfo['nFramesPerVolume'] != nFrames)
    inds_del = np.union1d(inds_del1, inds_del2)
    files = findAndSortFilesInDir(imgDir, ext=ext)
    files = np.delete(files, inds_del, axis=0)

    nImgs = np.delete(tifInfo['nImagesInFile'], inds_del, axis=0)
    imgPath = join(imgDir, files[0])
    img = imread(imgPath)
    imgDims = img.shape[-2:]
    print(f'Image dimensions: {(nImgs[0], *imgDims)}')
    images_delayed = [delayed(imread)(join(imgDir, f)) for f in files]
    arr = [da.from_delayed(d, shape=(nImgs, *imgDims),
                           dtype=img.dtype) for nImgs, d in
           zip(tifInfo['nImagesInFile'], images_delayed)]
    block = nCh*nFrames
    nImgs_even = block*(nImgs//block)
    arr = [a[:n] for n, a in zip(nImgs_even, arr)]
    arr = da.concatenate(arr, axis=0)
    arr = arr.reshape(-1, nFrames, nCh, *imgDims)
    arr = np.squeeze(np.swapaxes(arr, 1, 2))
    return arr, tifInfo


def denoise_wavelet(images, scheduler='threads', **kwargs):
    """
    Wrapper for applying denoise_wavelet from skimage.restoration to
    a stack of images using dask backend
    Parameters
    ----------
    images: array, ([T,] M, N)
        Images to denoise.
    scheduler: string; 'threads'(default)|'processes'
        Dask scheduler, see dask
    **kwargs: Keyword arguments to denoise_wavelet from skimage.restoration
        method: str, (default: 'BayesShrink')
        'mode': str, (default: 'soft')
        'wavelet': str, (default: 'db1')
    Returns
    -------
    images_den: array of shape images
        Denoised images
    """
    from skimage.restoration import denoise_wavelet
    if np.ndim(images) == 2:
        images = images[np.newaxis, ...]

    kwargs['method'] = kwargs.get('method', 'BayesShrink')
    kwargs['mode'] = kwargs.get('mode', 'soft')
    images_den = dask.compute(*[dask.delayed(denoise_wavelet)(img, **kwargs)
                                for img in images], scheduler=scheduler)
    images_den = np.array(images_den)
    return np.squeeze(images_den)


def filter_bilateral(images, diam=6, sigma_space=1, sigma_color=5000):
    import cv2
    if images.dtype != np.float32:
        images = images.astype(np.float32)

    if sigma_color is None:
        from apCode.machineLearningnelearning.ml import GMM
        gmm = GMM(n_components=1).fit(images.flatten()[:, None])
        sigma_color = gmm.covariances_[0,0]
    del_func = dask.delayed(cv2.bilateralFilter)
    images_bil = [del_func(img, diam, sigma_color, sigma_space)
                  for img in images]
    images_bil = np.array(dask.compute(*images_bil))
    return np.squeeze(images_bil)

def filter_gaussian(imgs, kSize = 5, sigma=1):
    """
    Fast, opencv GaussianBlur of images
    Parameters
    ----------
    imgs: array, (nImgs, *imgDims)
        Images
    kSize: int or 2-tuple of ints
        Kernel size
    sigma: int
        Filter sigma
    Returns
    --------
    imgs_flt: array, imgs.shape
        Filtered images
    """
    import cv2
    if np.ndim(kSize) < 2:
        kSize = np.ones((2, ), dtype=int)*kSize
    kSize = tuple((kSize//2)*2+1)
    imgs_flt = [dask.delayed(cv2.GaussianBlur)(_, kSize, sigma) for _ in imgs]
    imgs_flt = np.array(dask.compute(*imgs_flt))
    return imgs_flt


def getGlobalThr(img, thr0=[], tol=1/500, nIter_max=100):
    '''
    Gets an image or vector, uses an iterative algorithm to find a
    single threshold, which can be used for image binarization or
    noise discernment in a singal, etc.
    Inputs:
    img  - Can be an image or vector
    tol - Tolerance for iterations. If an iteration yields a value for
        threshold that differs from value from previous iteration by less
        than the tolerance value then iterations stop
    nMaxIter - Max # of iterations to run; can be used to prematurely
        terminate the iterations or if the values do not reach tolerance by
        a certain number of iterations
    thr0 - Initial threshold value from whence to begin looking for
        threshold. If empty, uses otsu's method to find a starting value
    '''
    from skimage.filters import threshold_otsu
    import numpy as np

    def DiffThr(img, thr0):
        sub = img[img < thr0]
        supra = img[img >= thr0]
        thr = 0.5*(np.mean(sub) + np.mean(supra))
        return thr

    if (len(np.shape(thr0)) == 0) or (len(thr0) == 0):
        thr0 = threshold_otsu(img)

    count = 0
    thr1 = DiffThr(img, thr0)
    dThr = np.abs(thr1-thr0)
    while (dThr > tol) & (count < nIter_max):
        thr = thr1
        thr1 = DiffThr(img, thr)
        dThr = np.abs(thr1-thr)
        count = count + 1
    return thr1


def getNumTimePoints(inDir):
    """ nTimePts = getNumTimePoints(inDir)
        Reads Stack_frequency.txt to return the number of temporal stacks
    """
    fp = os.path.join(inDir, 'Stack_frequency.txt')
    with open(fp) as file:
        nTimePts = int(file.readlines()[2])
    return nTimePts


def getStackDims(inDir):
    """ Parse xml file to get dimension information of experiment.
    Returns [x,y,z] dimensions as a list of ints
    """
    import xml.etree.ElementTree as ET
    dims = ET.parse(inDir+'ch0.xml')
    fp = os.path.join(inDir, 'ch0.xml')
    dims = ET.parse(fp)
    root = dims.getroot()
    for info in root.findall('info'):
        if info.get('dimensions'):
            dims = info.get('dimensions')
    dims = dims.split('x')
    dims = [int(float(num)) for num in dims]
    return dims


def getStackFreq(inDir):
    """Get the temporal data from the Stack_frequency.txt file found in
    directory inDir. Return volumetric sampling rate in Hz,
    total recording length in S, and total number
    of planes in a tuple.
    """
    f = open(inDir + 'Stack_frequency.txt')
    times = [float(line) for line in f]

    # third value should be an integer
    times[2] = int(times[2])
    return times


def getStackData(rawPath, frameNo=0):
    """Given rawPath, a path to .stack files, and frameNo, an int, load the
    .stack file for the timepoint given by frameNo from binary and return as a
    numpy array with dimensions=x,y,z"""

    import numpy as np
    from string import Template

    dims = getStackDims(rawPath)
    fName = Template('TM${x}_CM0_CHN00.stack')
    nDigits = 5

    tmpFName = fName.substitute(x=str(frameNo).zfill(nDigits))
    im = np.fromfile(rawPath + tmpFName, dtype='int16')
    im = im.reshape(dims[-1::-1])
    return im


class img(object):
    def convertBmp2Jpg(imgDir):
        '''Given an image dir (path), converts all .bmp images within to
         .jpg images '''
        import time
        from PIL import Image

        def bmp2Jpg(bmpPath):
            targetPath = bmpPath.split('.')[0] + '.jpg'
            im = Image.open(bmpPath)
            im.save(targetPath, format='jpeg')
            im.close()
            try:
                os.remove(bmpPath)
            except RuntimeError:
                print('Unable to delete...', bmpPath)

        tic = time.time()
        print('Getting .bmps in dir...')
        bmpsInDir = img.getImgsInDir(imgDir, imgExts=['.bmp'])
        bmpPaths = [os.path.join(imgDir, bmp) for bmp in bmpsInDir]
        print('Converting .bmps to .jpgs...')
        jpgPaths = [bmp2Jpg(bmpPath) for bmpPath in bmpPaths]
        print(int((time.time()-tic)/60), 'mins')
        return jpgPaths

    def cropImgsAroundPoints(images, pts, cropSize=100, keepSize=True,
                             n_jobs=32, verbose=0, pad_type='zero'):
        '''
        Crops an image stack around fish position coordinates
        Parameters
        ----------
        images: 3D array, shape = (T,M,N)
            Image stack with T images that needs to be cropped
        pts: 2D array, shape = (T,2)
            Coordinates of each point within each image; 1st and 2nd columns are x- and y- coordinates
            respectively
        cropSize: tuple, list, or array, or scalar
            Size (length and breadth) of the cropped array.
            If scalar, then crops out a square patch of the image.
        keepSize: Boolean
            If True, then all cropped images are of same size
        pad_type: string
            Specifies the type of padding to use when keeping image dimensions constant
        n_jobs: scalar
            Number of parallel workers
        vebose: scalar
            Verbosity of progress displayed during parallel processing. 0 = No output.
        Returns
        -------
        I_crop: 3D array, shape = (T,cropSize,cropSize)
            Cropped image stack

        '''
        if np.size(cropSize) == 1:
            cropSize = np.array([cropSize, cropSize])

        def cropImgAroundPt(img, pt, cropSize, keepSize):
            # --- Change from x,y to row,col coordinates
            pt = np.fliplr(np.array(pt).reshape(1, -1)).flatten().astype(int)
            pre = int(cropSize[0]/2)
            post = cropSize[0]-pre
            r = np.arange(np.max([0, pt[0]-pre]),
                          np.min([img.shape[0], pt[0]+post]))
            pre = int(cropSize[1]/2)
            post = cropSize[1]-pre
            c = np.arange(np.max([0, pt[1]-pre]),
                          np.min([img.shape[1], pt[1] + post]))
            img_crop = img[r.reshape(-1, 1), c]
            if keepSize:
                d = np.c_[(cropSize)-np.array(img_crop.shape)]
                d = np.max(np.concatenate((np.zeros(d.shape), d), axis=1),
                           axis=1).astype(int)
                pw = ((int(d[0]/2), d[0]-int(d[0]/2)),
                      (int(d[1]/2), d[1]-int(d[1]/2)))
            if (pad_type == 'nan') | (pad_type == 'zero'):
                mode = 'constant'
                if pad_type == 'nan':
                    const_val = np.nan
                else:
                    const_val = 0
                img_crop = np.pad(img_crop, pad_width=pw, mode=mode, constant_values=const_val)
            elif pad_type == 'edge':
                img_crop = np.pad(img_crop, pad_width=pw, mode=pad_type)

            return img_crop
        if np.ndim(images)==2:
            images = images[np.newaxis,...]

        pts = np.array(pts).reshape(-1,2)
        if len(pts) < len(images):
            pts = np.tile(np.array(pts),(len(images),1))

        if (n_jobs > 1) & (n_jobs > images.shape[0]):
            from joblib import Parallel, delayed
            parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
            I_crop = parallel(delayed(cropImgAroundPt)(img, pt, cropSize,
                                                       keepSize)
                               for img, pt in zip(images, pts))
        else:
            I_crop = [cropImgAroundPt(img, pt, cropSize, keepSize) for
                      img, pt in zip(images, pts)]
        return np.squeeze(np.array(I_crop))

    def findCentroid(img):
        """Returns the centroid of an image in x,y coordinates
        Parameters
        ----------
        img: 2D array, (m, n)
            Input image
        Returns
        -------
        cent: tuple, (2,)
            Centroid
        """
        r, c = np.where(img != None)
        wts = img.flatten()
        wts = wts/wts.sum()
        r_cent, c_cent = np.sum(r*wts), np.sum(c*wts)
        return (c_cent, r_cent)

    def findHighContrastPixels(images, zScoreThr=1, method=1):
        '''
        findHighContrastPixels - Finds high contrast pixels within an image images
            (mean of x and y gradients method)
        pxlInds = findHighContrastPixels(images,zScoreThr = 1)
        Inputs:
        images - Image in which to find pixels
        zScoreThr - Threshold in zscore for a pixel intensity to be considered high contrast
        Outputs:
        edgeInds - Indices of high contrast pixels (i.e., edges)
        I_grad - gradient image which is generated as an intermediate step in detecting edges
            method - Method 1 : I_grad = (np.abs(I_gradX) + np.abs(I_gradY)/2 (default, because faster)
                     Method 2: I_grad = np.sqrt(I_gradX**2 + I_gradY**2)
        '''
        import numpy as np
        if method ==1:
            I_grad = np.abs(np.gradient(images))
            I_grad = (I_grad[0] + I_grad[1])/2
        else:
            I_grad = np.gradient(images)
            I_grad =  (I_grad[0]**2 + I_grad[1]**2)**0.5
        thr  = np.mean(I_grad) + zScoreThr*np.std(I_grad)
        edgeInds = np.where(I_grad > thr)
        return edgeInds, I_grad

    def getImgsInDir(imgDir, imgExts =[], addImgExts = []):
        '''
        Given an imgDir (path), returns a list of the names of all the images in the
            directory, using common image extensions such as .jpg, .jpeg, .tif, .tiff,
            .bmp, .png
        Inputs:
        imgDir - Image directory (must be a path)
        imgExts - Default list of img extensions -
            ['.jpg','.jpeg','.tif','.tiff','.bmp','.png']
        '''
        import numpy as np
        import os

        if len(imgExts) == 0:
            imgExts = ['.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.png']
        if len(addImgExts) != 0:
            imgExts = list(np.union1d(imgExts, addImgExts))
        imgsInDir = []
        thingsInDir= os.listdir(imgDir)
        for ext in imgExts:
            blah = list(filter(lambda x: x.endswith(ext),thingsInDir))
            imgsInDir = list(np.union1d(imgsInDir,blah))
        return imgsInDir

    def gray2rgb(img, cmap='gray'):
        """ Given a grayscale image, returns rgb equivalent
        Inputs:
        cmap - Colormap; can be specified as 'gray', 'jet',
            or as plt.cm.Accent, etc
        """
        import matplotlib.pyplot as plt
        import numpy as np

        def standardize(x): return (x-x.min())/(x.max()-x.min())
        if np.abs(img).max() > 1:
            img = standardize(img)
        cmap = plt.get_cmap(cmap)
        img_rgb = np.delete(cmap(img), 3, 2)
        return img_rgb

    def histeq(img, nBins=256):
        '''
        Equalize the histogram of a grayscale image (similiar to MATLAB
        'histeq')
        Parameters
        img : array-like
            Image array whose histogram is to be equalized
        nBins : scalar
            Number of bins to use for the histogram

        Returns
        img_eq : array-like
            Histogram-equalized image
        '''
        # Get image histogram
        imhist, bins = np.histogram(img.flatten(), nBins, normed=True)
        cdf = imhist.cumsum()  # cumulative distribution function
        cdf = 255*cdf/cdf[-1]   # normalize

        # User linear interpolation of cdf to find new pixel values
        img2 = np.interp(img.flatten(), bins[:-1], cdf)
        return img2.reshape(img.shape), cdf

    def filtImgs(images, filtSize=5, kernel='median', process='parallel',
                 verbose=1):
        '''
        Processes images so as to make moving particle tracking easier
        I_proc = processImagesForTracking(images, filtSize=5)
        Parameters
        ----------
        images: 3D array of shape (nImages, nRows, nCols)
            Image stack to filter
        kernel: String or 2D array
            ['median'] | 'rect' | 'gauss' or array specifying the kernel
        filtSize: Scalar or 2-tuple
            Size of the kernel to generate if kernel is string
        '''
        from scipy import signal
        import apCode.SignalProcessingTools as spt

        if process.lower() == 'parallel':
            from joblib import Parallel, delayed
            import multiprocessing
            parFlag = True
            num_cores = np.min((multiprocessing.cpu_count(), 32))
        else:
            parFlag = False
        if np.ndim(images) < 3:
            images = images[np.newaxis, :, :]
        N = np.shape(images)[0]

        I_flt = np.zeros(np.shape(images))
        if isinstance(kernel, str):
            if kernel.lower() == 'median':
                if np.size(filtSize) > 1:
                    filtSize = filtSize[0]
                if np.mod(filtSize, 2) == 0:
                    #  For median, the filter size should be odd
                    filtSize = filtSize + 1
                    print(f'Median filter size must be odd,' +
                          ' changed to {filtSize}')
                if parFlag:
                    print('# of cores = {}'.format(num_cores))
                    parallel = Parallel(n_jobs=num_cores, verbose=verbose)
                    I_flt = parallel(delayed(signal.medfilt2d)(img, filtSize)
                                     for img in images)
                    I_flt = np.array(I_flt)
                else:
                    for imgNum, img in enumerate(images):
                        if np.mod(imgNum, 300) == 0:
                            print('Img # {0}/{1}'.format(imgNum, N))
                        I_flt[imgNum, :, :] = signal.medfilt2d(img, filtSize)
            elif kernel.lower() == 'rect':
                if np.size(filtSize) == 1:
                    ker = np.ones((filtSize, filtSize))
                else:
                    ker = np.ones(filtSize)
                ker = ker/ker.sum()
                ker = ker[np.newaxis, :, :]
                if parFlag:
                    parallel = Parallel(n_jobs=num_cores, verbose=verbose)
                    del_func = delayed(signal.convolve)
                    I_flt = parallel(del_func(img, ker, mode='same')
                                     for img in images)
                else:
                    I_flt = signal.convolve(images, ker, mode='same')
            elif kernel.lower() == 'gauss':
                if np.size(filtSize) == 1:
                    ker = spt.gausswin(filtSize)
                    ker = ker.reshape((-1, 1))
                    ker = ker*ker.T
                else:
                    ker1 = spt.gausswin(filtSize[0]).reshape((-1, 1))
                    ker2 = spt.gausswin(filtSize[0]).reshape((-1, 1))
                    ker = ker1*ker2.T
                ker = ker/ker.sum()
                if parFlag:
                    parallel = Parallel(n_jobs=num_cores, verbose=verbose)
                    del_func = delayed(signal.convolve)
                    I_flt = parallel(del_func(img, ker, mode='same')
                                     for img in images)
                    I_flt = np.array(I_flt)
                else:
                    ker = ker[np.newaxis, :, :]
                    I_flt = signal.convolve(images, ker, mode='same')
        else:
            ker = ker/ker.sum()
            if parFlag:
                parallel = Parallel(n_jobs=num_cores, verbose=5)
                del_func = delayed(signal.convolve)
                I_flt = parallel(del_func(img, ker) for img in images)
            else:
                ker = ker[np.newaxis, :, :]
                I_flt = signal.convolve(images, ker, mode='same')

        if np.shape(I_flt)[0] == 1:
            I_flt = np.squeeze(I_flt)
        return I_flt

    def gaussFilt(images, sigma=3, mode='nearest', n_jobs=32,
                  verbose=0, preserve_range=True):
        """
        Applies skimage.filters.gaussian to an image stack using parallel loop
        for speed.
        """
        from joblib import Parallel, delayed
        from skimage.filters import gaussian
        if np.ndim(images) == 2:
            images = images[np.newaxis, ...]
        if n_jobs > 1:
            try:
                I_flt = [dask.delayed(gaussian)(img, sigma=sigma, mode=mode,
                         preserve_range=preserve_range) for img in images]
                I_flt = np.array(dask.compute(*I_flt, scheduler='processes'))
            except RuntimeError:
                print('Dask failed, attempting with joblib...')
                parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
                del_func = delayed(gaussian)
                I_flt = parallel(del_func(img, sigma=sigma, mode=mode,
                                          preserve_range=preserve_range)
                                 for img in images)
                I_flt = np.array(I_flt)
        else:
            print('Parallel processing failed, trying serially...')
            I_flt = np.array([gaussian(img, sigma=sigma, mode=mode,
                                       preserve_range=True) for img in images])
        return np.squeeze(I_flt)

    def otsu(img, binary=False, mult=1):
        """
        Returns either a binary or grayscale image after thresholding with
        otsu's threshold
        Parameters
        ----------
        img: array-like
            Image or other array to threshold
        binary: boolean
            If True, then return binary image with values above otsu's
            threshold set to 1, and all other values set to zero
        mult: scalar
            Value by which otsu's threshold is multiplied by before
            thresholding
        Returns
        -------
        img_thr: array-like
            Otsu thresholded image
        """
        from skimage.filters import threshold_otsu
        import numpy as np
        if binary:
            img_bool = img > (threshold_otsu(img)*mult)
            return img_bool
        else:
            img_new = img.copy()
            inds = np.where(img_new < (mult*threshold_otsu(img)))
            img_new[inds] = 0
            return img_new

    def palplot(cMap, nPts=None):
        """
        My version of seaborn style palplot, except takes a single or list of
        colormap objects (e.g., plt.cm.jet) or colormap matrices as input and
        plots specified # of points from them
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if nPts is None:
            nPts = 255
        if (isinstance(cMap, list)) & (len(cMap) > 1):
            pass
        else:
            cMap = [cMap]
        clrsList = []
        for cmNum, cm in enumerate(cMap):
            if isinstance(cm, str):
                cm = plt.cm.get_cmap(cm)
                x = np.linspace(0, 255, nPts).astype(int)
                clrs = [cm(item) for item in x]
            elif isinstance(cm, list):
                x = np.linspace(0, len(cm)-1, nPts).astype(int)
                clrs = [cm[item] for item in x]
            elif isinstance(cm, np.ndarray):
                x = np.linspace(0, np.shape(cm)[0]-1, nPts).astype(int)
                clrs = [cm[item, :] for item in x]
            else:
                x = np.linspace(0, 255, nPts)
                clrs = [cm(int(item)) for item in x]
            clrsList.append(clrs)
            plt.style.use(('seaborn-dark', 'dark_background'))
            if np.shape(clrs)[1] == 4:
                plt.scatter(x, np.ones(np.shape(x))-cmNum, color=clrs, s=1000,
                            marker='s')
            else:
                plt.scatter(x, np.ones(np.shape(x))-cmNum, color=clrs, s=1000,
                            marker='s')
        plt.axis('off')
        return clrsList

    def readImagesInDir(imgDir = None, ext = 'bmp', imgNames = None, imgPaths = None,
                        n_jobs = 32, verbose = 0):
        '''
        Reads 2D images in a dir with specified ext and returns as a numpy array

        Parameters
        imgDir : string or None
            Path to the directory containing the images. If None, uses "imgPaths".
        ext : string
            Extension of the images to read, e.g., 'bmp', 'tiff', etc
        process : string
            'serial' or 'parallel'; determines whether to read the images in serial
            or in parallel
        numCores : scalar
            If process == 'parallel' then this specifies the # of cores to use
        '''
        import os
        from skimage.io import imread
        import numpy as np
        import apCode.FileTools as ft
        from joblib import Parallel, delayed
        import multiprocessing as mp

        if np.any(imgDir == None) & np.any(imgPaths == None):
            print('Either image directory or image paths must be provided')

        if np.any(imgPaths == None):
            if np.any(imgNames == None):
                imgPaths = [os.path.join(imgDir, img) for img in ft.findAndSortFilesInDir(imgDir,ext = ext)]
            else:
                imgPaths = [os.path.join(imgDir, img) for img in imgNames]

        dispChunk = int(len(imgPaths)/5)
        images=[]
        count = 0
        if n_jobs < 2:
            for img in imgPaths:
                images.append(imread(img))
                count = count + 1
                if np.mod(count,dispChunk)==0:
                    print(str(count) + '/' + str(len(imgPaths)), '\t')
        else:
            n_jobs = np.min((mp.cpu_count(),n_jobs))
            try:
                images = Parallel(n_jobs = n_jobs, verbose=verbose)(delayed(imread)(img) for img in imgPaths)
            except:
                import dask
                images = dask.compute(*[dask.delayed(imread)(ip) for ip in imgPaths])
        images = np.array(images)
        return images

    def resize(images, sz, n_jobs = 32, verbose = 0,preserve_dtype = True,**kwargs):
        """
        Resize an image array using PIL. For detailed help see 'resize' in PIL
        Parameters
        ----------
        img : array-like, 2D
            This is the array to be resized.
        sz : 2-tuple
            (width, height) of resized array
        n_jobs, verbose: see Parallel, delayed

        preserve_dtype: bool
            If True, then converts the data type of the rescaled images into the
            same as the original images
        **kwargs: see sklearn.transform.resize
        Returns
        -------
        img_resized : array-like
            Resized array
        """
#        import PIL as pil
        from skimage.transform import resize
        import numpy as np

#        def rs(im, sz):
#            pil_im = pil.Image.fromarray(np.uint8(im))
#            return np.array(pil_im.resize(sz))

        preserve_range = kwargs.get('preserve_range',True)
        kwargs['preserve_range'] = preserve_range
        if np.ndim(images)==2:
            I_rs = resize(images,sz,**kwargs)
        else:
            if n_jobs >1:
                from joblib import Parallel, delayed
                I_rs = Parallel(n_jobs= n_jobs, verbose = verbose)(delayed(resize)(im,sz,**kwargs) for im in images)
            else:
                I_rs =[resize(im,sz, **kwargs) for im in images]
        I_rs = np.array(I_rs)
        if preserve_dtype:
            I_rs = I_rs.astype(images.dtype)
        return I_rs

    def resize_discrete(images, sz):
        """
        Resize discrete images using PIL. Rescales images to 0-255 and
        converts to integer type before resizing. So, values are not
        preserved, but quantization is.
        Parameters
        ----------
        images: array, ([nImages,], nRows, nCols)
            Images to resize
        sz: tuple, (2,)
            Size of images after resizing.
        Returns
        -------
        images_rs: array, ([nImages,], *sz)
            Resized images
        """
        from PIL import Image
        from apCode.SignalProcessingTools import standardize
        from dask import compute, delayed
        import numpy as np
        def rs_int(img, sz):
            img = np.uint8(standardize(img)*255)
            return np.array(Image.fromarray(img).resize(sz))
        if np.ndim(images) ==2:
            images = images[np.newaxis,...]
        images_rs = compute(*[delayed(rs_int)(img,sz) for img in images])
        return np.squeeze(np.array(images_rs))

    def rgb2Gray(I_stack):
        '''
        Given an image stack of RGB images, returns its grayscale version
        Inputs:
        I_stack - Input image stack of RGB images. Must have shape T X M X N X 3,
            where T is usually time, M,N are image dimensions, 3 indicates the color
            channels
            '''
        import numpy as np
        import sys
        if len(np.shape(I_stack)) !=4:
            sys.exit('Input stack dimensions must be T x M x N x 3')

        def rgb2g(img):
            return np.dot(img[...,:3],[0.299, 0.587, 0.114])
        return [rgb2g(img) for img in I_stack]

    def rgba2rgb(images):
        """"
        Given an rgba image stack, where a is the alpha channel, returns an rgb
            image scaled by the alphas in the alpha channel. This can be useful
            for opening using ImageJ
        Parameters:
        images - array-like of shape(z,m,n,4), where z is the number of image slices,
            m, n are the rows and columns of each image, and 4 corresponds to the
            r,g,b, and a (alpha) channels respectively.
        Returns:
        images - array-like, shape = (z, m, n, 3)
        """
        import numpy as np
        import sys
        if isinstance(images, list):
            images = np.array(images)
        if np.shape(images)[-1] <4:
            print('Image must have 3 color and 1 alpha channel')
            sys.exit()
        elif np.ndim(images) ==3:
            images = images[np.newaxis,:,:,:]
        elif np.ndim(images)<3:
            print('RGB image must have at least 3 dimensions')
            sys.exit()
        alphaMask = images[:,:,:,-1]
        alphaMask = np.tile(alphaMask[:,:,:,np.newaxis],[1,1,1,3])
        I_rgb = images[:,:,:,:3]*alphaMask
        return I_rgb

    def rotate(images, angle, n_jobs:int = 32, verbose:int =0, preserve_dtype:bool = True,**kwargs):
        """
        Wrapper for applying skimage.transform.rotate to an image stack
        """
        import numpy as np
        from skimage.transform import rotate
        from joblib import Parallel, delayed
        def wrap_rot(func):
            def wrapper_func(*args,**kwargs):
                dtype = args[0].dtype
                kwargs['preserve_range'] = True
                img = func(*args,**kwargs)
                return img.astype(dtype)
            return wrapper_func

        if np.ndim(images) ==2:
            images = images[np.newaxis,...]
        kwargs['preserve_range'] = kwargs.get('preserve_range', True)
        kwargs['mode'] = kwargs.get('mode','wrap')

        if preserve_dtype:
            rotate = wrap_rot(rotate)

        if n_jobs <2:
            I_rot = np.array([rotate(img,angle,**kwargs) for img in images])
        else:
            try:
                I_rot = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(rotate)
                (img,angle,**kwargs) for img in images))
            except:
                import dask
                I_rot = np.array(dask.compute(*[dask.delayed(rotate)(img,angle,**kwargs) for img in images]))
        return np.squeeze(I_rot)

    def saveImages(images,imgDir = [], imgNames =[], fmt = 'bmp', cmap = 'gray',
                   dispChunk = None, n_jobs = 20, verbose = 0):
        """
        Saves an image stack

        Parameters
        ----------
        images: (m,n,k), array-like
            Image stack to save
        imgDir: string
            Path to directory where images are to be saved
        imgNames: list
            List of strings to give images
        dispChunk: scalar
            If np.inf, then does not print output of image numbers when saving
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        if isinstance(cmap,str):
            if cmap.lower()=='gray':
                from skimage.io import imsave
            else:
                imsave = plt.imsave
        elif cmap.name == 'gray':
            from skimage.io import imsave
        else:
            imsave = plt.imsave

        if np.ndim(images)==2:
            images = images[np.newaxis,...]

        if len(imgDir)==0:
            imgDir = os.getcwd()
        elif os.path.exists(imgDir) == False:
            os.mkdir(imgDir)
        if len(imgNames)==0:
            imgNames = ('Img' + '%.6d' % num + '.' + fmt for num in range(np.shape(images)[0]))
        if dispChunk == None:
            dispChunk = np.inf
        else:
            dispChunk = int((np.shape(images)[0])/30)

        if (images.shape[0]<n_jobs):
            for imgNum,imgName in enumerate(imgNames):
                filePath = os.path.join(imgDir,imgName)
                imsave(filePath,images[imgNum])
#                if np.mod(imgNum,dispChunk)==0:
#                    print(imgNum)
        else:
            from joblib import Parallel, delayed
            Parallel(n_jobs = n_jobs, verbose = verbose)(delayed(imsave)(os.path.join(imgDir,imgName),img) for imgName,img in zip(imgNames,images))

    def make_video(images, outvid=None, fps=5, size=None,
                   is_color=True, format="XVID"):
        """
        Create a video from a list of images.

        @param      outvid      output video
        @param      images      list of images to use in the video
        @param      fps         frame per second
        @param      size        size of each frame
        @param      is_color    color
        @param      format      see http://www.fourcc.org/codecs.php
        @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

        The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
        By default, the video will have the size of the first image.
        It will resize every image to this size before adding them to the video.
        """
        from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
        import os
        fourcc = VideoWriter_fourcc(*format)
        vid = None
        for image in images:
            if not os.path.exists(image):
                raise FileNotFoundError(image)
            img = imread(image)
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()
        return vid

def interp3d(vol, dx = 1, dy = 1, dz = 0.25, method = 'linear'):
    """
    Interpolate a 3D volume. Presently, only supports "linear" and "nearest neighbor"
    interpolation.
    Parameters
    ----------
    vol: array, (nSlices, nRows, nCols) or (Z, Y, X)
        Volume to interpolate.
    dx, dy, dz: scalars
        Resolution along the x, y, and z dimensions respectively. Here, z = nSlices.
    method: string
        'linear' or 'nearest'. Presently, the script uses
        scipy.interpolate.RegularGridInterpolator, which does not support other types
        of interpolation. scipy.interpolate.griddata supports 'cubic', but did not
        implement it here.
    Returns
    -------
    vol_interp: array, (nSlices_new, nRows_new, nCols_new) or (Z_new, Y_new, X_new)
    """
    from scipy.interpolate import RegularGridInterpolator
    z = np.arange(vol.shape[0])
    x = np.arange(vol.shape[2])
    y = np.arange(vol.shape[1])
    vol_swap = np.transpose(vol,(1,2,0))
    interp = RegularGridInterpolator((y,x,z),vol_swap,method = method)
    dx, dy, dz = int(np.round(len(x)/dx)), int(np.round(len(y)/dy)), int(np.round(len(z)/dz))
    xx = np.linspace(x[0], x[-1], dx)
    yy = np.linspace(y[0],  y[-1], dy)
    zz = np.linspace(z[0], z[-1],dz)
    X,Y,Z = np.meshgrid(xx,yy,zz)
    xx,yy,zz= X.ravel(), Y.ravel(), Z.ravel()
    points = [*zip(yy,xx,zz)]
    vol_interp = interp(points).reshape(vol_swap.shape[0], vol_swap.shape[1],-1)
    return np.transpose(vol_interp, (2,0,1))

def ipca(images, components:int=50, batch:int=1000):
    """
    Iterative Principal Component analysis, see sklearn.decomposition.incremental_pca.IncrementalPCA
    Parameters
    ----------
    components (default 50) = number of independent components to return
    batch (default 1000)  = number of pixels to load into memory simultaneously in IPCA.
        More requires more memory but leads to better fit.
    Returns
    -------
    eigenseries: principal components (pixel time series) and associated singular values
    eigenframes: eigenframes are obtained by multiplying the projected frame matrix by the
    projected movie (whitened frames?)
    proj_frame_vectors: the reduced version of the movie vectors using only the principal
    component projection.
    This code was copied from CaImAn
    https://caiman.readthedocs.io/en/master/index.html#
    """
    import numpy as np
    from sklearn.decomposition.incremental_pca import IncrementalPCA
    # vectorize the images
    num_frames, h, w = np.shape(images)
    frame_size = h * w
    frame_samples = np.reshape(images, (num_frames, frame_size)).T

    # run IPCA to approxiate the SVD
    ipca_f = IncrementalPCA(n_components=components, batch_size=batch)
    ipca_f.fit(frame_samples)

    # construct the reduced version of the movie vectors using only the
    # principal component projection

    proj_frame_vectors = ipca_f.inverse_transform(ipca_f.transform(frame_samples))

    # get the temporal principal components (pixel time series) and
    # associated singular values

    eigenseries = ipca_f.components_.T

    # the rows of eigenseries are approximately orthogonal
    # so we can approximately obtain eigenframes by multiplying the
    # projected frame matrix by this transpose on the right

    eigenframes = np.dot(proj_frame_vectors, eigenseries)
    return eigenseries, eigenframes, proj_frame_vectors

def denoise_ipca(images, components:int=50, batch:int=1000):
    """
    Create a denoised version of the movie using only the first 'components' components
    """
    _, _, clean_vectors = ipca(images, components = components, batch = batch)
    images_clean = np.reshape(np.float32(clean_vectors.T), np.shape(images))
    return images_clean

class morphology():
    __codeDir = r'V:\Code\Python\code'
    import sys as __sys
    __sys.path.append(__codeDir)
    import apCode.geom as __geom
    endpoints_curve_2d = __geom.endpoints_curve_2d

    def neighborhood(img,coords, n=1):
        """
        Returns the neighborhood in an image
        Parameters
        ----------
        img: 2D array (M,N)
            Image in which to detect neighborhood
        coords: array, (K,2)
            Coordinates of pixels to get neighborhoods for.
            coords[0,:] is the 1st pixel, and  coords[0,0] is the
            row index of this pixel
        n: scalar, integer
            Neighorhood radius. For e.g., n = 1, returns 9 pixels,
            including 8 neighbors and the pixel value at the coordinate.
            Likewise, n = 2, returns 25 neighbor pixel values
        Returns
        -------
        P: array, (K,n+2,n+2)
            Pixel values neighboring the input coordinates
        N: array, (K,2,n+1)
            Sparse neighborhood indices.
            For the k^{th} coordinate, where k = {1, 2,..., K}
            N[k,0,3] gives the row indices of the neighborhood and
            N[k,1,3] gives the column indices.
        """
        import numpy as np
        if np.size(coords) == 2:
            coords = [coords]
        P, N = [], []
        for r, c in coords:
            inds = np.array([np.arange(r-n, r+n+1), np.arange(c-n, c+n+1)])
            rowInds = np.where((inds[0] >= 0) & (inds[0] < img.shape[0]))[0]
            colInds = np.where((inds[1] >= 0) & (inds[1] < img.shape[1]))[0]
            keepInds = np.intersect1d(rowInds, colInds)
            inds = inds[:, keepInds]
            rr, cc = np.meshgrid(inds[0], inds[1])
            p = img[rr.astype(int), cc.astype(int)]
            N.append(inds)
            P.append(p)
        return P, N

    def thin_weighted(img, nhood=4, points=None):
        """
        Returns the x,y coordinates of the thinned line resulting from
        skimage.morphology.thin, but after weighting with pixel intensity
        Parameters
        ---------
        img: 2D array (M,N)
            Image to thin
        nhood: scalar
            Neighborhood of pixel weighting
        points: array, (K,2)
            Points to be weighted. If None or [], them uses thinning obtain
            this information. However, it weighting needs to be done
            iteratively, then it helps to have these points
        Returns
        -------
        xy: array, (2,N)
            xy[0], and xy[1] are the x and y coordinates respectively of the
            thinned lines in image
            """
        from skimage.morphology import thin

        def row_wts(img, r, c, nhood):
            nRows = np.shape(img)[0]
            x = np.arange(r-nhood, r+nhood+1)
            y = np.ones((len(x),))*c
            inds = np.array((x, y)).astype(int)
            keepInds = np.where((inds[0] >= 0) & (inds[0] < nRows))[0]
            inds = inds[:, keepInds]
            wts = img[inds[0], inds[1]]
            wts = wts/np.sum(wts)
            r_wt = np.dot(x[keepInds], wts)
            return r_wt

        def col_wts(img, r, c, nhood):
            nCols = np.shape(img)[1]
            y = np.arange(c-nhood, c+nhood+1)
            x = np.ones((len(y),))*r
            inds = np.array((x, y)).astype(int)
            keepInds = np.where((inds[1] >= 0) & (inds[1] < nCols))[0]
            inds = inds[:, keepInds]
            wts = img[inds[0], inds[1]]
            wts = wts/np.sum(wts)
            c_wt = np.dot(y[keepInds], wts)
            return c_wt
        if np.size(points) < 2:
            R, C = np.where(thin(img))
        else:
            R, C = points[:, 1], points[:, 0]
        xy = []
        for r, c in zip(R, C):
            r_wt = row_wts(img, r, c, nhood)
            c_wt = col_wts(img, r, c, nhood)
            xy.append([c_wt, r_wt])
        return np.array(xy)


class plot(object):
    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """
        Draw an ellipse with a given position and covariance
        **kwargs are those used by ax.add_patch

        From:
        VanderPlas, J. Python Data Science Handbook (2016).

        """
        from matplotlib.patches import Ellipse
        import matplotlib.pyplot as plt
        import numpy as np
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))


def projFigure(vol, limits, plDims=[16, 10, 5], zscale=5, colors='gray',
               title=None):

    """Display vol.max(dim) - vol.min(dim) for dims in [0,1,2]
    Heavily adapted from Jason Wittenbach's crossSectionPlot.
    """
    import matplotlib.pyplot as plt

    x, y, z = plDims
    grid = (y+z, x+z)
    zRat = zscale*(float(y)/z)
    plt.figure(figsize=grid[-1::-1])
    ax1 = plt.subplot2grid(grid, (0, 0), rowspan=y, colspan=x)
    plt.imshow(vol.max(0) + vol.min(0), clim=limits, cmap=colors,
               origin='leftcorner', interpolation='Nearest')
    ax1.axes.xaxis.set_ticklabels([])

    if title:
        plt.title(title)

    # plot the x-z view (side-on)
    plt.subplot2grid(grid, (y, 0), rowspan=z, colspan=x)
    plt.imshow(vol.max(1)+vol.min(1), aspect=zRat, clim=limits, cmap=colors,
               origin='leftcorner', interpolation='Nearest')

    # plot the y-z view (head-on)
    ax3 = plt.subplot2grid(grid, (0, x), rowspan=y, colspan=z)
    plt.imshow((vol.max(2)+vol.min(2)).T, aspect=1/zRat, clim=limits,
               cmap=colors, origin='leftcorner', interpolation='Nearest')

    ax3.axes.yaxis.set_ticklabels([])


def tToZStacks(inDir):
    '''
    tToZStacks(inDir):
    Writes the tStacks in 'inDir' to zStacks in 'inDir\\zStacks'
    '''
    import sys
    sys.path.insert(0,
                    'C:/Users/pujalaa/Documents/Code/Python/code/codeFromNV')
    sys.path.insert(0, 'C:/Users/pujalaa/Documents/Code/Python/code/util')
    import time
    import numpy as np
    import volTools

    imgDir = inDir
    fileExt = '.stack'
    fileStem = 'Plane'
    stackDims = volTools.getStackDims(imgDir)
    nTimePts = volTools.getNumTimePoints(imgDir)

    outDir = os.path.join(imgDir, 'zStacks')
    if os.path.isdir(outDir):
        print(outDir, '\n ...already exists!')
    else:
        os.mkdir(outDir)

    print('Writing files...')
    startTime = time.time()
    for stackNum in range(stackDims[2]):
        print('Reading plane ' + str(stackNum+1) + '...')
        fn = fileStem + '%0.2d' % (stackNum+1) + fileExt
        fp = os.path.join(imgDir, fn)
        blah = np.fromfile(fp, dtype='uint16')
        blah = np.reshape(blah, (nTimePts, stackDims[1], stackDims[0]))
        print('Writing plane ' + str(stackNum+1) + '...')
        for timePt in range(nTimePts):
            fn = 'TM' + '%0.05d' % timePt + '.bin'
            fp = os.path.join(outDir, fn)
            file = open(fp, 'ab')
            file.write(blah[timePt])
            file.close()
    print(int(time.time()-startTime)/60, 'min')


def pol2cart(th, rho):
    '''
    Transforms from polar to cartesian coordinates
    Parameters
    ----------
    th: array-like, (N,)
        Angles in radians
    rho: array-like, (N,)
        Radius
    Returns
    -------
    x, y: array-like, (2,N)
        x, y coordinates in Cartesian space
    NB: Rounds x,y values to 3 decimal places
    '''
    x = rho * np.cos(th)
    y = rho * np.sin(th)
    return np.array([x, y])


def radiatingLinesAroundAPoint(pt, lineLength, dTheta=15, dLine=1):
    '''
    Given the coordinates of a point, returns the list of coordinates of a
     series of lines radiating from that point
    lines = radiatingLinesAroundAPoint(pt, lineLength, dTheta = 15, dLine=1)
    Parameters
    ----------
    pt: tupe, (2,)
        x,y coordinates of a point from which the lines should radiate
    lineLength: int
        Length in pixels of the the line segments
    dTheta: scaler
        Angular spacing of the lines around the point. For instance setting
        if dTheta = 90, returns 4 lines at right angles to each other.
    dLine: scalar
        Radial distance between points in the line
    '''

    import numpy as np
    lines = []
    xInds = []
    yInds = []
    thetas = np.arange(0, 360, dTheta)
    lineLengths = np.arange(1, lineLength+1, dLine)
    for theta in thetas:
        inds = list(map(lambda x: pol2cart(x, theta), lineLengths))
        xInds = np.array(list(ind[0] for ind in inds)) + pt[0]
        yInds = np.array(list(ind[1] for ind in inds)) + pt[1]
        line = np.array([xInds, yInds])
        lines.append(line)
    return np.array(lines)


class Register():
    """
    Registers an image or image stack to a reference object
    Parameters
    ----------
    n_jobs: integer
        Number of parallel jobs (if backend == 'joblib')
    verbose: integer
        See Parallel, delayed from joblib
    upsample_factor: scalar
        Factor by which to upsample images before registration. This is useful
        either for subpixel registration (upsample_factor > 1) or for speeding
        up registration (upsample_factor <1).
    backend: string, 'joblib' (default) or 'dask'
        Backend to use in running registration. If 'joblib' uses Parallel,
        delayed with 'loky' backend. If 'dask' then uses multithreading and
        dask
    scheduler: string, 'threads' or 'processes' (default)
        IFF backend == 'dask', this specifies the type of scheduler to use.
        See dask.compute
    regMethod: string, 'st' (standard translation), 'cr' (caiman rigid), or
        'cpwr' (caiman piecewise rigid)
        Specifies the registration method. For more info on 'cr' or 'cpwr' see
        caiman.motion_correction.MotionCorrect(). 'st' is implemented using
        skimage.feature.register_translation.
    """
    def __init__(self, n_jobs=32, verbose=0, upsample_factor=1, backend='dask',
                 scheduler='processes', regMethod='st', filtSize=None,
                 patchPerc=(20, ), patchOverlapPerc=(60, ),
                 maxShiftPerc=(20, )):
        """
        Parameters
        ----------
        n_jobs: int
            Number of parallel workers
        verbose: int
            Verbosity of Parallel workers.
        backend: str, 'dask' or 'joblib'
            Which parallel processing framework to use. If 'dask' then uses
            dask intead of joblib's Parallel
        scheduler: str, 'processes' or 'threads'
            If backend = 'dask', then this determines whether to use parallel
            processes or threads
        regMethod: str, 'st', 'cr', 'cpwr'
            Registration algorithm to use.
            'st' - skimage translation
            'cr' - caiman rigid (backend is still 'st', but there may be other
                differences)
            'cpwr' - caiman piecewise rigid
        upsample_factor: scalar
            If regMethod = 'st', them upsampling factor for images.
        filtSize: None or scalar
            Gaussian filter sigma for filtering before registering. The
            filtering is only done to compute registration shifts. The
            registered images wil not be filtered
        patchPerc: tuple-like, (n,) where n = 1 or 2 or scalar
            If regMethod = 'cpwr', then this determines the patch size in units
            of percentage of the 1st and 2nd image dimensions. For example,
            if image is M by N, then patchPerc = (10, 20) results in patchSize
            = (M*10/100, N*20/100)
            If patchPerc has only one element then uses symmetric patches
            with size determined by smaller image dimension.
        patchOverlapPerc: tuple-like or scalar (see patchPerc)
            Percentage by which patch sizes overlap in each dimension.
            For example, (60, 60) results in 60% (relative to patch dimensions)
            overlap in patches. The stride then = patchSize - overlap
        maxShiftPerc: tuple-like or scalar (see patchPerc)
            Maximum shift allowed during registration
        """
        self.n_jobs_ = n_jobs
        self._verbose = verbose
        self.upsample_factor_ = upsample_factor
        self.backend_ = backend
        self.scheduler_ = scheduler
        self.regMethod_ = regMethod
        self.filtSize_ = filtSize
        self._patchPerc = patchPerc
        self._patchOverlapPerc = patchOverlapPerc
        self._maxShiftPerc = maxShiftPerc

    @staticmethod
    def correlate_to_ref(images, ref=None):
        """Correlate images to a reference or images.mean(axis=0)
        Parameters
        ----------
        images: array, (nImgs, *imgDims)
            Image stack
        ref: array, imgDims
            Reference image to correlate each image with
        Returns
        -------
        corrs: array, (nImgs,)
            Correlations, such that corrs[i] = correlation(images[i], ref)
        """

        def corrImgs(img1, img2):
            return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

        if ref is None:
            ref = images.mean(axis=0)
        corrs = [dask.delayed(corrImgs)(img, ref) for img in images]
        corrs = dask.compute(*corrs, scheduler='processes')
        return corrs

    def fit(self, images, ref=None):
        """
        Registers images against reference and returns translation coordinates
        from registration
        Parameters
        ----------
        images: array, ([nTimePts,], nRows, nCols)
            Image stack to register
        ref: array of shape (nRwos, nCols) or None
            Reference image to register against. If None, then uses mean of
            image stack along first dimension (i.e. time).
        Returns
        -------
        regObj: object
            Registration object with registration shifts stored in
            regObj.translation_coords_. The method regObj.transform() will
            apply registration parameters computed from images to other image
            stacks
        """
        import numpy as np
        from skimage.feature import register_translation as rt
        if np.ndim(images) < 3:
            images = images[np.newaxis, :, :]

        if self.filtSize_ is not None:
            images = img.gaussFilt(images, sigma=1)
        if ref is None:
            ref = images.mean(axis=0)
        patchSize, overlaps, max_shifts = None, None, None
        max_deviation_rigid = None
        if self.regMethod_ == 'st':
            if self.n_jobs_ <= 1:
                shifts =\
                    [rt(ref, img, upsample_factor=self.upsample_factor_)[0]
                     for img in images]
                shifts = np.array(shifts)
            else:
                if self.backend_ == 'joblib':
                    from joblib import Parallel, delayed
                    from multiprocessing import cpu_count
                    self.n_jobs_ = np.min((self.n_jobs_, cpu_count()))
                    shifts = Parallel(n_jobs=self.n_jobs_,
                                      verbose=self._verbose)
                    (delayed(rt)(ref, img) for img in images)
                    shifts = np.array([shift[0] for shift in shifts])
                elif self.backend_ == 'dask':
                    import dask
                    shifts_lazy = [dask.delayed(rt)
                                   (ref, img,
                                    upsample_factor=self.upsample_factor_)
                                   for img in images]
                    foo = dask.compute(shifts_lazy,
                                       scheduler=self.scheduler_)[0]
                    shifts = np.array([_[0] for _ in foo])
                else:
                    print('Please specify valid backend ("joblib" or "dask")')
                    shifts = None
        elif (self.regMethod_ == 'cr') | (self.regMethod_ == 'cpwr'):
            print('Registering with caiman...')
            import caiman as cm
            from caiman.motion_correction import MotionCorrect
            cm.stop_server()
            n_processes = np.maximum(np.minimum(int(psutil.cpu_count()),
                                                images.shape[0]-2), 1)
            if 'dview' in locals():
                cm.stop_server(dview=dview)
            c, dview, n_processes =\
                cm.cluster.setup_cluster(backend='local',
                                         n_processes=n_processes,
                                         single_thread=False,
                                         ignore_preexisting=True)
            imgDims = np.array(images.shape[-2:])

            def castToArr(x):
                return np.array(x) * np.ones((2,))

            if np.ndim(self._patchPerc) < 2:
                patchSize = np.min(imgDims)*castToArr(self._patchPerc)/100
            else:
                patchSize = np.array(imgDims)*np.array(self._patchPerc)/100
            if np.ndim(self._patchOverlapPerc) < 2:
                overlaps = np.min(patchSize)*castToArr(self._patchOverlapPerc)
                overlaps = overlaps/100
            else:
                overlaps = patchSize*np.array(self._patchOverlapPerc)/100
            if np.ndim(self._maxShiftPerc) < 2:
                max_shifts = np.min(imgDims)*castToArr(self._maxShiftPerc)
                max_shifts = max_shifts/100
            else:
                max_shifts = (imgDims*np.array(self._maxShiftPerc)/100)
            patchSize = patchSize.astype(int)
            overlaps = overlaps.astype(int)
            strides = patchSize - overlaps
            max_shifts = max_shifts.astype(int)
            max_deviation_rigid = max_shifts[0]
            shifts_opencv = True
            border_nan = 'copy'
            if self.regMethod_ == 'cpwr':
                pw_rigid = True
            else:
                pw_rigid = False
            fname_now = cm.movie(images).save('temp.mmap', order='C')
            mc = MotionCorrect([fname_now], dview=dview, max_shifts=max_shifts,
                               strides=strides, overlaps=overlaps,
                               max_deviation_rigid=max_deviation_rigid,
                               nonneg_movie=True, border_nan=border_nan,
                               shifts_opencv=shifts_opencv, pw_rigid=pw_rigid)
            mc.motion_correct(save_movie=False, template=ref)
            attrs = mc.__dict__.keys()
            if 'x_shifts_els' in attrs:
                shifts = np.array([mc.x_shifts_els, mc.y_shifts_els])
                shifts = shifts.transpose(1, 0, 2)
            else:
                shifts = np.array(mc.shifts_rig)
            self.apply_shifts_movie = mc.apply_shifts_movie
            registration_constraints = dict(strides=strides, overlaps=overlaps,
                                            max_shifts=max_shifts)
            registration_constraints['max_deviation_rigid'] =\
                max_deviation_rigid
            self.registration_constraints_ = registration_constraints
            cm.stop_server(dview=dview)
        self.translation_coords_ = shifts
        self.patchSize_ = patchSize
        self.overlaps_ = overlaps
        self.max_shifts_ = max_shifts
        self.max_deviation_rigid_ = max_deviation_rigid
        return self

    def transform(self, images):
        """
        Register a stack of images using translation coordinates
        computed using the fit method
        Parameters
        ----------
        images: array, ([nTimePts,], nRows, nCols)
            Image to stack to which to apply registration parameters computed
            from the same or a different image stack of similar dimensions
            (such as a different channel, for instance)
        Returns
        -------
        I_shifted: array of shape(images)
            Registered image stack.
        """
        from scipy.ndimage import fourier_shift
        from numpy.fft import fftn, ifftn
        if self.regMethod_ in ['cr', 'cpwr']:
            import caiman as cm
        shifts = self.translation_coords_
        def f(img, s): return np.real(ifftn(fourier_shift(fftn(img), s)))
        if np.ndim(images) < 3:
            images = images[np.newaxis, :, :]
            shifts = np.array(shifts).reshape((1, -1))
        if (self.regMethod_ == 'cr') | (self.regMethod_ == 'cpwr'):
            if 'dview' in locals():
                cm.stop_server(dview=dview)
            c, dview, n_processes =\
                cm.cluster.setup_cluster(backend='local', n_processes=None,
                                         single_thread=False,
                                         ignore_preexisting=True)
            I_shifted = self.apply_shifts_movie(images)
            cm.stop_server(dview=dview)
        else:
            if self.n_jobs_ <= 1:
                I_shifted = [f(img, s) for img, s in zip(images, shifts)]
            else:
                if self.backend_ == 'joblib':
                    from joblib import Parallel, delayed
                    from multiprocessing import cpu_count
                    self.n_jobs_ = np.min((self.n_jobs_, cpu_count()))
                    parallel = Parallel(n_jobs=self.n_jobs_,
                                        verbose=self._verbose)
                    I_shifted =\
                        parallel(delayed(f)(img, s) for img, s in
                                 zip(images, shifts))
                elif self.backend_ == 'dask':
                    import dask
                    I_lazy = [dask.delayed(f)(img, s) for img, s in
                              zip(images, shifts)]
                    I_shifted = dask.compute(I_lazy, scheduler=self.scheduler_,
                                             n_workers=self.n_jobs_)[0]
                else:
                    print('Please specify valid backend ("joblib" or "dask")')
                    I_shifted = None
        return np.squeeze(np.array(I_shifted))

def register_volumes(static, moving, nBins=32, sampling_prop=None,
                     level_iters=[100, 50, 2], sigmas=[5.0, 2.0, 0],
                     factors=[8, 3, 1], params0=None,
                     static_grid2world=None, moving_grid2world=None,
                     verbosity=2):
    """Register two volumes using AffineTransform3D in the dipy package
    Paramters
    ---------
    static: array, (nSlices, nRows, nCols)
        Reference volume
    moving: array, (nSlices, nRows, nCols)
        Volume to register. Need not have same dimensions as static
    nBins: int
        Number of histogram bins
    sampling_prop, level_iters, sigmas, factors, params0: see dipy
    static_grid2world: None or array, (4, 4)
        Static to physical space mapping affine transformation matrix. If None,
        then identity matrix
    moving_grid2world: None or array, (4, 4)
        Moving to physical space mapping affine transformation matrix. If None,
        then identity matrix
    Returns
    -------
    moving_transformed: array, static.shape
        Transformed volume
    affine: object
        Object with stored transformation parameters and methods for
        transforming other volumes

    """
    from dipy.align.imaffine import (transform_centers_of_mass,
                                     MutualInformationMetric,
                                     AffineRegistration)
    from dipy.align.transforms import (AffineTransform3D,
                                       TranslationTransform3D)
    metric = MutualInformationMetric(nBins, sampling_prop)
    affreg = AffineRegistration(metric=metric, level_iters=level_iters,
                                sigmas=sigmas, factors=factors,
                                verbosity=verbosity)
    dtype_orig = static.dtype
    static = static - static.min()
    moving = moving - moving.min()
    static = static.astype(np.float64)
    moving = moving.astype(np.float64)
    if static_grid2world is None:
        static_grid2world = np.eye(4)
    static, static_nii = volToNifti(static, grid2world=static_grid2world)

    if moving_grid2world is None:
        moving_grid2world = np.eye(4)
    moving, moving_nii = volToNifti(moving, grid2world=moving_grid2world)

    print('Initial center of mass alignment...')
    com = transform_centers_of_mass(static, static_nii.affine,
                                    moving, moving_nii.affine)
    moving_com = com.transform(moving)
    reg = affreg.optimize(static, moving, AffineTransform3D(), params0,
                                  static_nii.affine, moving_nii.affine,
                                  starting_affine=com.affine)

    moving_affine = reg.transform(moving)
    moving_affine = np.transpose(moving_affine, (2,1,0))
    return moving_affine, reg


def threshold_multi(img, n=3):
    """
    Applies otsu iteratively to find multiple thresholds
    Parameters
    ----------
    img: array, (m [,n])
        Array of values such as an image
    n: scalar
        Number of thresholds to return
    Returns
    -------
    thr: array, (n,)
        Thesholds
    imq_quant: array, (m [,n])
        Quantized array, where discrete levels are values between thresholds.
    """
    from skimage.filters import threshold_otsu as otsu
    import numpy as np
    thr = np.zeros((n,))
    img_thr = img.ravel()
    img_quant = (img*0).astype(int)
    thr_up = np.infty
    for n_ in range(n):
        thr[n_] = otsu(img_thr)
        aboveInds = np.where((img > thr[n_]) & (img <= thr_up))
        thr_up = thr[n_]
        img_quant[aboveInds] = n-n_
        img_thr = np.delete(img_thr, np.where(img_thr > thr[n_]))
    return thr, img_quant


def volToNifti(vol, grid2world=None):
    """
    Niftify volume and return transposed vol as well as
    Nifti1 vol object.
    Parameters
    ----------
    vol: array, (nSlices, nRow, nCols)
        Original volume where dimensions are arranged in typical python format.
    grid2world: array of shape (4,4) or None
        Affine matrix that gives a mapping of the volumes grid to the world.
        If None, uses the identity matrix.
    Returns
    -------
    vol_new: array, (X,Y,Z)
        New volume with dimensions transposed.
    vol_nii: Nifti1 volume object
    """
    import nibabel as nib
    if grid2world is None:
        grid2world = np.eye(4)
    vol_nii = nib.Nifti1Image(np.transpose(vol, (2, 1, 0)), grid2world)
    vol = np.squeeze(vol_nii.get_data())
    return vol, vol_nii
