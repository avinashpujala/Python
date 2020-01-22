import numpy as np
from skimage.filters import threshold_otsu


def getGlobalThr(img, tol=0.002, nIter_max=100, thr0=None):
    '''
    Gets an image or vector, uses an iterative algorithm to find a single
    threshold, which can be used for image binarization or noise discernment in
    a signal, etc.

    Parameters
    ----------
    img : image / vector
        input image
    tol : float
        Tolerance for iterations. If an iteration yields a value for threshold
        that differs from value from previous iteration by less than the
        tolerance value then iterations stop.
    nMaxIter : int
        Max # of iterations to run; can be used to prematurely terminate the
        iterations or if the values do not reach tolerance by a certain number
        of iterations.
    thr0 : float
        Initial threshold value from whence to begin looking for threshold. If
        empty, uses Otsu's method to find a starting value
    '''

    def DiffThr(img, thr0):
        sub = img[img < thr0]
        supra = img[img >= thr0]
        thr = 0.5*(np.mean(sub) + np.mean(supra))
        return thr

    if thr0 is None:
        thr0 = threshold_otsu(img)

    count = 0
    thr1 = DiffThr(img,thr0)
    dThr = np.abs(thr1-thr0)
    while (dThr > tol) & (count < nIter_max):
        thr = thr1
        thr1 = DiffThr(img,thr)
        dThr = np.abs(thr1-thr)
        count = count + 1
        # print('Iter #', str(count), 'dThr = ', str(dThr*100))
    return thr1

def get_multi_threshold(img, nThr, **kwargs):
    """
    Get multiple thresholds for image quantization.

    Parameters
    ----------
    img : ndarray image
        Image to quantize and get thresholds for.
    nThr : int
        Number of threshold or levels of quantization
    tol : float (optional, default = 0.002)
        tolerance (see `GetGlobalThr`)
    nMaxIter : int (optional, default = 100)
        Max # of iterations for each threshold
    minThr : UNKNOWN_DTYPE (optional, default = 0)
        Minimum threshold. If next threshold level falls below this value, then
        skips iteration
    Returns
    -------
    thr : ndarray (length nThr)
        A sequence of thresholds
    img_quant : ndarray (same dimensions as input image)
        Quantized image

    """
    tol = kwargs.get('tol', 0.002)
    nMaxIter = kwargs.get('nMaxIter', 100)
    minThr = kwargs.get('minThr', 0)

    thr = np.zeros(nThr)
    imgDims = img.shape
    nPxls = np.product(imgDims) # this is the split
    img_quant = np.zeros(imgDims)
    for jj in range(nThr):
        # print(f"Getting threshold #{jj}")
        thr0 = threshold_otsu(img)
        thr[jj] = getGlobalThr(img, tol, nMaxIter, thr0)
        (one_rs, one_cs) = (img > thr[jj]).nonzero()
        if (0 < len(one_rs) < nPxls) and thr[jj] > minThr:
            img_quant[one_rs, one_cs] = nThr - jj + 1
            img[one_rs, one_cs] = 1
        else:
            # print(f'Only {jj-1} levels of quantization achieved!')
            break
    return (thr, img_quant)
