# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:45:14 2018

@author: pujalaa
"""
from sklearn.mixture import GaussianMixture as _GMM
from pathlib import Path
import dask
from dask.diagnostics import ProgressBar
import numpy as np
import os
import keras


def augmentImageData(images_train, images_mask, upsample=10,
                     aug_set=('rn', 'sig', 'log', 'inv', 'heq', 'rot', 'rs')):
    """
    Augment image data prior to training to increase the size of the training
    data, and minimize overfitting to particular dataset.
    Parameters
    ----------
    images_train: array, (nImgs, *imgDims)
        Training images.
    images_mask: array, (T, M, N)
        Masks for the training images
    upsample: scalar
        The factor by which the training set approximately grows after
        augmentation. If upsample = s, then the augmented training set
         will have s*T images
    aug_set: list of strings
        A list of strings wherein each string is an abbreviation for a type of
        augmentation. The allowed strings are
        'rn' - random noise; random noise is introduced into the images
        'rot' - rotation by a random angle between 0 and 360 degrees
        'sig' - sigmoid adjustment
        'log' - logarithmic adjustment
        'inv' - intensity inversion; dim pixels become bright and vice versa.

        'heq' - histogram equalization; should account for variations in
                lighting conditions.
        'rs' -  rescaling around mask and resizing back to old dimensions
        'sob' - Sobel filter to make the edges pronounced
        By default, each of these transormations ('rs' is omitted by default)
        is included with equal frequency, but this can be controlled by
        altering 'aug_set'. For instance, aug_set = ('rn', 'rn', 'rn', 'heq')
        will only introduce random noise injection and histogram equalization
        with the latter occurring thrice as often.
    Returns
    -------
    I_aug: array, (T_aug, M, N[,nChannels]), where T_aug = upsample*T
        Augmented training image set
    M_aug: array, (T_aug, M, N)
        Augmented mask set
    A_aug: List, (T_aug,)
        List of transformation applied to each of the images in the I_aug
    """
#    from skimage.util import random_noise
    from skimage import exposure
    from apCode.volTools import img as img_
#    from skimage.color import gray2rgb, rgb2gray
#    from skimage.exposure import rescale_intensity
    from skimage.filters import sobel
    from apCode.SignalProcessingTools import standardize
    upsample = np.max((1, upsample))
    nImgs = int(images_train.shape[0]*upsample)-images_train.shape[0]
    imgInds = np.random.choice(np.arange(images_train.shape[0]), size=nImgs,
                               replace=True)

    def map255(x): return (standardize(x)*255).astype(int)

    def apply_func(func, imgs, *args, **kwargs):
        if np.ndim(imgs) == 3:
            imgs = np.transpose(imgs, (2, 0, 1))
            imgs_trans = [func(_, *args, **kwargs) for _ in imgs]
            imgs_trans = np.array(imgs_trans).transpose(1, 2, 0)
        else:
            imgs_trans = func(imgs, *args, **kwargs)
        return imgs_trans
    images, M = [], []
    A = []
    count = 0
    for img, mask in zip(images_train[imgInds], images_mask[imgInds]):
        ind_ = np.mod(count, len(aug_set))
        aug_ = aug_set[ind_]
        if aug_ == 'rn':
            def func_now(x): return map255(x + np.random.randn(*x.shape)/7)
            images.append(dask.delayed(func_now)(img))
            M.append(mask)
            A.append(aug_)
        elif aug_ == 'rot':
            theta = int(np.random.choice(range(0, 360), size=1))

            def func_now(x):
                return apply_func(img_.rotate, x, theta, preserve_dtype=True)

            images.append(dask.delayed(func_now)(img))
            M.append(dask.delayed(func_now)(mask))
            A.append(aug_)
        elif aug_ == 'sig':
            def func_now(x):
                foo = apply_func(exposure.adjust_sigmoid, x.astype(int))
                foo = map255(foo)
                return foo
            images.append(dask.delayed(func_now)(img))
            M.append(mask)
            A.append(aug_)
        elif aug_ == 'log':
            def func_now(x):
                return apply_func(exposure.adjust_log, x)
            images.append(dask.delayed(func_now)(img))
            M.append(mask)
            A.append(aug_)
        elif aug_ == 'inv':
            def func_now(x): return x.max()-x
            images.append(dask.delayed(func_now)(img))
            M.append(mask)
            A.append(aug_)
        elif aug_ == 'heq':
            def func_now(x):
                return map255(apply_func(exposure.equalize_hist, x))
            images.append(dask.delayed(func_now)(img))
            M.append(mask)
            A.append(aug_)
        elif aug_ == 'rs':
            sf = np.random.choice(np.linspace(1.1, 1.5, 5), size=1)
            def func_now(x): return apply_func(rescaleIsometrically, x, sf)
            images.append(dask.delayed(func_now)(img))
            M.append(dask.delayed(func_now)(mask))
            A.append('rs')
        elif aug_ == 'sob':
            def func_now(x): return map255(apply_func(sobel, x))
            images.append(dask.delayed(func_now)(img))
            M.append(mask)
            A.append('sob')
        count += 1
    images = np.array(dask.compute(*images))
    masks = []
    for mask in M:
        if not isinstance(mask, np.ndarray):
            masks.append(mask.compute())
        else:
            masks.append(mask)
    masks = np.array(masks)
    augs = np.array(A)
    images = np.concatenate((images_train, images), axis=0)
    masks = np.concatenate((images_mask, masks), axis=0)
    augs = np.concatenate((np.array(['None']*images_train.shape[0]), augs))
    inds_shuffle = np.random.choice(np.arange(images.shape[0]),
                                    size=images.shape[0], replace=False)
    images, masks = images[inds_shuffle], masks[inds_shuffle]
    augs = augs[inds_shuffle]
    return images, masks, augs


def copyImgsNNTraining(imgDirs, prefFrameRangeInTrl=None,
                       nImgsForTraining: int = 50, overWrite: bool = False,
                       prefix_saved='images_train', copy: bool = True,
                       path_level=2):
    """ A convenient function for randomly selecting images from the specified
    list of directories and from within the range specified of frames
    (typically, peri-stimulus to maximize postural diversity) within each
    directory, and then writing those images in a directory labeled with the
    specfied prefix and suffixed with a timestamp.
    Parameters
    ----------
    rootDir: string
        Path to the root directory where all the fish images (in subfolders
        whose names end with "behav" by my convention) from a particular
        experimental are located.
    prefFrameRangeInTrl: 2-tuple
        Preferred range of frames for selecting images from within a trial
        directory. Typically, peri-stimulus range.
    nImgsForTraining: integer
        Number of images to select and copy
    overWrite:bool
        If True, overwites existing directory and saves images afresh.
    prefix_saved: str
        String prefix to use for the name of the directory where images
        for training will be copied to.
    copy: bool
        If True, will copy the images to the specified location
    path_level: int
        Specifies how many directory tree levels up from the location of where
        the training image folders are located to go to for saving/copying
        images.
        path_level = 0 will copy images to imgDirs[0]
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

    def join(p, x): return [os.path.join(p, _) for _ in x]
    files_sel = []
    for id in imgDirs:
        filesInDir = ft.findAndSortFilesInDir(id, ext='bmp')
        if np.any(prefFrameRangeInTrl is None):
            inds_sel = np.arange(len(filesInDir))
        else:
            inds_sel = np.arange(prefFrameRangeInTrl[0],
                                 prefFrameRangeInTrl[-1])
        if len(filesInDir) > np.max(inds_sel):
            files_sel.extend(join(id, filesInDir[inds_sel]))
        else:
            print(f'{len(filesInDir)} images in {id}, skipped')
    files_sel = np.random.choice(np.array(files_sel), size=nImgsForTraining,
                                 replace=False)
    ext = files_sel[0].split('.')[-1]
#    rootDir = os.path.split(imgDirs[0])[0]
    rootDir = str(Path(imgDirs[0]).parents[path_level])

    def rename(path_, n_, ext): return os.path.join(path_,
                                                    'img_{:05}.{}'.format(n_,
                                                                          ext))
    dst = os.path.join(rootDir, f'{prefix_saved}_{timestamp("min")}')
    dsts = [dask.delayed(rename)(dst, n, ext) for n, path_ in
            enumerate(files_sel)]
    if copy:
        if not os.path.exists(dst):
            os.mkdir(dst)
            dask.compute([dask.delayed(sh.copy)(src, dst_) for src, dst_ in
                          zip(files_sel, dsts)], scheduler='threads')
            np.save(os.path.join(dst,
                                 f'sourceImagePaths_{timestamp("min")}.npy'),
                    files_sel)
        else:
            if overWrite:
                from shutil import rmtree
                rmtree(dst)
                os.mkdir(dst)
                dask.compute([dask.delayed(sh.copy)(src, dst_) for src, dst_ in
                              zip(files_sel, dsts)], scheduler='threads')
                np.save(os.path.join(dst,
                                     f'sourceImagePaths_' +
                                     '{timestamp("min")}.npy'),
                        files_sel)
            else:

                print(f'{dst} already exists! Delete directory to re-select' +
                      'images or set overWrite = True')
    return files_sel, dst


def elastic_transform(image, alpha=5, sigma=1, alpha_affine=20,
                      random_state=None):
    """Elastic deformation of images as described in [Simard2003]
        (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    import cv2
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.ndimage.filters import gaussian_filter
    import numpy as np
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size,
                        center_square[1]-square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1],
                           borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
#    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
                          np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)),
    np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=1,
                           mode='reflect').reshape(shape)


def loadPreTrainedUnet(path_to_unet, search_dir=None, name_prefix='trainedU'):
    """
    Automatically searches for save uNet file in relevant path or specified
    path (overrides) and returns the loaded unet model object
    Parameters
    ----------
    path_to_unet: string or None
        If specified, returns the unet at the end of this path. If None,
        searches for unet file in search_dir.
    search_dir: string or None
        If specified ant path_to_unet == None, searches for unet file in this
        directory
    name_prefix: string
        The string used in filtering files that are possibly the unet files
    Returns
    -------
    unet: model object
        The pre-trained unet model object
    """
    from apCode.machineLearning.unet import model
    import apCode.FileTools as ft
    import os
    custom_objects = dict(dice_coef=model.dice_coef,
                          focal_loss=model.focal_loss)
    if path_to_unet is not None:
        print('Loading u net...')

        unet = model.load_model(path_to_unet, custom_objects=custom_objects)
    else:
        file_u = ft.findAndSortFilesInDir(search_dir, ext='h5',
                                          search_str=name_prefix)
        if len(file_u) > 0:
            file_u = file_u[-1]
            path_to_unet = os.path.join(search_dir, file_u)
            print('Loading unet...')
            unet = model.load_model(path_to_unet,
                                    custom_objects=custom_objects)
        else:
            print('No uNet found in search path, explicitly specify path')
            unet = None
    return unet


class PatchifyImgs:
    """
    Class with functions for converting an image stack of grayscale images into
    a 2D matrix useful for machine learning procedures
    Parameters
    ----------
    patchSize: 2-tuple-like or int
        The dimension of the patches to slide over images and convert into
        feature vectors
    stride: int or 2-tuple-like
        Determines the step size for sliding the window when extracting image
        patches
    n_jobs: integer
        Number of parallel workers
    verbose: integer
        Verbosity of the output when running parallel processes
    """

    def __init__(self, patchSize=(50, 50), stride=1, n_jobs=32, verbose=0):
        if np.ndim(patchSize) < 2:
            patchSize_ = np.ones((2,), dtype=int)*patchSize
        if np.ndim(stride) < 2:
            stride = np.ones((2, ), dtype=int)*stride
        self.patchSize_ = patchSize
        self.stride_ = stride
        self._n_jobs = n_jobs
        self._verbose = verbose

    def _im2col(self, img):
        """
        Converts patches of an image into feature vectors
        """
        import numpy as np
        self._imgShape = img.shape
        patchSize = self.patchSize_
        r = np.arange(0, img.shape[0], self.stride_[0])
        overInds = np.where((r+patchSize[0]) > img.shape[0])[0]
        r[overInds] = r[overInds]-(r[overInds]+patchSize[0]-img.shape[0])
        c = np.arange(0, img.shape[1], self.stride_[1])
        overInds = np.where((c + patchSize[1]) > img.shape[1])[0]
        c[overInds] = c[overInds]-(c[overInds] + patchSize[1]-img.shape[1])

        p, rc = [], []
        for r_now in r:
            r_inds = np.arange(r_now, r_now + patchSize[0])
            for c_now in c:
                c_inds = np.arange(c_now, c_now + patchSize[1])
                foo = img[r_inds[0]:r_inds[-1]+1, c_inds[0]:c_inds[-1]+1]
                p.append(foo.ravel())
                rc.append([r_inds, c_inds])
        self._rc = rc
        return np.array(p)

    def _col2im(self, patches):
        """
        Coverts from patch length feature vectors to original image
        """
        import numpy as np
        img = np.zeros(self.imgShape_)
        patchSize = self.patchSize_
        for num, rc in enumerate(self._rc):
            r_inds, c_inds = rc[0], rc[1]
            img[r_inds[0]:r_inds[-1]+1,
                c_inds[0]:c_inds[-1]+1] = patches[num].reshape(patchSize)
        return img

    def reshape_to_img_patches(self, I_patches):
        """
        Reshape 2D image_patches returned by patchifyImgs().transform into a
        more accessible 4D shape arranged by images, and individual patches.
        Parameters
        ----------
        I_patches: array (K, D), where K = nImages*nPatchesPerImage,
                    D = patchSize[0]*patchSize[1]
        Returns
        -------
        I_patches_new: array (nImages, nPatchesPerImage, patchSize[0],
                              patchSize[1])
        """
        return I_patches.reshape(self.nImages_, self.nPatches_,
                                 *self.patchSize_)

    def revert(self, I_patches) -> "(nImgs, *imgDims)":
        """
        Converts back from feature vector matrix to image stack
        Parameters
        ---------
        I_patches: array, (nImgs*nPatches, patchSize[0]*patchSize[1])
            Feature array where dimensionality of features is the number of
            pixels in a patch
        Returns:
        I: (nImgs, nRows, nCols)
        """
        import numpy as np
        nImgs = int(I_patches.shape[0]/self.nPatches_)
        I_patches = I_patches.reshape(nImgs, self.nPatches_,
                                      I_patches.shape[-1])
        if self._n_jobs < 2:
            I_recon = np.array([self._col2im(img_patches) for img_patches
                                in I_patches])
        else:
            from joblib import Parallel, delayed
            del_func = delayed(self._col2im)
            parallel = Parallel(n_jobs=self._n_jobs, verbose=self._verbose)
            I_recon = parallel(del_func(ip_) for ip_ in I_patches)
            I_recon = np.array(I_recon)
        return np.squeeze(I_recon)

    def revert_for_nn(self, images_patches) -> "(nImgs, *imgDims)":
        """
        Reverses transform_for_nn to give back original images
        """
        images = images_patches.reshape(images_patches.shape[0], -1)
        images = self.revert(images)
        return images

    def revert_preserve(self, images_patches) -> "(nImgs, nPatches, *patchDims)":
        """
        Reconstructs images from patches
        Parameters
        ----------
        I_patches: array, (nImages, nPatches, patchHeight, patchWidth)
        """
        m = images_patches.shape[0]*images_patches.shape[1]
        n = images_patches.shape[2]*images_patches.shape[3]
        images_patches = images_patches.reshape(m, n)
        return self.revert(images_patches)

    def transform(self, images) -> "(nImgs*nPatches, patchDims[0]*patchDims[1])":
        """
        Converts images into a feature matrix of dimensionality corresponding
        to the number of pixels in a patch.
        Parameters
        ----------
        imaegs: array, (T, M, N)
        Returns
        -------
        images_patches: array, (K,D)
            Where K is the total number of patches from all the images in the
            image stack (nImages*nPatchesPerImage)
            D is the dimensionality of the feature vectors
            (patchSize[0]*patchSize[1])
        """
        import numpy as np
        if np.ndim(images) == 2:
            images = images[np.newaxis, ::]
        if self._n_jobs < 2:
            images_patches = np.array([self._im2col(img) for img in images])
        else:
            from joblib import Parallel, delayed
            foo = self._im2col(images[0])  # Need this because running in
            # parallel does not append ._rc to
            # self, which is required for
            # reversion
            foo = foo*2  # To suppress code suggestion
            del_func = delayed(self._im2col)
            parallel = Parallel(n_jobs = self._n_jobs, verbose = self._verbose)
            images_patches = parallel(del_func(img) for img in images)
            images_patches = np.array(images_patches)
        self.nPatches_ = images_patches.shape[1]
        self.imgShape_ = images.shape[1:]
        self.nImages_ = images.shape[0]
        m = images_patches.shape[0]*images_patches.shape[1]
        images_patches = images_patches.reshape(m, images_patches.shape[2])
        return images_patches


    def transform_for_nn(self, images) -> "(nImgs*nPatchs, *patchDims)":
        """Convert images to patches in a format suitable for inputting to
        a U net
        Parameters
        ----------
        images: array, (nImgs, *imgDims)
        Returns
        -------
        images_patch: array, (nImgs*nPatches, *patchDims)
        """
        images_patches = self.transform(images)
        images_patches = images_patches.reshape(-1, *self.patchSize_)
        return images_patches

    def transform_preserve(self, images) -> "(nImgs, nPatches, *patchDims)":
        """
        Converts images into patches without concatenating patches from
        different images
        Parameters
        ----------
        I: array, (nImages, imageHeight, imageWidth)
        Returns
        -------
        I_patches: array, (nImages, nPatches, patchHeight, patchWidth)
        """
        I_patches = self.transform(images)
        return self.reshape_to_img_patches(I_patches)


def gmm_information_vs_nComponents(X, comps=5, warm_start=True,
                                   verbose=0, covariance_type='full',
                                   n_init=1, scheduler=None):
    """
    Given a dataset returns the AIC and BIC for upto specified number of
    components of the Gaussian Mixture Model(GMM) from sklearn.mixture
    Parameters
    ----------
    X: array, (n_samples,n_features)
        Dataset on which to run the GMM
    comps: Scalar or iterable
        Number of components over which to compute the AIC and BIC. If scalar,
        then computes IC from 1 to this number. If iterable, then each element
        of the iterable specifies the number of components to try
    warm_start: bool
        Determines if algorithm should be initialized from previous fit. See
        sklearn.mixture.GaussianMixture documentation.
    covariance_type: str
        'full': Each component has its own general covariance matrix
        'tied': All component have same covariance matrix
        'diag': Each component has own diagonal covariance matrix
        'sperical': Each compoment has its own single variance
    n_init: int
        Number of initializations
    scheduler: str or None
        Dask scheduler 'processes' or 'threads'
    Returns
    -------
    ic: dictionary
        Dictionary with fields 'aic' and 'bic' which gives these numbers
        for the different components
    """
    def getMetrics(X, gmm):
        gmm = gmm.fit(X)
        return (gmm.aic(X), gmm.bic(X))
    if np.ndim(comps) == 0:
        comps = np.arange(comps)+1
    metrics = []
    for iComp, comp in enumerate(comps):
        gmm = _GMM(n_components=comp, covariance_type=covariance_type,
                   n_init=n_init, verbose=verbose, warm_start=warm_start)
        foo = dask.delayed(getMetrics)(X, gmm)
        metrics.append(foo)
    with ProgressBar():
        if scheduler is not None:
            metrics = np.array(dask.compute(*metrics), scheduler=scheduler)
        else:
            metrics = np.array(dask.compute(*metrics))
    ic = {'aic': metrics[:, 0], 'bic': metrics[:, 1]}
    return ic


class plot(object):
    def gmm(model, X, label=True, ax=None, cmap='viridis',
            patchColor=None, **kwargs):
        """
        Plots the results of the Gaussian mixture model obtained using
            sklearn.mixture.GaussianMixture
        as semi-transparent ellipses. The data must be in 2D space
        Parameters:
        ----------
        model - Gaussian mixture model object specified as follows:
            model = GMM(n_components = n_components)
        X - The original 2-D dataset
        label - Boolean; If true, labels the data points in the scatter plot
        ax - Axes object; If given, then plots on specified axes, else on the
             current axes
        cmap - String or colormap object; Specifies the colormap to use for the
               scatter plot
        patchColor - String or 3- or 4-tuple; Color of the elliptical patches
        Taken from:
        VanderPlas, J. Python Data Science Handbook (2016).
        """
        import matplotlib.pyplot as plt
        from apCode.volTools import plot as plot
        import numpy as np
        draw_ellipse = plot.draw_ellipse
        ax = ax or plt.gca()
        labels = model.fit(X).predict(X)
        if label:
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, zorder=2,
                       **kwargs)
        else:
            ax.scatter(X[:, 0], X[:, 1], zorder=2, **kwargs)
        w_factor = 0.15 / model.weights_.max()
        for pos, covar, w in zip(model.means_, model.covariances_,
                                 model.weights_):
            draw_ellipse(pos, covar, alpha=np.sqrt(w) * w_factor,
                         color=patchColor)


def plotMontageOfImageCollections(images, images_prob, nCols: int = 4,
                                  rescale_intensity: bool = True):
    """
    Convenience function for plotting a montage of images and their
    correponding probability maps (predicted by a neural network) or masks
    (for training a neural network) so as to visually assess performance or
    correpondence between masks and images
    Parameters
    ----------
    images, images_prob: arrays, (numberOfImages, imageHeight, imageWidth)
        Image collections to plot side-by-side in montage
    nCols: int
        Number of columns in the montage. Number of rows is computed
        automatically.
    rescale_intensity: bool
        If True, rescales the intensities of each of the images in the montage
    Returns:
    m: array, (imageHeight*nRows,2*imageWidth*nCols)
        The montage of images
    """
    from skimage.util import montage
    import matplotlib.pyplot as plt

    if len(images)/nCols == 0:
        nRows = len(images)//nCols
    else:
        nRows = len(images)//nCols + 1
    ar = 0.5*images[0].shape[0]/images[0].shape[1]
    figLen = int(20*ar*nRows/nCols)
    m = [montage((img, img_prob), grid_shape=(1, 2),
                 rescale_intensity=rescale_intensity) for img, img_prob
         in zip(images, images_prob)]
    m = montage(m, grid_shape=(nRows, nCols),
                rescale_intensity=rescale_intensity)
    plt.figure(figsize=(20, figLen))
    plt.imshow(m)
    return m


class GMM(_GMM):
    """
    My wrapper for sklearn.mixture.GaussianMixture. Offers some additional
    useful methods
    """

    def get_components(self, X):
        """
        Given a Gaussian Mixture Model (GMM), returns the additive components
        of the model
        Parameters
        ----------
        model - GMM object fitted to the data; For e.g.,
            from sklearn.mixture import GaussianMixture as GMM
            model = GMM(n_components = 3).fit(X), where X is the input data
        X - Array of size (n_samples,n_features) to which the GMM was fit.

        Returns
        -------
        components - Array of size (n_samples,n_components); the Gaussian
        components weighted, and added to fit the data
        """
        import numpy as np
        if np.ndim(X) == 1:
            X = X.reshape((-1, 1))
        N = np.shape(X)[0]
        Y = np.array([np.linspace(np.min(feature), np.max(feature), N)
                      for feature in X.T]).T
        responsibilities = self.predict_proba(Y)
        pdf = np.exp(self.score_samples(Y))
        if np.ndim(pdf) == 1:
            pdf = pdf.reshape((-1, 1))
        comp = responsibilities*pdf
        return comp

    @staticmethod
    def information_versus_nComponents(X, comps=5, verbose=False):
        """
        Given a dataset returns the AIC and BIC for upto specified number of
        components of the Gaussian Mixture Model(GMM) from sklearn.mixture
        Parameters
        ----------
        X: array, (n_samples,n_features)
            Dataset on which to run the GMM
        comps: Scalar or iterable
            Number of components over which to compute the AIC and BIC. If
            scalar, then computes IC from 1 to this number. If iterable, then
            each element of the iterable specifies the number of components to
            try
        Returns
        -------
        ic: dictionary
            Dictionary with fields 'aic' and 'bic' which gives these numbers
            for the different components
        """
        import numpy as np

        def getMetrics(X, model):
            # model = GMM(n_components = n_components).fit(X)
            model = model.fit(X)
            return (model.aic(X), model.bic(X))
        if np.ndim(comps) == 0:
            comps = np.arange(comps)+1
        metrics = []
        for iComp, comp in enumerate(comps):
            if verbose:
                print(f'{iComp}/{len(comps)}; comp = {comp}')
            model = GMM(n_components=comp)
            model.n_components = comp
            foo = dask.delayed(getMetrics)(X, model)
            metrics.append(foo)
        with ProgressBar():
            metrics = np.array(dask.compute(*metrics))
        ic = {'aic': metrics[:, 0], 'bic': metrics[:, 1]}
        return ic

    def relabel_by_norm(self, labels):
        return self.relabel_by_sorted(labels, self.sorted_labels())

    def relabel_by_sorted(self, labels, orderedLabels):
        """
        Given an array of labels, relabels with respect to the specified label
        order such that labels == n --> orderedLabels[n].
        Parameters
        ----------
        labels: array, (N,)
            1D array of labels with at most k unique values.
        orderedLabels: array, (k,)
            The unique set of all labels in the desired order
        Returns
        --------
        labels_sorted: array, (N,)
            Labels relabeled to match the specified order
        """
        import numpy as np
        il = []
        for ind, ol in enumerate(orderedLabels):
            il.append((np.where(labels == ol)[0], ind))
        labels_sorted = labels.copy()
        for il_ in il:
            labels_sorted[il_[0]] = il_[1]
        return labels_sorted

    def sorted_labels(self):
        """
        Returns the labels in order (descending) sorted by the norms (means in
        1D case) of the mean vector (dimensionality = n_features) of each of
        the components of the fit model.
        """
        import numpy as np
        if not hasattr(self, 'means_'):
            print('The model must be fitted first!')
            return None
        norms_gmm_means = np.linalg.norm(self.means_, axis=1)
        labels_sorted = np.argsort(norms_gmm_means)[::-1]
        self.orderedLabels_ = labels_sorted
        return labels_sorted


def prepare_imgs_for_unet(imgs, unet):
    """
    Prepare images to be input to a U net
    Parameters
    ----------
    imgs: array, (nImgs, *imgDims[, nCh])
        Images
    unet: object
        U net model
    Returns
    -------
    imgs_u: array, (nImgs, *imgsDims_new, nCh)
        Images ready to be input to U net
    """
    import apCode.volTools as volt
    uDims = unet.input_shape[1:3]
    if np.ndim(imgs)==3:
        nCh = 1
    elif np.ndim(imgs)==4:
        nCh = imgs.shape[-1]
    elif np.ndim(imgs)==2:
        nCh = 1
    else:
        raise IOError('Check images dimension: Must be between 2 and 4')
    imgDims = imgs.shape[1:3]
    d = np.sum((np.array(uDims)-np.array(imgDims))**2)
    if d > 0:
        imgs_rs = volt.img.resize(imgs, uDims)
    else:
        imgs_rs = imgs
    if nCh==1:
        imgs_rs = imgs_rs[..., None]
    return imgs_rs

def prepareImagesForUnet(images, sz=(512, 512)):
    """
    Resizes and adjusts dimensions of image or image stack for training or
    predicting with u-net
    """
    import numpy as np
    from apCode.volTools import img as im
    if np.ndim(images) == 2:
        images = images[np.newaxis, ..., np.newaxis]
    elif np.ndim(images) == 3:
        images = images[np.newaxis, ...]

    images_new = []
    for c in range(images.shape[-1]):
        images_new.append(im.resize(images[..., c], sz, preserve_dtype=True))
    images_new = np.transpose(np.array(images_new), (1, 2, 3, 0))
    return images_new


def readAllTrainingImagesAndMasks(rootDir,
                                  prefix_trainingDir: str = 'imgs_train',
                                  save: bool = True):
    """
    Within the specified root directory ("rootDir") searches for all training
    images and corresponding  masks and read these into two separate image
    arrays that can be used for training a potentially generalizable neural
    network. The directories with traning images are searched based on the
    specified prefix ("prefix_trainingDir") which is assumed to exist in the
    names of the training directories. The mask directories are found
    similiarly, but the names of the mask directories are thought to differ
    only in the substitution of "train" with "mask". For eg., if a traninging
    directory has the name "imgs_train-201909018-0221", then the corresponding
    mask directory is thought to have the name "imgs_mask-20190918-0221.zip".
    Note the extension .zip indicates that the masks were created using ImageJ.
    Currently, the script expects this. In the future, the program may become
    more general and read other roi types.
    Parameters
    ----------
    rootDir: string
        The path to the root directory whose subdirectories are recursively
        scanned for training images and masks.
    prefix_trainingDir: str
        The string to match when filtering subdirectories for training images
        directories.
    save: bool
        If true, save images and masks in a hdf file in the root directory
        with the name trainingImagesAndMasks.h5
    Returns
    -------
    out: dictionary
        Has the following key-value pairs:
        images_train, masks_train: arrays, (numberOfImages, imageHeight,
                                            imageWidth)
    """
    from apCode.FileTools import findAndSortFilesInDir
    from apCode.volTools import img as im
    from apCode.util import timestamp
    from apCode.hdf import save_dict_to_hdf5
    trainDirs = [x[0] for x in os.walk(rootDir) if prefix_trainingDir in x[0]]

    # Iterate through each training directory and find matching mask directory
    trainMaskDirs = []
    for td in trainDirs:
        dir_, name_ = os.path.split(td)
        name_mask = name_.replace('train', 'mask')
        md = findAndSortFilesInDir(dir_, ext='.zip', search_str=name_mask)
        if len(md) > 0:
            trainMaskDirs.append([td, os.path.join(dir_, md[-1])])

    # Read training images and masks from each pair of directories and store in
    # two separate arrays
    images_train, masks_train = [], []
    for tmd in trainMaskDirs:
        print(f'Reading from {os.path.split(tmd[0])[0]}')
        imgs_ = im.readImagesInDir(tmd[0])
        masks_ = readImageJRois(tmd[1], imgs_.shape[-2:])[0][:imgs_.shape[0]]
        print(f'{imgs_.shape[0]} images, {masks_.shape[0]} masks')
        images_train.extend(imgs_)
        masks_train.extend(masks_)
    out = dict(images_train=np.array(images_train),
               masks_train=np.array(masks_train))
    if save:
        hFilePath = os.path.join(rootDir,
                                 f'trainingImagesAndMasks_{timestamp()}.h5')
        save_dict_to_hdf5(out, hFilePath)
        print(f'Saved to hdf file at {hFilePath}')
    return out


def read_training_images_and_masks(dirs_img, dirs_mask, imgDims=(256, 256)):
    """
    Read images and masks from lists of directories and resize to specified
    dimensions if need be.
    Parameters
    ----------
    dirs_img: list-like
        List of directories containing training images
    dirs_mask: list-like
        List of directories containing masks
    imgDims: tuple or None
        Dimensions of the output images and masks. If None, then does not
        resize
    """
    import apCode.volTools as volt
    imgs, masks = [], []
    for dir_img, dir_mask in zip(dirs_img, dirs_mask):
        imgs_ = volt.img.readImagesInDir(dir_img)
        masks_ = readImageJRois(dir_mask, imgs_.shape[-2:])[0][:imgs_.shape[0]]
        if imgDims is not None:
            if np.sum(np.abs(np.array(imgDims)-np.array(imgs_.shape[-2:])))>0:
                imgs_ = volt.img.resize(imgs_, imgDims)
                masks_ = volt.img.resize(masks_, imgDims)
        imgs.append(imgs_)
        masks.append(masks_)
    if imgDims is not None:
        imgs = np.concatenate(imgs, axis=0)
        masks = np.concatenate(masks, axis=0)
    return imgs, masks


def read_training_images_and_masks_2ch(dirs_img, dirs_mask,
                                       imgDims=(896, 896), back_func='mean'):
    """
    Read images and masks from lists of directories and resize to specified
    dimensions if need be.
    Parameters
    ----------
    dirs_img: list-like
        List of directories containing training images
    dirs_mask: list-like
        List of directories containing masks
    imgDims: tuple or None
        Dimensions of the output images and masks. If None, then does not
        resize
    back_func: str, 'max', 'min', 'mean' or 'median'
    """
    import apCode.volTools as volt
    imgs, masks = [], []
    for dir_img, dir_mask in zip(dirs_img, dirs_mask):
        imgs_ = volt.img.readImagesInDir(dir_img)
        masks_ = readImageJRois(dir_mask, imgs_.shape[-2:])[0][:imgs_.shape[0]]
        func = eval(f'np.{back_func}')
        bgd = func(imgs_, axis=0)[np.newaxis,...]
        # bgd = imgs_.max(axis=0)[np.newaxis,...]
        if imgDims is not None:
            if np.sum(np.abs(np.array(imgDims)-np.array(imgs_.shape[-2:])))>0:
                imgs_ = volt.img.resize(imgs_, imgDims)
                masks_ = volt.img.resize(masks_, imgDims)
                bgd = volt.img.resize(bgd, imgDims)
        bgd = np.tile(bgd, [len(imgs_), 1, 1])[..., np.newaxis]
        imgs_ = imgs_[..., np.newaxis]
        imgs_bgd = np.concatenate((imgs_, bgd), axis=3)
        imgs.append(imgs_bgd)
        masks.append(masks_)
    if imgDims is not None:
        imgs = np.concatenate(imgs, axis=0)
        masks = np.concatenate(masks, axis=0)
    return imgs, masks


def readImageJRois(roiZipPath, imgDims, multiLevel=True, levels=None):
    """
    Reads ROI .zip file saved by ImageJ ROI manager and returns these as
    well as image masks. Assumes that polygonal ROIs have been used.
    Parameters
    ----------
    roiZipPath: string
        Path to the .zip folder containing ImageJ ROIs saved by ROI manager
    imgDims: 2-tuple
        Dimensions of the images associated with the ROIs
    Returns
    -------
    I: array, (T,*imgDims)
        Image stack of T mask
    rois: ordered dict
        Ordered dictionary containing information abour ImageJ Rois. They keys
        are the names/labels of the ROIs
    """
    from read_roi import read_roi_zip
    import numpy as np

    def ellipseCoords(roi, imgDims):
        from skimage.draw import ellipse
        c = (roi['left']+roi['left']+roi['width'])*0.5
        r = (roi['top']+roi['top']+roi['height'])*0.5
        r_radius = roi['height']/2
        c_radius = roi['width']/2
        rr, cc = ellipse(r, c, r_radius, c_radius)
        inds_del = np.where((rr < 0) | (rr >= imgDims[0]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        inds_del = np.where((cc < 0) | (cc >= imgDims[1]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        return rr, cc

    def polyCoords(roi, imgDims):
        from skimage.draw import polygon
        poly = np.array([roi['x'], roi['y']])
        rr, cc = polygon(poly[1], poly[0], imgDims)
        inds_del = np.where((rr < 0) | (rr >= imgDims[0]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        inds_del = np.where((cc < 0) | (cc >= imgDims[1]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        return rr, cc

    def rectCoords(roi, imgDims):
        from skimage.draw import rectangle
        start = (roi['top'], roi['left'])
        extent = (roi['height'], roi['width'])
        rr, cc = rectangle(start, extent=extent)
        inds_del = np.where((rr < 0) | (rr >= imgDims[0]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        inds_del = np.where((cc < 0) | (cc >= imgDims[1]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        return rr, cc

    rois = read_roi_zip(roiZipPath)
    roiKeys = np.array(list(rois.keys()))
    zVec = []
    for _ in roiKeys:
        foo = rois[_]['position']
        if isinstance(foo, dict):
            zVec.append(foo['slice'])
        else:
            zVec.append(foo)
    zVec = np.array(zVec)
    images = np.zeros((zVec.max(), *imgDims), dtype=np.int)
    count = 0
    for z in np.unique(zVec):
        inds = np.where(zVec == z)[0]
        if levels is not None:
            lvlInds = levels
        else:
            lvlInds = np.arange(len(inds))
        rks = roiKeys[inds[lvlInds]]
        for iRk, rk in enumerate(rks):
            lvl = lvlInds[iRk]
            roi_ = rois[rk]
            mask = np.zeros(imgDims, dtype=np.int)
            rr, cc = [], []
            if roi_['type'] in ('polygon', 'freehand'):
                rr, cc = polyCoords(roi_, imgDims)
                rois[rk]['coords'] = (rr, cc)
                mask[rr, cc] = 1
                rois[rk]['mask'] = mask
            elif roi_['type'] == 'oval':
                rr, cc = ellipseCoords(roi_, imgDims)
                rois[rk]['coords'] = (rr, cc)
                mask[rr, cc] = 1
                rois[rk]['mask'] = mask
            elif roi_['type'] == 'rectangle':
                rr, cc = rectCoords(roi_, imgDims)
                rois[rk]['coords'] = (rr, cc)
                mask[rr, cc] = 1
                rois[rk]['mask'] = mask
            else:
                print(r'Could not extract coordinates for roi # {} of ' +
                      'type {}'.format(count, roi_['type']))
                rois[rk]['coords'] = ()
                rois[rk]['mask'] = mask
            count = count + 1
            if multiLevel:
                images[z-1, rr, cc] = lvl+1
            else:
                images[z-1, rr, cc] = 1
    return np.squeeze(np.array(images)), rois


def readImageJRois_old(roiZipPath, imgDims):
    """
    Reads ROI .zip file saved by ImageJ ROI manager and returns these as
    well as image masks. Assumes that polygonal ROIs have been used.
    Parameters
    ----------
    roiZipPath: string
        Path to the .zip folder containing ImageJ ROIs saved by ROI manager
    imgDims: 2-tuple
        Dimensions of the images associated with the ROIs
    Returns
    -------
    I: array, (T,*imgDims)
        Image stack of T mask
    rois: ordered dict
        Ordered dictionary containing information abour ImageJ Rois. They keys
        are the names/labels of the ROIs
    """
    from read_roi import read_roi_zip
    import numpy as np

    def ellipseCoords(roi, imgDims):
        from skimage.draw import ellipse
        c = (roi['left']+roi['left']+roi['width'])*0.5
        r = (roi['top']+roi['top']+roi['height'])*0.5
        r_radius = roi['height']/2
        c_radius = roi['width']/2
        rr, cc = ellipse(r, c, r_radius, c_radius)
        inds_del = np.where((rr < 0) | (rr >= imgDims[0]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        inds_del = np.where((cc < 0) | (cc >= imgDims[1]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        return rr, cc

    def polyCoords(roi, imgDims):
        from skimage.draw import polygon
        poly = np.array([roi['x'], roi['y']])
        rr, cc = polygon(poly[1], poly[0], imgDims)
        inds_del = np.where((rr < 0) | (rr >= imgDims[0]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        inds_del = np.where((cc < 0) | (cc >= imgDims[1]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        return rr, cc

    def rectCoords(roi, imgDims):
        from skimage.draw import rectangle
        start = (roi['top'], roi['left'])
        extent = (roi['height'], roi['width'])
        rr, cc = rectangle(start, extent=extent)
        inds_del = np.where((rr < 0) | (rr >= imgDims[0]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        inds_del = np.where((cc < 0) | (cc >= imgDims[1]))[0]
        rr = np.delete(rr, inds_del)
        cc = np.delete(cc, inds_del)
        return rr, cc

    rois = read_roi_zip(roiZipPath)
    roiKeys = np.array([_ for _ in rois.keys()])
    zVec = []
    for _ in roiKeys:
        foo = rois[_]['position']
        if isinstance(foo, dict):
            zVec.append(foo['slice'])
        else:
            zVec.append(foo)
    zVec = np.unique(np.array(zVec))
    images = np.zeros((zVec.max(), *imgDims))
    count = 0
    for rk in roiKeys:
        roi_ = rois[rk]
        mask = np.zeros(imgDims, dtype=np.int)
        rr, cc = [], []
        if roi_['type'] in ('polygon', 'freehand'):
            rr, cc = polyCoords(roi_, imgDims)
            rois[rk]['coords'] = (rr, cc)
            mask[rr, cc] = 1
            rois[rk]['mask'] = mask
        elif roi_['type'] == 'oval':
            rr, cc = ellipseCoords(roi_, imgDims)
            rois[rk]['coords'] = (rr, cc)
            mask[rr, cc] = 1
            rois[rk]['mask'] = mask
        elif roi_['type'] == 'rectangle':
            rr, cc = rectCoords(roi_, imgDims)
            rois[rk]['coords'] = (rr, cc)
            mask[rr, cc] = 1
            rois[rk]['mask'] = mask
        else:
            print('Could not extract coordinates for roi # {} of ' +
                  'type {}'.format(count, roi_['type']))
            rois[rk]['coords'] = ()
            rois[rk]['mask'] = mask
        count = count + 1
        z = np.where(zVec == roi_['position'])[0]
        images[z, rr, cc] = 255
    return np.squeeze(np.array(images)), rois


def rescaleIsometrically(images, *args, pad_type='mean', **kwargs):
    from skimage.transform import rescale
    from apCode.volTools import cropND

    def rescale_and_crop(img, *args, **kwargs):
        img_dims = img.shape
        img_dtype = img.dtype
        kwargs['preserve_range'] = kwargs.get('preserve_range', True)
        kwargs['multichannel'] = kwargs.get('multichannel', False)
        img_rs = rescale(img, *args, **kwargs)
        dims_diff = np.array(img_rs.shape) - np.array(img_dims)
        if np.all(dims_diff > 0):
            img_rs = cropND(img_rs, img_dims)
        else:
            dims_diff = np.abs(dims_diff)
            pre_one = dims_diff[0]//2 if np.mod(dims_diff[0], 2) == 0 else\
                dims_diff[0]//2 + 1
            post_one = int(dims_diff[0] - pre_one)
            pre_two = dims_diff[1]//2 if np.mod(dims_diff[1], 2) == 0 else\
                dims_diff[1]//2 + 1
            post_two = int(dims_diff[1]-pre_two)
            img_rs = np.pad(img_rs, ((pre_one, post_one), (pre_two, post_two)),
                            mode=pad_type)
        if kwargs['preserve_range']:
            img_rs = img_rs.astype(img_dtype)
        return img_rs
    if np.ndim(images) == 2:
        images = images[np.newaxis, ...]
    images = [dask.delayed(rescale_and_crop)(img, *args, **kwargs)
              for img in images]
    images = dask.compute(*images)
    return np.squeeze(np.array(images))


def rescale(images, masks, pad_type='edge', n_jobs=32, verbose=0):
    """
    Given a set of images and masks returns randomly rescaled images with same
    dimensions as the original images so that a neural network (such as a
    U-net), or the like, can be trained on images where the object of interest
    is present at different scales. This can be a type of data augmentation
    operation to help generalize a classifier and make it more robust.
    Parameters
    ----------
    images: array, ([T], M, N)
        Images to be trained on.
    masks: array ([T], M, N)
        Masks that have non-zero values only for pixels where object of
        interest is present
    pad_type: string, see numpy.pad for options
        Determines the type of padding when images are cropped before resizing.
    n_jobs: scalar
        # of parallel workers
    verbose: scalar
        Verbosity of parallel loop output.
    Returns
    -------
    images_rs, masks_rs: Rescaled images and masks
    """
    import numpy as np
    import apCode.volTools as volt

    def rescale_(img, mask, pad_type='edge'):
        coords = np.nonzero(mask)
        img_sub = img[coords[0].min():coords[0].max(),
                      coords[1].min():coords[1].max()]
        dim_max = np.max(img_sub.shape)
        cent = np.flipud(np.median(coords, axis=1))
        dSize = np.array(img.shape)-dim_max
        sz = np.random.choice(np.arange(dim_max, dim_max+dSize.max()-1))
        img_crop = volt.img.cropImgsAroundPoints(img, cent, cropSize=sz,
                                                 pad_type=pad_type)
        mask_crop = volt.img.cropImgsAroundPoints(mask, cent, cropSize=sz,
                                                  pad_type=pad_type)
        img_crop = volt.img.resize(img_crop, img.shape)
        mask_crop = volt.img.resize(mask_crop, mask.shape)
        return img_crop, mask_crop
    if np.ndim(images) == 2:
        img_crop, mask_crop = rescale_(images, masks, pad_type=pad_type)
        return img_crop, mask_crop
    else:
        if len(images) < n_jobs:
            IM = np.array([rescale_(img, mask, pad_type=pad_type)
                           for img, mask in zip(images, masks)])
            images, M = IM[:, 0], IM[:, 1]
        else:
            from joblib import Parallel, delayed
            IM = Parallel(n_jobs=n_jobs, verbose=0)(delayed(rescale_)(
                img, mask, pad_type=pad_type) for img, mask in
                zip(images, masks))
            images, M = IM[:, 0], IM[:, 1]
        return images, M


def retrainU(uNet, imgsOrPath, masksOrPath, upSample=6, imgExt: str = 'bmp',
             n_jobs: int = 32, verbose: int = 0, epochs: int = 50,
             saveModel: bool = True, multiLevel=False,
             aug_set=('rn', 'sig', 'log', 'inv', 'heq', 'rot', 'rs'),
             checkPoint=False, **fit_kwargs):
    """
    Further trains a pre-trained U net model
    Parameters
    ----------
    uNet: Keras U net model
    imgsOrPath: string or array of shape (numberOfImages, imageHeight,
                                          imageWidth)
        Training image array or path to folder containing training images
    masksOrPath: string or array of same shape as imgsOrPath
        Array of image masks or path to .zip folder containing ROIs generated
        in ImageJ
    upSample: scalar
        Factor by which to increase the number of training images during
        augmentation step
    imgExt: string
        Extension of the training images for reading properly
    n_jobs, verbose: See Parallel, delayed from joblib
    epochs: Scalar
        Number of training epochs when fitting U net
    saveModel: bool
        If True (default), saves the model as an hdf (.h5) file with
        timestamped filename
    checkPoint: bool
        If True, then uses ModelCheckPoint callback from keras to store
        the weights from the best epoch
    Returns
    -------
    uNet: The trained uNet
    """
    import time
    import apCode.volTools as volt
    from apCode.machineLearning import ml
    import apCode.behavior.FreeSwimBehavior as fsb
    if not isinstance(imgsOrPath, np.ndarray):
        images = volt.img.readImagesInDir(imgsOrPath, verbose=0, ext=imgExt)
    else:
        images = imgsOrPath
    imgShape = images.shape
    if not isinstance(masksOrPath, np.ndarray):
        masks = ml.readImageJRois(masksOrPath, imgShape[1:],
                                  multiLevel=False)[0][:images.shape[0]]
    else:
        masks = masksOrPath

    print('Augmenting...')
    images, masks = ml.augmentImageData(images, masks, upsample=upSample,
                                        aug_set=aug_set)[:2]

    images = prepare_imgs_for_unet(images, uNet)
    masks = prepare_imgs_for_unet(masks, uNet)
    masks_max = masks.max()
    if (not multiLevel) & (masks_max > 1):
        masks = (masks/masks_max).astype(int)
    print('Training unet...')
    if keras.backend.backend().lower()=='tensorflow':
        # Define Tensorboard as a Keras callback
        from keras.callbacks import TensorBoard
        tensorboard = TensorBoard(log_dir='.\logs', histogram_freq=1,
                                  write_images=True)
        keras_callbacks = [tensorboard]
    if checkPoint:
        from keras.callbacks import ModelCheckpoint
        checkpointer = ModelCheckpoint(filepath="best_weights.hdf5",
                                       monitor = 'val_acc',
                                       verbose=1,
                                       save_best_only=True)
        print('Checkpointing!')
        uNet.fit(images, masks, verbose=verbose, epochs=epochs,
                 callbacks=[checkpointer], **fit_kwargs)
    else:
        uNet.fit(images, masks, verbose=verbose, epochs=epochs, **fit_kwargs)
    if saveModel:
        dir_save = os.path.split(imgsOrPath)[0]
        file_save = 'trainedUnet_{}.h5'.format(time.strftime('%Y%m%dT%H%m'))
        uNet.save(os.path.join(dir_save, file_save))
        print('Saved trained model at {} as \n{}'.format(dir_save, file_save))
    return uNet
