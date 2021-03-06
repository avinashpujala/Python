

#import sys
#sys.path.insert(1, 'C:/Users/pujalaa/Documents/Code/Python/code/codeFromNV')
#sys.path.insert(1, 'C:/Users/pujalaa/Documents/Code/Python/code/util')

def angleBetween2DVectors(v1,v2):
    '''
    Given a list or array of 2D vectors, returns the angle (in radians) between 
        each of the vectors such that a sweep from from the 1st vec to the second
        2nd vec in the counterclockwise direction returns negative angles whereas a 
        sweep in the clockwise direction results in positive angles
    Inputs:
    v1, v2 - The 2 input vectors of size N x 2
    '''
    import numpy as np
    v1,v2 = np.array(v1), np.array(v2)
    if len(np.shape(v1))>1:
        if np.shape(v1)[1] !=2:
            v1 = np.transpose(v1)
        if np.shape(v2)[1] !=2:
            v2 = np.transpose(v2)
        v1 = v1[:,0] + v1[:,1]*1j
        v2= v2[:,0] + v2[:,1]*1j
    else:
        v1 = v1[0] + v1[1]*1j
        v2 = v2[0] + v2[1]*1j
    angle =  np.angle(v1*np.conj(v2))
    return angle   

def animate_images(I, fps = 30, display = True, save = False, 
                   savePath = None,**kwargs):
    """
    Movie from an image stack
    Parameters
    ----------
    I: array, (T,M,N)
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
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation
    plt.rcParams['animation.ffmpeg_path'] = r'V:\Code\Python\FFMPEG\bin\ffmpeg.exe'
    from IPython.display import HTML
    import time
    
    N = I.shape[0]    
    cmap = kwargs.get('cmap', 'gray')
    interp = kwargs.get('interpolation', 'nearest')
    dpi = kwargs.get('dpi', 30)
    plt.style.use(('seaborn-poster', 'dark_background'))
    fh = plt.figure(dpi = dpi, facecolor='k')
    ax = fh.add_subplot(111, frameon = False)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#    ax.set_frame_on(False)
    im = ax.imshow(I[0], cmap= cmap, interpolation = interp, 
                   vmin = I.min(), vmax = I.max())    
    
    def update_img(n):        
        im.set_data(I[n])   
        ax.set_title('Frame # {}'.format(n))        
    
    ani = animation.FuncAnimation(fh,update_img, np.arange(N),interval= 1000/fps, 
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


def calculateTimePtsFromTPlane(imgDir,fileExt = '.stack'):
    import sys, os,time
    sys.path.insert(0, 'C:/Users/pujalaa/Documents/Code/Python/code/codeFromNV')
    sys.path.insert(0, 'C:/Users/pujalaa/Documents/Code/Python/code/util')
    import tifffile as tf
    import numpy as np
    fileName = 'Plane01' + fileExt
    fp = os.path.join(imgDir,fileName)
    print('Reading plane data...')
    startTime = time.time()
    with open(fp,'rb') as file:
        A = file.read()
    print(int(time.time()-startTime),'sec')
    I = tf.TiffFile(os.path.join(imgDir,'ave.tif')).asarray()
    nPxlsInSlice = np.shape(I)[1]*np.shape(I)[2]
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
    import numpy as np
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.array([theta, rho])  

class cv(object):
    '''
    A set of routines used in computer vision. Most of the code is from the book
    "Programming Computer Vision with Python" by Jan Erik Solem (2012)    
    '''
    def pca(X):
        """
        Principal Components Analysis
        Parameters
        X: array-like
            Matrix of size M x N,  M = # of observations, and N = dimensionality
            of the training data.
        
        Returns
        V : projection matrix with important dimensions first
        S: variance
        mean_X : mean
        """
        import numpy as np
        import scipy as sp
        # Get dimensions
        nObs, dim = X.shape
        
        # Center data
        mean_X = X.mean(axis = 0)
        X = X - mean_X
        if dim > nObs:
            # PCA - compact trick used
            M = np.dot(X,X.T) # covariance matrix
            e,EV = np.linalg.eigh(M) # eigenvalues and eigenvectors
            tmp = np.dot(X.T,EV).T #  this is the compact trick
            V = tmp[::-1] # reverse, since last eigenvectors are the ones we want
            S = sp.sqrt(e)[::-1] # reverse, since eigenvalues are in increasing order
            V = V + 0j # This is a hack that AP implemented to prevent error with complex numbers
            for i in range(V.shape[1]):
                V[:,i] /= S
            V = sp.absolute(V)
            S = sp.absolute(S)
        else:
            # PCA - SVD used
            U,S,V = np.linalg.svd(X)
            V = V[:nObs] # only makes sense to return the first nObs vectors        
        
        # Return the projection matrix, the variance and the mean
        return V, S, mean_X
        
def denoiseImages(images, scheduler = 'threads',**kwargs):
    """
    Wrapper for applying denoise_wavelet from skimage.restoration to
    a stack of images using dask backend
    Parameters
    ----------
    images: array, ([T,] M, N)
        Images to denoise.
    scheduler: string; 'threads'(default)|'processes'
        Dask scheduler, see dask
    **kwargs: Keyword arguments to denoise_wavelet
    Returns
    -------
    images_den: array of shape images
        Denoised images
    """
    import numpy as np
    import dask
    from skimage.restoration import denoise_wavelet
    if np.ndim(images)==2:
        images = images[np.newaxis,...]
    method = kwargs.get('method', 'BayesShrink')
    mode = kwargs.get('mode','soft')
    images_den = dask.compute(*[dask.delayed(denoise_wavelet)(img, method = method, mode = mode)\
                                for img in images], scheduler = scheduler)
    return np.squeeze(np.array(images_den))

def getGlobalThr(img,thr0 = [],tol = 1/500,nIter_max= 100):
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
    
    def DiffThr(img,thr0):
        sub = img[img<thr0]
        supra = img[img>=thr0]
        thr = 0.5*(np.mean(sub) + np.mean(supra))
        return thr
        
    if (len(np.shape(thr0))==0) or (len(thr0)==0):
        thr0 = threshold_otsu(img)
    
    count = 0
    thr1 = DiffThr(img,thr0)
    dThr = np.abs(thr1-thr0)
    while (dThr > tol) & (count < nIter_max):
        thr = thr1
        thr1 = DiffThr(img,thr)
        dThr = np.abs(thr1-thr)
        count = count + 1
#        print('Iter #', str(count), 'dThr = ', str(dThr*100))
    return thr1
  
def getNumTimePoints(inDir):
    """ nTimePts = getNumTimePoints(inDir)
        Reads Stack_frequency.txt to return the number of temporal stacks
    """
    import os
    fp = os.path.join(inDir,'Stack_frequency.txt')
    with open(fp) as file:
        nTimePts = int(file.readlines()[2])
    return nTimePts
      
def getStackDims(inDir):
    """ Parse xml file to get dimension information of experiment.
    Returns [x,y,z] dimensions as a list of ints
    """
    import xml.etree.ElementTree as ET

    dims = ET.parse(inDir+'ch0.xml')
    import os
    fp = os.path.join(inDir,'ch0.xml')
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
    """Given rawPath, a path to .stack files, and frameNo, an int, load the .stack file
    for the timepoint given by frameNo from binary and return as a numpy array with dimensions=x,y,z"""

    import numpy as np
    from string import Template

    dims = getStackDims(rawPath)
    fName = Template('TM${x}_CM0_CHN00.stack')
    nDigits = 5

    tmpFName = fName.substitute(x=str(frameNo).zfill(nDigits))
    im = np.fromfile(rawPath + tmpFName,dtype='int16')
    im = im.reshape(dims[-1::-1])
    return im

class img(object):
    def convertBmp2Jpg(imgDir):
        '''
        convertBmp2Jpg - Given an image dir (path), converts all .bmp images within
            to .jpg images    
            '''
        import time, os
        #import multiprocessing
        from PIL import Image
        #from joblib import Parallel, delayed
        def bmp2Jpg(bmpPath):
            targetPath = bmpPath.split('.')[0] + '.jpg'
            #im = Image.open(bmpPath).convert('RGB')
            im = Image.open(bmpPath)
            im.save(targetPath,format = 'jpeg')
            im.close()
            try:
                os.remove(bmpPath)
            except:
                print('Unable to delete...', bmpPath)
        
        tic = time.time()
        print('Getting .bmps in dir...')
        bmpsInDir= img.getImgsInDir(imgDir,imgExts=['.bmp'])
        bmpPaths = [os.path.join(imgDir, bmp) for bmp in bmpsInDir]
        print('Converting .bmps to .jpgs...')
        jpgPaths = [bmp2Jpg(bmpPath) for bmpPath in bmpPaths]
        print(int((time.time()-tic)/60), 'mins')
        return jpgPaths

    def cropImgsAroundPoints(I,pts, cropSize = 100, keepSize = True, 
                             n_jobs = 32, verbose = 0, pad_type = 'zero'):
        '''
        Crops an image stack around fish position coordinates
        Parameters
        ----------
        I: 3D array, shape = (T,M,N)
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
        import numpy as np    
        if np.size(cropSize)==1:
            cropSize = np.array([cropSize, cropSize])
        def cropImgAroundPt(img,pt,cropSize,keepSize):
            pt = np.fliplr(np.array(pt).reshape(1,-1)).flatten().astype(int) # Change from x,y to row,col coordinates
            pre = int(cropSize[0]/2)
            post = cropSize[0]-pre
            r = np.arange(np.max([0,pt[0]-pre]), 
                          np.min([img.shape[0],pt[0]+post]))
            pre = int(cropSize[1]/2)
            post = cropSize[1]-pre
            c = np.arange(np.max([0,pt[1]-pre]),
                          np.min([img.shape[1],pt[1] + post]))
            img_crop = img[r.reshape(-1,1),c]
            if keepSize:
                d = np.c_[(cropSize)-np.array(img_crop.shape)]
                d = np.max(np.concatenate((np.zeros(d.shape),d),axis =1),axis = 1).astype(int)            
                pw = ((int(d[0]/2), d[0]-int(d[0]/2)),
                      (int(d[1]/2), d[1]-int(d[1]/2)))            
            if (pad_type == 'nan') | (pad_type == 'zero'):
                mode = 'constant'
                if pad_type == 'nan':
                    const_val = np.nan
                else:
                    const_val = 0
                img_crop =  np.pad(img_crop, pad_width = pw, mode = mode, constant_values = const_val)
            elif pad_type == 'edge':
                img_crop = np.pad(img_crop, pad_width = pw, mode = pad_type)
           
            return img_crop
        if np.ndim(I)==2:
            I = I[np.newaxis,...]
            pts = np.array(pts).reshape(1,-1)
        if (n_jobs > 1) & (n_jobs > I.shape[0]):
            from joblib import Parallel, delayed
            I_crop = Parallel(n_jobs = n_jobs, verbose= verbose)(delayed(cropImgAroundPt)(img,pt, cropSize, keepSize) 
            for img, pt in zip(I, pts))
        else:
            I_crop = [cropImgAroundPt(img, pt, cropSize, keepSize) for img, pt in zip(I, pts)]
        return np.squeeze(np.array(I_crop))    
    
    def findHighContrastPixels(I, zScoreThr = 1, method = 1):
        '''
        findHighContrastPixels - Finds high contrast pixels within an image I (mean of x and y gradients method)
        pxlInds = findHighContrastPixels(I,zScoreThr = 1)
        Inputs:
        I - Image in which to find pixels
        zScoreThr - Threshold in zscore for a pixel intensity to be considered high contrast
        Outputs:
        edgeInds - Indices of high contrast pixels (i.e., edges)
        I_grad - gradient image which is generated as an intermediate step in detecting edges
            method - Method 1 : I_grad = (np.abs(I_gradX) + np.abs(I_gradY)/2 (default, because faster)
                     Method 2: I_grad = np.sqrt(I_gradX**2 + I_gradY**2)
        '''
        import numpy as np
        if method ==1:
            I_grad = np.abs(np.gradient(I))
            I_grad = (I_grad[0] + I_grad[1])/2
        else:
            I_grad = np.gradient(I)        
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
            imgExts = ['.jpg','.jpeg','.tif','.tiff','.bmp','.png']
        if len(addImgExts) !=0:
            imgExts= list(np.union1d(imgExts,addImgExts))
        imgsInDir = []
        thingsInDir= os.listdir(imgDir)
        for ext in imgExts:
            blah = list(filter(lambda x: x.endswith(ext),thingsInDir))
            imgsInDir = list(np.union1d(imgsInDir,blah))
        return imgsInDir

    def gray2rgb(img,cmap = 'gray'):
        """ Given a grayscale image, returns rgb equivalent
        Inputs:
        cmap - Colormap; can be specified as 'gray', 'jet', 
            or as plt.cm.Accent, etc
        """
        import matplotlib.pyplot as plt
        import numpy as np
    
        def standardize(x):
            x = (x-x.min())/(x.max()-x.min())
            
        if np.abs(img).max()>1:
            img = standardize(img)    
        cmap = plt.get_cmap(cmap)
        img_rgb = np.delete(cmap(img),3,2)
        return img_rgb
    
    def histeq(img,nBins = 256):
        '''
        Equalize the histogram of a grayscale image (similiar to MATLAB 'histeq')
        Parameters
        img : array-like
            Image array whose histogram is to be equalized
        nBins : scalar
            Number of bins to use for the histogram
        
        Returns
        img_eq : array-like
            Histogram-equalized image
        '''     
        import numpy as np
        
        # Get image histogram
        imhist,bins = np.histogram(img.flatten(), nBins, normed = True)
        cdf = imhist.cumsum() # cumulative distribution function
        cdf = 255*cdf/cdf[-1] # normalize
        
        # User linear interpolation of cdf to find new pixel values
        img2 = np.interp(img.flatten(),bins[:-1],cdf)
        
        return img2.reshape(img.shape), cdf
        
    
    
    def filtImgs(I,filtSize = 5, kernel = 'median', 
                 process = 'parallel', verbose = 1):
        '''
        Processes images so as to make moving particle tracking easier
        I_proc = processImagesForTracking(I,filtSize = 5)
        Parameters
        ----------
        I: 3D array of shape (T,M,N), where T = # of images, M, N = # of rows
            and columns respectively
            Image stack to filter
        kernel: String or 2D array
            ['median'] | 'rect' | 'gauss' or array specifying the kernel
        filtSize: Scalar or 2-tuple
            Size of the kernel to generate if kernel is string
        '''
        from scipy import signal
        import numpy as np
        import apCode.SignalProcessingTools as spt
        #import time
        
        if process.lower() == 'parallel':
            from joblib import Parallel, delayed
            import multiprocessing
            parFlag = True
            num_cores = np.min((multiprocessing.cpu_count(),32))
        else:
            parFlag = False        
        #tic = time.time()
        if np.ndim(I)<3:
            I = I[np.newaxis,:,:]
        N = np.shape(I)[0]       
        
        I_flt = np.zeros(np.shape(I))
        if isinstance(kernel, str):
            if kernel.lower()=='median':
                #print('Median filtering...')
                if np.size(filtSize)>1:
                    filtSize = filtSize[0]
                if np.mod(filtSize,2)==0:
                    filtSize = filtSize+1 # For median, the filter size should be odd
                    print('Median filter size must be odd, changed to {}'.format(filtSize))
                if parFlag:
                    print('# of cores = {}'.format(num_cores))
                    I_flt = Parallel(n_jobs = num_cores,verbose = verbose)(delayed(signal.medfilt2d)(img,filtSize) for img in I)
                    I_flt = np.array(I_flt)
                else:                    
                    for imgNum, img in enumerate(I):
                        if np.mod(imgNum,300)==0:
                            print('Img # {0}/{1}'.format(imgNum,N))
                        I_flt[imgNum,:,:] = signal.medfilt2d(img,filtSize)                   
            elif kernel.lower() == 'rect':
                #print('Rectangular filtering...')
                if np.size(filtSize)==1:
                    ker = np.ones((filtSize,filtSize))
                else:
                    ker = np.ones(filtSize)
                ker = ker/ker.sum()
                ker = ker[np.newaxis,:,:]
                if parFlag:
                    I_flt = Parallel(n_jobs = num_cores,verbose = verbose)(delayed(signal.convolve)(img,ker, mode = 'same') for img in I)
                else:
                    I_flt = signal.convolve(I,ker,mode = 'same')
            elif kernel.lower()=='gauss':
                #print('Gaussian filtering...')
                if np.size(filtSize)==1:
                    ker = spt.gausswin(filtSize)
                    ker = ker.reshape((-1,1))
                    ker = ker*ker.T
                else:
                    ker1 = spt.gausswin(filtSize[0]).reshape((-1,1))
                    ker2 = spt.gausswin(filtSize[0]).reshape((-1,1))
                    ker = ker1*ker2.T
                ker = ker/ker.sum()                
                if parFlag:
                    I_flt = Parallel(n_jobs = num_cores,verbose = verbose)(delayed(signal.convolve)(img,ker, mode = 'same') for img in I)
                    I_flt = np.array(I_flt)
                else:
                    ker= ker[np.newaxis,:,:]
                    I_flt = signal.convolve(I,ker,mode = 'same')
        else:
            ker = ker/ker.sum()                
            if parFlag:
                I_flt = Parallel(n_jobs = num_cores,verbose = 5)(delayed(signal.convolve)(img,ker) for img in I)
            else:
                ker= ker[np.newaxis,:,:]
                I_flt = signal.convolve(I,ker,mode = 'same')
       
        #print(int(time.time()-tic), 'sec')
        if np.shape(I_flt)[0]==1:
            I_flt = np.squeeze(I_flt)
        
        return I_flt  
    
    def gaussFilt(I, sigma = 3, mode= 'nearest', n_jobs = 32, verbose = 0, preserve_range = True):
        """
        Applies skimage.filters.gaussian to an image stack using parallel loop for speed.
        """
        import numpy as np
        from joblib import Parallel, delayed
        from skimage.filters import gaussian
        if np.ndim(I)==2:
            I = I[np.newaxis,...]
        if n_jobs >1:
            I_flt = np.array(Parallel(n_jobs = n_jobs, verbose = verbose)(delayed(gaussian)(img, sigma = sigma, mode = mode, preserve_range = True) for img in I))
        else:
            I_flt = np.array([gaussian(img, sigma = sigma, mode = mode, preserve_range = True) for img in I])
        return I_flt

    def otsu(img, binary = False, mult = 1):
        """
        Returns either a binary or grayscale image after thresholding with otsu's 
        threshold
        Parameters
        ----------
        img: array-like
            Image or other array to threshold
        binary: boolean
            If True, then return binary image with values above otsu's threshold set to 1,
            and all other values set to zero
        mult: scalar
            Value by which otsu's threshold is multiplied by before thresholding
        
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
            r,c = np.where(img_new < (mult*threshold_otsu(img)))
            img_new[r,c] = 0
            return img_new
        
    def palplot(cMap,nPts = None):
        """
        My version of seaborn style palplot, except takes a single or list of
        colormap objects (e.g., plt.cm.jet) or colormap matrices as input and
        plots specified # of points from them    
        """
        import numpy as np
        import matplotlib.pyplot as plt
    
        if nPts is None:
            nPts = 255
        if (isinstance(cMap, list)) & (len(cMap)>1):
            pass
        else:
            cMap = [cMap]
        clrsList = []
        for cmNum, cm in enumerate(cMap):
            if isinstance(cm,str):
                cm = plt.cm.get_cmap(cm)
                x = np.linspace(0,255,nPts).astype(int)
                clrs = [cm(item) for item in x]           
            elif isinstance(cm, list):        
                x = np.linspace(0,len(cm)-1,nPts).astype(int)
                clrs = [cm[item] for item in x]          
            elif isinstance(cm,np.ndarray):
                x = np.linspace(0,np.shape(cm)[0]-1,nPts).astype(int)
                clrs = [cm[item,:] for item in x]  
            else:
                x = np.linspace(0,255,nPts)
                clrs = [cm(int(item)) for item in x]             
            clrsList.append(clrs)
            plt.style.use(('seaborn-dark','dark_background'))
            if np.shape(clrs)[1]==4:           
                plt.scatter(x,np.ones(np.shape(x))-cmNum, color = clrs, s = 1000, 
                            marker = 's');
            else:
                plt.scatter(x,np.ones(np.shape(x))-cmNum, color = clrs, s = 1000, 
                            marker = 's');           
        plt.axis('off')
        return clrsList
    
    
    def readImagesInDir(imgDir,ext = 'bmp', imgNames = None, process = 'parallel', numCores = 32,
                        verbose = 0):
        '''
        Reads 2D images in a dir with specified ext and returns as a numpy array
        
        Parameters
        imgDir : string
            Path to the directory containing the images
        ext : string
            Extension of the images to read, e.g., 'bmp', 'tiff', etc
        process : string
            'serial' or 'parallel'; determines whether to read the images in serial
            or in parallel
        numCores : scalar
            If process == 'parallel' then this specifies the # of cores to use
        '''
        import os
#        from matplotlib import image
        from skimage.io import imread
        import numpy as np       
        import apCode.FileTools as ft
        from joblib import Parallel, delayed
        import multiprocessing as mp
        
        if np.any(imgNames == None):
            imgsInDir = ft.findAndSortFilesInDir(imgDir,ext = ext)
        else:
            imgsInDir = imgNames
            
        dispChunk = int(len(imgsInDir)/5)
        I=[]
        count = 0 
        #print('Reading images...')
        if process.lower() == 'serial':
            for img in imgsInDir:
                I.append(imread(os.path.join(imgDir,img)))
                count = count + 1
                if np.mod(count,dispChunk)==0:
                    print(str(count) + '/' + str(len(imgsInDir)), '\t')
        else:
            numCores = np.min((mp.cpu_count(),numCores))
            I = Parallel(n_jobs=numCores, verbose=verbose)(delayed(imread)
            (os.path.join(imgDir,img)) for img in imgsInDir) 
        I = np.array(I)
        #print('Done')
        return I

    def resize(I, sz, n_jobs = 32, verbose = 0,preserve_dtype = True,**kwargs):
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
        if np.ndim(I)==2:
            I_rs = resize(I,sz,**kwargs)    
        else:
            if n_jobs >1:
                from joblib import Parallel, delayed
                I_rs = Parallel(n_jobs= n_jobs, verbose = verbose)(delayed(resize)(im,sz,**kwargs) for im in I)
            else:
                I_rs =[resize(im,sz, **kwargs) for im in I]    
        I_rs = np.array(I_rs)
        if preserve_dtype:
            I_rs = I_rs.astype(I.dtype)
        return I_rs   
    
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
    
    def rgba2rgb(I):
        """"
        Given an rgba image stack, where a is the alpha channel, returns an rgb
            image scaled by the alphas in the alpha channel. This can be useful
            for opening using ImageJ
        Parameters:
        I - array-like of shape(z,m,n,4), where z is the number of image slices,
            m, n are the rows and columns of each image, and 4 corresponds to the
            r,g,b, and a (alpha) channels respectively.        
        Returns:
        I - array-like, shape = (z, m, n, 3)
        """
        import numpy as np
        import sys        
        if isinstance(I, list):
            I = np.array(I)
        if np.shape(I)[-1] <4:
            print('Image must have 3 color and 1 alpha channel')
            sys.exit()
        elif np.ndim(I) ==3:
            I = I[np.newaxis,:,:,:]
        elif np.ndim(I)<3:
            print('RGB image must have at least 3 dimensions')
            sys.exit()
        alphaMask = I[:,:,:,-1]
        alphaMask = np.tile(alphaMask[:,:,:,np.newaxis],[1,1,1,3])
        I_rgb = I[:,:,:,:3]*alphaMask
        return I_rgb    
               
    def rotate(I, angle, n_jobs = 32, verbose =0,**kwargs):
        """
        Wrapper for applying skimage.transform.rotate to an image stack
        """
        import numpy as np
        from skimage.transform import rotate
        from joblib import Parallel, delayed
        if np.ndim(I) ==2:
            I = I[np.newaxis,...]
        preserve_range = kwargs.get('preserve_range', True)
        I_rot = np.array([rotate(img,angle,preserve_range = preserve_range,**kwargs) for img in I])
#        I_rot = np.array(Parallel(n_jobs = n_jobs, verbose = 0)(delayed(rotate)(img,angle) for img in I))
        return I_rot
    
    def saveImages(I,imgDir = [], imgNames =[], fmt = 'bmp', cmap = 'gray', 
                   dispChunk = None, n_jobs = 20, verbose = 0):
        """
        Saves an image stack
        
        Parameters
        ----------
        I: (m,n,k), array-like
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
        
        if np.ndim(I)==2:
            I = I[np.newaxis,...]
        
        if len(imgDir)==0:
            imgDir = os.getcwd()
        elif os.path.exists(imgDir) == False:
            os.mkdir(imgDir)
        if len(imgNames)==0:
            imgNames = ('Img' + '%.6d' % num + '.' + fmt for num in range(np.shape(I)[0]))
        if dispChunk == None:
            dispChunk = np.inf
        else:
            dispChunk = int((np.shape(I)[0])/30)
        
        if (I.shape[0]<n_jobs):
            for imgNum,imgName in enumerate(imgNames):
                filePath = os.path.join(imgDir,imgName)    
                imsave(filePath,I[imgNum])
#                if np.mod(imgNum,dispChunk)==0:
#                    print(imgNum)
        else:
            from joblib import Parallel, delayed            
            Parallel(n_jobs = n_jobs, verbose = verbose)(delayed(imsave)(os.path.join(imgDir,imgName),img) for imgName,img in zip(imgNames,I))
                    
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

class morphology():
    __codeDir = r'V:\Code\Python\code'
    import sys as __sys
    __sys.path.append(__codeDir)        
    import apCode.geom as __geom        
    endpoints_curve_2d = __geom.endpoints_curve_2d
    def neighborhood(img,coords, n = 1):
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
        if np.size(coords)==2:
            coords = [coords]
        P,N = [],[]
        for r,c in coords:        
            inds = np.array([np.arange(r-n,r+n+1),np.arange(c-n,c+n+1)])
            rowInds = np.where((inds[0] >=0) & (inds[0] < img.shape[0]))[0]
            colInds = np.where((inds[1]>=0) & (inds[1] < img.shape[1]))[0]
            keepInds = np.intersect1d(rowInds,colInds)
            inds = inds[:,keepInds]
            rr,cc = np.meshgrid(inds[0],inds[1])
            p = img[rr.astype(int),cc.astype(int)]
            N.append(inds)
            P.append(p)
        return P,N
    
    def thin_weighted(img, nhood = 4, points = None):
        """
        Returns the x,y coordinates of the thinned line resulting from
        skimage.morphology.thin, but after weighting with pixel intensity
        Parameters
        ---------
        img: 2D array (M,N)
            Imge to thin
        nhood: scalar
            Neighborhood of pixel weighting
        points: array, (K,2)
            Points to be weighted. If None or [], them uses thinning obtain this 
            information. However, it weighting needs to be done iteratively,
            then it helps to have these points
    
        Returns
        -------
        xy: array, (2,N)
            xy[0], and xy[1] are the x and y coordinates respectively of the 
            thinned lines in image
            """
    
        from skimage.morphology import thin
        import numpy as np
        def row_wts(img,r,c,nhood):
            nRows = np.shape(img)[0]
            x = np.arange(r-nhood, r+nhood+1)
            y = np.ones((len(x),))*c
            inds = np.array((x,y)).astype(int)
            #inds = np.array([np.arange(r-nhood,r+nhood+1),[c]*(2*nhood+1)])
            #inds = inds.astype(int)
            keepInds = np.where((inds[0]>=0) & (inds[0]<nRows))[0]
            inds = inds[:,keepInds]
            wts = img[inds[0],inds[1]]
            wts = wts/np.sum(wts)
            r_wt = np.dot(x[keepInds],wts)
            return r_wt
        def col_wts(img,r,c,nhood):
            nCols = np.shape(img)[1]
            y = np.arange(c-nhood, c+nhood+1)
            x = np.ones((len(y),))*r
            inds = np.array((x,y)).astype(int)
            #inds = np.array([[r]*(2*nhood+1),np.arange(c-nhood,c+nhood+1)])
            #inds = inds.astype(int)
            keepInds = np.where((inds[1]>=0) & (inds[1]<nCols))[0]
            inds = inds[:,keepInds]
            wts = img[inds[0],inds[1]]   
            wts = wts/np.sum(wts)
            c_wt = np.dot(y[keepInds],wts)
            return c_wt
        if np.size(points)<2:
            R,C = np.where(thin(img))
        else:
            R, C = points[:,1], points[:,0]
        xy = []
        for r,c in zip(R,C):            
            r_wt = row_wts(img,r,c,nhood)
            c_wt = col_wts(img,r,c,nhood)
            xy.append([c_wt,r_wt])
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

def projFigure(vol, limits, plDims=[16,10,5], zscale=5, colors='gray', title=None):
     
    """Display vol.max(dim) - vol.min(dim) for dims in [0,1,2]
    Heavily adapted from Jason Wittenbach's crossSectionPlot. 
    """
    import matplotlib.pyplot as plt

    x, y, z = plDims
    grid = (y+z, x+z)
    zRat = zscale*(float(y)/z)
    plt.figure(figsize=grid[-1::-1])   
    
    # plot the x-y view (top-down)
    ax1 = plt.subplot2grid(grid, (0, 0), rowspan=y, colspan=x)
    plt.imshow(vol.max(0) + vol.min(0), clim=limits, cmap=colors, origin='leftcorner', interpolation='Nearest')
    ax1.axes.xaxis.set_ticklabels([])
    
    if title:
        plt.title(title)
    
    # plot the x-z view (side-on)
    ax2 = plt.subplot2grid(grid, (y, 0),rowspan=z, colspan=x)
    plt.imshow(vol.max(1)+vol.min(1), aspect=zRat, clim=limits, cmap=colors, origin='leftcorner', interpolation='Nearest')
    
    # plot the y-z view (head-on)
    ax3 = plt.subplot2grid(grid, (0, x), rowspan=y, colspan=z)
    plt.imshow((vol.max(2)+vol.min(2)).T, aspect=1/zRat, clim=limits, cmap=colors, origin='leftcorner', interpolation='Nearest')

    ax3.axes.yaxis.set_ticklabels([])

      
def tToZStacks(inDir):
    '''
    tToZStacks(inDir):
    Writes the tStacks in 'inDir' to zStacks in 'inDir\\zStacks'
    '''
    import sys, os
    sys.path.insert(0, 'C:/Users/pujalaa/Documents/Code/Python/code/codeFromNV')
    sys.path.insert(0, 'C:/Users/pujalaa/Documents/Code/Python/code/util')
    import time
    import numpy as np
    import volTools
    
    imgDir = inDir
   # allFilesInDir = os.listdir(imgDir)
    fileExt = '.stack'
    fileStem = 'Plane'

    #tStacks = np.sort(list(filter(lambda x: x.startswith(fileStem),\
    # allFilesInDir)))

    stackDims = volTools.getStackDims(imgDir)
    nTimePts = volTools.getNumTimePoints(imgDir)

    outDir = os.path.join(imgDir,'zStacks')
    if os.path.isdir(outDir):
        print(outDir, '\n ...already exists!')    
    else:
        os.mkdir(outDir)

    print('Writing files...')
    startTime = time.time()
    for stackNum in range(stackDims[2]):
        print('Reading plane ' + str(stackNum+1) + '...')
        fn = fileStem + '%0.2d' % (stackNum+1) + fileExt
        fp = os.path.join(imgDir,fn)
        blah = np.fromfile(fp,dtype = 'uint16')
        blah = np.reshape(blah,(nTimePts,stackDims[1], stackDims[0]))
        print('Writing plane ' + str(stackNum+1) + '...')
        for timePt in range(nTimePts):        
            fn = 'TM' + '%0.05d' % timePt + '.bin'
            fp = os.path.join(outDir,fn)        
            file = open(fp,'ab')
            file.write(blah[timePt])
            file.close()
    print(int(time.time()-startTime)/60,'min')

def pol2cart(th,rho):
    import numpy as np
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
    #x,y = np.round(x*1000)/1000, np.round(y*1000)/1000
    return np.array([x, y])

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
    dLine - Radial distance between points in the line     '''
   
    import numpy as np
    lines = []
    xInds = []
    yInds = []
    thetas = np.arange(0,360,dTheta)
    lineLengths = np.arange(1,lineLength+1,dLine)
    for theta in thetas:
        inds = list(map(lambda x: pol2cart(x,theta), lineLengths))
        xInds = np.array(list(ind[0] for ind in inds)) + pt[0]
        yInds = np.array(list(ind[1] for ind in inds)) + pt[1]
        line = np.array([xInds, yInds])
        lines.append(line)
    return np.array(lines)          

class Register(object):
    """ 
    Registers an image or image stack to a reference object
    Parameters
    ----------
    n_jobs: integer
        Number of parallel jobs (if backend == 'joblib')
    verbose: integer
        See Parallel, delayed from joblib
    upsample_factor: scalar
        Factor by which to upsample images before registration. This is useful either for subpixel 
        registration (upsample_factor > 1) or for speeding up registration (upsample_factor <1).
    backend: string, 'joblib' (default) or 'dask'
        Backend to use in running registration. If 'joblib' uses Parallel, delayed with 'loky' 
        backend. If 'dask' then uses multithreading and dask
    scheduler: string, 'threads' or 'processes' (default)
        IFF backend == 'dask', this specifies the type of scheduler to use. See dask.compute
    regMethod: string, 'st' (standard translation), 'cr' (caiman rigid), or 
        'cpwr' (caiman piecewise rigid)
        Specifies the registration method. For more info on 'cr' or 'cpwr' see 
        caiman.motion_correction.MotionCorrect(). 'st' is implemented using 
        skimage.feature.register_translation.
    """
    def __init__(self, n_jobs =32, verbose = 0, upsample_factor = 1, backend = 'joblib',\
                 scheduler = 'processes', regMethod = 'st'):
        self.n_jobs_ = n_jobs
        self._verbose = verbose
        self.upsample_factor_ = upsample_factor
        self.backend_ = backend
        self.scheduler_ = scheduler
        self.regMethod_ = regMethod
        
    def fit(self, I, ref = None):
        """ 
        Registers images against reference and returns translation coordinates
        from registration
        Parameters
        ----------
        I: array, ([T,], M, N)
            Image stack to register
        ref: array of shape (M, N) or None
            Reference image to register against. If None, then uses mean of image stack along
            first dimension (i.e. time).
        Returns
        -------
        regObj: object
            Registration object with registration shifts stored in regObj.translation_coords_.
            The method regObj.transform() will apply registration parameters computed from I
            to other image stacks
        """
        import numpy as np
        from skimage.feature import register_translation
        if np.ndim(I)<3:
            I = I[np.newaxis,:,:]
        if np.any(ref == None):
            ref = I.mean(axis = 0)
        if self.regMethod_ == 'st':
            if self.n_jobs_ <=1:
                shifts = np.array([register_translation(ref,img, upsample_factor = self.upsample_factor_)[0] for img in I])            
            else:
                if self.backend_ == 'joblib':
                    from joblib import Parallel, delayed
                    from multiprocessing import cpu_count
                    self.n_jobs_ = np.min((self.n_jobs_, cpu_count()))
                    shifts = Parallel(n_jobs = self.n_jobs_, verbose = self._verbose)\
                    (delayed(register_translation)(ref, img) for img in I)
                    shifts = np.array([shift[0] for shift in shifts])                
                elif self.backend_ == 'dask':
                    import dask
                    shifts_lazy = [dask.delayed(register_translation)\
                                   (ref,img, upsample_factor = self.upsample_factor_) for img in I]
                    foo = dask.compute(shifts_lazy, scheduler = self.scheduler_)[0]
                    shifts = np.array([_[0] for _ in foo])                
                else:
                    print('Please specify valid backend ("joblib" or "dask")')
                    shifts = None
        elif (self.regMethod_ == 'cr') | (self.regMethod_ == 'cpwr'):
            import caiman as cm
            from caiman.motion_correction import MotionCorrect
            cm.stop_server()
            if 'dview' in locals():
                cm.stop_server(dview = dview)
            c, dview, n_processes = cm.cluster.setup_cluster(backend='local',\
                                                             n_processes=None, single_thread=False,\
                                                             ignore_preexisting=True)
            imgDims = np.array(I.shape[-2:])
            strides = np.ceil(imgDims/3).astype(int)
            overlaps = np.ceil(strides/1.5).astype(int)
            max_shifts = np.ceil(imgDims/10).astype(int)
            max_deviation_rigid = max_shifts[0]
            shifts_opencv = True
            border_nan = 'copy'
            if self.regMethod_ == 'cpwr':
                pw_rigid = True
            else:
                pw_rigid = False
            mc = MotionCorrect(I, dview = dview, max_shifts= max_shifts,\
                               strides=strides, overlaps= overlaps,\
                               max_deviation_rigid= max_deviation_rigid,\
                               nonneg_movie = True, border_nan=border_nan,\
                               shifts_opencv=shifts_opencv, pw_rigid= pw_rigid)                            
            mc.motion_correct(save_movie=False, template=ref)
            attrs = mc.__dict__.keys()
            if 'x_shifts_els' in attrs:
                shifts = np.array([mc.x_shifts_els, mc.y_shifts_els]).transpose(1,0,2)
            else:
                shifts = np.array(mc.shifts_rig)
            self.apply_shifts_movie = mc.apply_shifts_movie
            registration_constraints = dict(strides = strides,overlaps = overlaps,\
                                            max_shifts = max_shifts,\
                                            max_deviation_rigid = max_deviation_rigid)
            self.registration_constraints_ = registration_constraints
            cm.stop_server(dview = dview)
        self.translation_coords_ = shifts        
        return self
        
    def transform(self,I):
        """ 
        Register a stack of images using translation coordinates
        computed using the fit method 
        Parameters
        ----------
        I: array, ([T,], M, N)
            Image to stack to which to apply registration parameters computed from the same or
            a different image stack of similar dimensions (such as a different channel, for instance)
        Returns
        -------
        I_shifted: array of shape(I)
            Registered image stack.
        """
        import numpy as np
        from scipy.ndimage import fourier_shift
        from numpy.fft import fftn, ifftn
        import caiman as cm
        shifts = self.translation_coords_
        f = lambda img,s: np.real(ifftn(fourier_shift(fftn(img),s)))
        if np.ndim(I)<3:
            I = I[np.newaxis, :, :]
            shifts = np.array(shifts).reshape((1,-1))
        if (self.regMethod_ == 'cr') | (self.regMethod_ == 'cpwr'):            
            if 'dview' in locals():
                cm.stop_server(dview = dview)
            c, dview, n_processes = cm.cluster.setup_cluster(backend = 'local',\
                                                             n_processes=None, single_thread=False,\
                                                             ignore_preexisting=True)
            I_shifted = self.apply_shifts_movie(I)
            cm.stop_server(dview = dview)
        else:
            if self.n_jobs_ <=1:
                I_shifted = [f(img,s) for img, s in zip(I, shifts)]
            else:            
                if self.backend_ == 'joblib':
                    from joblib import Parallel, delayed
                    from multiprocessing import cpu_count
                    self.n_jobs_ = np.min((self.n_jobs_, cpu_count()))                
                    I_shifted = Parallel(n_jobs = self.n_jobs_, verbose = self._verbose)(delayed(f)(img, s) for img, s in zip(I, shifts))
                elif self.backend_ == 'dask':
                    import dask
                    I_lazy = [dask.delayed(f)(img, s) for img, s in zip(I, shifts)]
                    I_shifted = dask.compute(I_lazy, scheduler = self.scheduler_, n_workers = self.n_jobs_)[0]
                else:
                    print('Please specify valid backend ("joblib" or "dask")')
                    I_shifted = None                    
        return np.squeeze(np.array(I_shifted))

def threshold_multi(img, n  = 3):
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
    thr= np.zeros((n,))
    img_thr = img.ravel()
    img_quant = (img*0).astype(int)
    thr_up = np.infty
    for n_ in range(n):
        thr[n_] = otsu(img_thr)       
        aboveInds = np.where((img>thr[n_]) & (img <= thr_up))
        thr_up = thr[n_]      
        img_quant[aboveInds] = n-n_
        img_thr = np.delete(img_thr,np.where(img_thr>thr[n_]))
    return thr, img_quant