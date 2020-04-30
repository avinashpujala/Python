
#from apCode.FileTools import sublistsFromList
import numpy as np

class BootstrapStat(object):
    def __init__(self, func='mean', combSize=2, nCombs=None, replace=False):
        if isinstance(func, str):
            func = eval(f'np.{func}')
        self.func = func
        self.combSize=combSize
        self.nCombs=nCombs
        self.replace=replace

    def fit(self, items):
        nItems = len(items)
        if self.nCombs is None:
            self.nCombs = self.numCombinations(nItems, self.combSize)
        inds = np.arange(nItems)
        combInds=[]
        count=0
        while count < self.nCombs:
            inds_ = np.random.choice(inds, size=self.combSize,
                                     replace=self.replace)
            combInds.append(inds_)
            count += 1
        self.combInds = np.array(combInds)
        return self

    def fit_transform(self, items):
        combs, items_comb = self.fit(items).transform(items)
        return combs, items_comb

    def transform(self, items):
        if not hasattr(self, 'combInds'):
            print('Must be fit first')
            return None
        items_comb, combs = [], []
        for inds in self.combInds:
            items_ = items[inds]
            comb_ = self.func(items_, axis=0)
            items_comb.append(items_)
            combs.append(comb_)
        return np.array(combs), np.array(items_comb)


    @staticmethod
    def numCombinations(nItems, combSize):
        num = np.math.factorial(nItems)
        den = np.math.factorial(combSize)*np.math.factorial(nItems-combSize)
        return num/den


class CombineItems(object):
    """
    Choose unique n-combinations from a set of items,
    applies the specified function and returns both the combinations
    and the results of the applied function.
    Parameters
    ----------
    items: array-like, (N,[,T,M,...])
        Collection of items, where N is the total number of
        items of any dimension
    func: string, function object, or None
        Function to apply to the n-combinations. For, example
        if func = 'mean', then will yield mean (along axis = 0) for
        each n-combination.
        If func is None, then will return combinations without applying any
        function to them. For instance, if func = np.prod, and a 3-combination
        yields (a, b, c) then the applied function will yield a*b*c
        Note: func should accept axis parameter.
    n: integer
        Number of items to combine per combination. For eg.,
        if items = np.arange(4), and n = 2, then the possible
        combinations are (0,1),(0,2),(0,3),(1,2),(1,3),(2,2),(2,3)
    N: integer or none
        This many combinations will randomly be chosen and returned.
        If None, then all combinations returned
    """
    def __init__(self, func='mean', n=2, N=None, replace=False):
        import numpy as np
        if isinstance(func, str):
            if func.lower() == 'mean':
               self.func = np.mean
            else:
                print('Only "add" is an acceptable string, please input a function object!')
                self.func = None
        else:
            self.func = func
        self.n = n
        self.N = N
    def fit(self,items):
        import numpy as np
        from itertools import combinations
        nItems = len(items)
        combs = np.array(list(combinations(np.arange(nItems), self.n)))
        combs = combinations(np.arange(nItems), self.n)
        nCombs_max = self.howManyCombinations(nItems, self.n)
        if self.N == None:
            self.N = combs.shape[0]
        if self.N > len(combs):
            print(f'Only {len(combs)} combinations possible!')
            self.N = nCombs_max
        randInds = np.random.choice(np.arange(nCombs_max), size=self.N,
                                    replace=replace)
        combs = np.arange(list(combs))[randInds]
        self.combs = combs
        return self

    def transform(self, items):
        import numpy as np
        if self.func == None:
            print('Function is None, returning with only combinations')
            C = []
            for c in self.combs:
                items_sub = items[c]
                C.append(items_sub,)
            C = np.array(C)
            return C
        else:
            C = []
            for c in self.combs:
                items_sub = items[c]
                C.append(self.func(items_sub,axis = 0))
            C = np.array(C)
            return C

    @staticmethod
    def howManyCombinations(nItems, n=2):
        import numpy as np
        num = int(np.math.factorial(nItems))
        den = np.math.factorial(n)*np.math.factorial(nItems-n)
        return num/den

def findStrInList(s,L, case_sensitive = False):
    """
    Find a string in a list of strings
    Parameters
    ----------
    s: string
        String to search for.
    L: list
        List of strings within which to search
    case_sensitive: bool
        If True, then case-sensitive searching.
    Returns
    -------
    inds: array, (n,)
        Indices of items in list (L) where the string (s) is found.
    """
    import re
    import numpy as np
    inds = []
    for count, l in enumerate(L):
        if case_sensitive:
            matchLen = re.findall(s,l)
#            m = re.match(s,l)
        else:
            matchLen = re.findall(s.lower(), l.lower())
#            m = re.match(s.lower(), l.lower())
        if len(matchLen)>0:
            inds.append(count)
    return np.array(inds)

def get_overlapping_blocks(x, blockSize, stride = 1):
    """
    Given an iterable, returns a list with sublists containing items
    from the iterable with specified (stride) amount of overlap between
    successive sublists.
    Parameters
    ----------
    x: iterable
        Iterable to get blocks from.
    blockSize: int
        Size of each block.
    stride: int
        Amount (# of items) by which to shift in collecting overlapping blocks.
    Returns
    -------
    blocks: list or array
        List/array of blocks with some overlap
    """
    import numpy as np
    blocks = []
    for i in np.arange(0,len(x)-blockSize,stride):
        blocks.append(x[i:i+blockSize])
    blocks.append(x[-blockSize:])
    try:
        blocks = np.array(blocks)
    except:
        pass
    return blocks

def get_blocks_of_repeats(x):
    """
    Given an iterable returns sublists of blocks of repeating values in the iterable
    Parameters
    ----------
    x: iterable
        Iterable (such as array or list) with possible blocks of repeating values.
    Returns
    -------
    sup: list
        Each sublist within this list holds blocks of repeating values in the
        original iterable.
    inds_sup: list
        Each sublist within this list the indices from the original iterable
        of blocks of repeating.
    """
    sup = []
    prev = x[0]
    sub = []
    inds_sup = []
    inds_sub = []
    for count, _ in enumerate(x):
        if _ == prev:
            sub.extend([_])
            inds_sub.extend([count])
            prev = _
        else:
            sup.append(sub)
            inds_sup.append(inds_sub)
            sub = [_]
            inds_sub = [count]
            prev = _
    sup.append(sub)
    inds_sup.append(inds_sub)
    return sup, inds_sup


def getContiguousBlocks(iterable):
    """
    Given an iterable (e.g., range(10)), returns a list of
    sublists, where each sublist is a contiguous block from the
    input iterable
    """
    import more_itertools as mit
    blocks = [list(group) for group in mit.consecutive_groups(iterable)]
    return blocks

def is_picklable(obj):
    """
    Checks if an object is pickleable. Can be function, array, etc.
    Parameters
    ----------
    obj: python object
        Object to check for pickleability.
    Returns
    -------
    ans: bool
        True or False
    """
    import pickle
    try:
        pickle.dumps(obj)
    except pickle.PicklingError:
        return False
    return True

def locateItemsInSetsOfItems(sub, sup):
    """
    Given an array of items spread across several lists of items, returns
    a dictionary holding the indices of lists containing the items as well as
    the indices of the items within each of the lists
    Parameters
    ----------
    sub: array-like
        Array of items to find in sup
    sup: list
        List of arrays/sublists of items in which to look
        for the items in sup
    Returns
    -------
    d: dictionary with keys, "supInds", "subInds"
        Dictionary wherein the key 'supInds' contains
        indices of list elements holding input indices,
        and 'subInds' contains indices within each list
        element where input indices are present
    """
    import numpy as np
    d = dict(supInds = [], subInds = [])
    for iSup, s in enumerate(sup):
        inBool = np.in1d(s,sub)
        if np.any(inBool):
            inInds = np.where(inBool)[0]
            d['supInds'].append(iSup)
            d['subInds'].append(inInds)
    return d

def sequenceMatch(seq,seqList, case_sensitive = True):
    """ Given a sequence (say, string), returns an array of values
    indicating the degree of match between seq and each sequence in a list.
    Parameters
    ----------
    seq: sequence (str) to match
    seqList: list of sequences, (n,)
    Returns
    -------
    m: array, (n,)
        Degree of match of seq to each item in seqList
    """
    from difflib import SequenceMatcher
    import numpy as np
    if case_sensitive:
        return np.array([SequenceMatcher(None,seq,s).ratio() for s in seqList])
    else:
        return np.array([SequenceMatcher(None,seq.lower(),s.lower()).ratio() for s in seqList])

class plot(object):
    """
    Set of plotting tools
    """
    def rose(thetas,bins = 24,bottom = 0,radians = False, normed = False,
             ylim ='auto', **kwargs):
        """
        Generates a rose plot (circular histogram) from a set angular values
        Parameters:
        thetas - Array-like; set of angle values to generate rose plot for.
        bins - Scalar, sequence, or string;
            If scalar, specifies number of bins. If, sequence, specifies the bin
            edges. If string, then specifies method for estimating bins
            (see numpy.histogram).
            (when scalar).
        radians - Boolean; If True then assumes angles are given in  and
            not radians
        bottom - Scalar; Determines the location of the bottoms of the bars
        ylim - 2-element array or str. If ylim = 'auto', then automatically
            determines
        **kwargs - Key, value pairs for plots
        """
        import numpy as np
        import matplotlib.pyplot as plt


        thetas = np.delete(thetas,np.where(np.isnan(thetas)))
        if not radians:
            thetas = thetas*np.pi/180

        #--- Histogram of angles
        radii, ticks = np.histogram(thetas,bins = bins, density = normed)
        #if normed:
         #   radii= radii/np.sum(radii)
        #ticks = (ticks[0:-1] + ticks[1:])/2
        ticks = ticks[0:-1]
        #ticks = np.linspace(0,2*np.pi,len(radii),endpoint = False)

        #--- Width of each bin of the plot
        #width = (2*np.pi)/(len(radii))
        width = (np.max(ticks)-np.min(ticks))/len(radii)

        #--- Polar plot
        #plt.figure()
        #ax = plt.subplot(polar= True)
        ax = plt.gca(projection = 'polar')
        bars = ax.bar(ticks,radii,width = width,bottom = bottom,**kwargs)

        # Set theta zero location to east
        ax.set_theta_zero_location("E")
        if ylim!= 'auto':
            ax.set_ylim(ylim)

        ax.set_xticks((0,np.pi/2,np.pi, 3*np.pi/2,))
        if not radians:
            ax.set_xticklabels(('$0^o$','$90^o$','$180^o$','$270^o$'))
        else:
            ax.set_xticklabels(('$0$','$\pi/2$','$\pi$','$3\pi/2$'))
        ax.grid(linestyle = ':');

        fh = ax.get_figure()
        fh.canvas.draw()
        ytl = [item.get_text() for item in ax.get_yticklabels()]
        blah = ['' for item in ytl[:-2]]
        ytl[:-2] = blah
        ax.set_yticklabels(ytl)
        return bars, ax, (ticks,radii)

def parallelize(*args, axis:int = 0, n_jobs = None, verbose:int = 0,
                useDask:bool = False, **kwargs):
    """
    Decorator function for parallelizing a given function over a specified axis.
    Parameters
    ----------
    func: function
        Function to run in parallel
    data: primary into to 'func'
        data can be array, in which case the parameter 'axis' can be non-zero.
    axis: int
        Axis to parallelize over
    n_jobs, verbose: see Parallel, delayed from joblib. Here, if n_jobs == None, then
        uses about half of the available workers
    *args, **kwargs: Arguments and keyword arguments to 'func' as well as to Parallel

    NOTE: Will first attempt with joblib, failing which, will switch to dask
    Returns
    -------
    out: Output of 'func' that was passed as input
    """
    from joblib import Parallel, delayed
    import numpy as np
    def takeAndSwap(x,ind,axis,axes_swap):
        return np.swapaxes(np.take(x,ind,axis),*axes_swap)

    isArray = False
    if len(args)==0:
        raise IOError('No arguments given!')

    if len(args)<2:
        raise IOError('At least 2 arguments required; the first one must be a function and the second one must be the first argument to the function')
    func = args[0]
    arr = args[1]

    if not callable(func):
        raise IOError('First input must be a function')
    if len(args)>2:
        args = args[2:]
    else:
        args = ()
    if isinstance(arr, np.ndarray):
        isArray = True
        n_iter = arr.shape[axis]
    else:
        n_iter = len(arr)

    if n_jobs == None:
        import os
        n_jobs = os.cpu_count()//2

    try:
        if useDask:
            10/0
        out = Parallel(n_jobs = n_jobs, verbose = verbose)(delayed(func)(arr,*args,**kwargs)for i in range(n_iter))
    except:
        import dask
        print('Using dask instead of joblib')
        out = dask.compute(*[dask.delayed(func)(takeAndSwap(arr,i,axis,(0,axis)),*args,**kwargs) for i in range(n_iter)], num_workers = n_jobs)
    if isArray:
        out = np.asarray(out)
    return out

def timestamp(till = 'hour'):
    """
    Returns timestamp (string) till the specified temporal
    resolution
    """
    import time
    if till.lower() == 'year':
        ts = time.strftime('%Y')
    elif till.lower() == 'month':
        ts = time.strftim('%Y%m')
    elif till.lower() == 'day':
        ts = time.strftime('%Y%m%d')
    elif till.lower() == 'hour':
        ts = time.strftime('%Y%m%d-%H')
    elif (till.lower() == 'minute') | (till.lower() == 'min') :
         ts = time.strftime('%Y%m%d-%H%M')
    else:
        ts = time.strftime('%Y%m%d-%H%M%S')
    return ts

def to_ascii(strList):
    import numpy as np
    """ Convert a list of utf-encoded strings to ascii for saving in HDF file"""
    strList_new = [s.encode(encoding = 'ascii', errors = 'ignore') for s in strList]
    if isinstance(strList,np.ndarray):
        strList_new = np.array(strList_new)
    return strList_new
def to_utf(strList):
    import numpy as np
    """Convert a list of ascii-encoded strings to utf"""
    strList_new = [s.decode(encoding = 'utf-8', errors = 'ignore') for s in strList]
    if isinstance(strList,np.ndarray):
        strList_new = np.array(strList_new)
    return strList_new

def yOffMat(x):
    """
    Given a set of timeseries, returns the matrix, which when subtracted prevents
    overlap of the signals along the y-axis when plotted. The shift is just enough
    to prevent overlap.
    Parameters
    ----------
    x: array, (nSignals, nSamples)
    Returns
    -------
    ys: array, same shape as x
        plot((x-ys).T) prevents
    """
    import numpy as np
    return np.cumsum(np.insert(np.abs(x.min(axis = 1))[:-1] +\
                               np.abs(x.max(axis = 1))[1:],0,0))[:,np.newaxis]
