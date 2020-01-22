# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:53:30 2019
Contains routines to work with HDF (Hierarchical Data Files) 
files such as those read by h5py and saved by MATLAB save(..., '-v7.3')
which are typically over 2GB

@author: pujalaa
"""

#import h5py

def checkKeys(dict):
    import scipy.io as spio
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = toDict(dict[key])
    return dict 

def createOrAppendToHdf(hFile, keyName, arr, verbose = False):
    """
    Create a dataset (can be within in group) within a given HDF file
    if the dataset with specified name doesn't exist else append to the
    existing dataset
    Parameters
    ----------
    hFile: HDF file object
        HDF file object to create or append dataset in.
    keyName: str
        Path to the dataset in the HDF file. For e.g., 'foo' or 'foo/bar'.
    arr: array
        Array to write to the dataset.
    Returns
    -------
    hFile: HDF file object
        HDF file object with dataset.
    """
    if not keyName in hFile:
        if verbose:
            print(f'Creating {keyName} in hdf file')            
        hFile.create_dataset(keyName, data = arr,\
                             maxshape = (None, *arr.shape[1:]),\
                             compression = 'lzf')
    else:
        if verbose:
            print(f'Appending to {keyName} in h5 file')
        hFile[keyName].resize((hFile[keyName].shape[0] + arr.shape[0]), axis = 0)
        hFile[keyName][-arr.shape[0]:] = arr
    return hFile

def h5pyToDict(obj):
    import numpy as np
    """
    Copy h5py file to dictionary variable
    """
    d = {}
    for key, val in zip(obj.keys(),obj.values()):#         
        d[key] = np.squeeze(np.array(val))
    return d

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    import scipy.io as spio
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return checkKeys(data)  

def toDict(matobj):
    import scipy.io as spio
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = toDict(elem)
        else:
            dict[strg] = elem
    return dict

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    import h5py
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic, verbose = False):
    """
    Does what the name says.
    Parameters
    ----------
    hFile: hdf file object
    path: string
        Path in hFile where to save dictioary. For instance if you want to create
        a group 'x' where the dictionary is to be saved then specify path as '/x/'
    dic: dictionary
        Dictionary to store
    verbose: boolean
        If True prints out the names of the dictionary items that could not be 
        saved
        
    """
    import numpy as np
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            try:
                recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
            except:
                if verbose:
                    print('Could not save {} of type {}'.format(key, type(item)))
        else:
#            raise ValueError('Cannot save %s type'%type(item))
            if verbose:
                print('Could not save {} of type {}'.format(key, type(item)))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    import h5py
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    import h5py
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
#            ans[key] = item.value
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def strFromHDF(hFile, refList):
    """
    Given an hdf (h5py) file and reference list holding strings, returns
    a list of strings
    """
    strList = [u''.join(chr(c) for c in hFile[ref]) for ref in refList]
    return strList
    

if __name__ == '__main__':
    import numpy as np

    data = {'x': 'astring',
            'y': np.arange(10),
            'd': {'z': np.ones((2,3)),
                  'b': b'bytestring'}}
    print(data)
    filename = 'test.h5'
    save_dict_to_hdf5(data, filename)
    dd = load_dict_from_hdf5(filename)
    print(dd)
    # should test for bad type

 