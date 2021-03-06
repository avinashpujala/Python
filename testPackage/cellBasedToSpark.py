

# -*- coding: utf-8 -*-

# Code to convert cell-based neural activity data from the Ahren's lab into a format usable in Thunder
#
# The main function can be imported and used in another piece of code, or the file can be executed to run an
# interactive version of the conversion

from scipy.io import loadmat
from numpy import array, hstack, fromfile, float32, double, squeeze, reshape
from json import dump
from os.path import realpath, expanduser

def _getDataFromMat(path, name, suffix=''):
    suffix = suffix + '.mat'
    filePath = path + name + suffix
    try:
        return loadmat(filePath)[name]
    except NotImplementedError:
        raise NotImplementedError("currently does not support newer HDF5 Matlab file format")
    except IOError:
        raise ValueError("cannot open mat file {}".format(filePath))

def _getDataFromStackf(path, name, suffix=''):
    suffix = suffix + '.stackf'
    filePath = path + name + suffix
    try:
        return fromfile(filePath, dtype=float32)
    except IOError:
        raise ValueError("cannot open stack file {}".format(filePath))

def positionGetter(data, i):
    f = lambda v: v[i][0]
    return array(map(f, data))

def cellBasedToSpark(readPath, writePath, suffix=''):

    readPath = realpath(expanduser(readPath)) + '/'
    writePath = realpath(expanduser(writePath)) + '/'

    print("reading {}".format(readPath + 'cell_resp_dim' + suffix))
    ncells, ntimes = tuple(*_getDataFromMat(readPath, 'cell_resp_dim', suffix))

    print("reading {}".format(readPath + 'cell_info' + suffix))
    cellInfo = squeeze(_getDataFromMat(readPath, 'cell_info', suffix))
    xyCoords = positionGetter(cellInfo, 0)
    zCoords = positionGetter(cellInfo, 2)
    coords = hstack((xyCoords, zCoords))
    coords = coords - 1

    print("reading {}".format(readPath + 'cell_resp' + suffix))
    data = _getDataFromStackf(readPath, 'cell_resp', suffix)
    data = reshape(data, (ncells, ntimes), order='F')

    combined = hstack((coords, data))

    outputFile = writePath + 'data.bin'
    print("writing {}".format(outputFile))
    try:
        combined.astype(double).tofile(outputFile)
    except IOError:
        raise ValueError("cannot open file {}".format(outputFile))

    jsonData = {"valuetype" : "float",
                "keytype" : "float",
                "nvalues" : int(ntimes),
                "nkeys": 3}
    jsonFile = writePath + 'conf.json'
    print("writing {}".format(jsonFile))
    try:
        jsonFid = open(jsonFile, 'w')
    except IOError:
        raise ValueError("cannot open file {}".format(outputFile))
    dump(jsonData, jsonFid, indent=3)
    jsonFid.close()


if __name__ == "__main__":
    readPath = input("Path to data: ")
    writePath = input("Path to output: ")
    suffix = input("Suffix: ")
    print("converting data")
    cellBasedToSpark(readPath, writePath, suffix)

    print("conversion complete")

    print("conversion complete")


