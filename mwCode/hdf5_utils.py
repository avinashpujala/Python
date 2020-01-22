import h5py
import numpy as np

def h5read(file):
    with h5py.File(file, 'r') as f:
        return np.array(f['data'][:])

def h5write(file, array):
    with h5py.File(file, 'w') as f:
        f.create_dataset('data', data=array)
