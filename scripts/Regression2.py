# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:56:57 2015

@author: pujalaa
"""

#%% Specifying Data Directories
# Specify directories with data
rawDir = 'Y:\SPIM\Avinash\Registered\May 2015\5-9-2015_relaxed_dpf4\Ephys/Fish2/'
epDir = 'Y:/Avinash/May 2015/Fish2/Ephys/'
fileName = 'G50_T1_IPI60.10chFlt'

procDir = rawDir + 'proc/'
serDir = rawDir + 'series/'
regDir = 'Y:/Avinash/May 2015/Fish2/'


#%% Specifying fixed variables
exposureTime = 18e-3
baselinePeriod = 10


#%% Importing Requisite Functions
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import split,join,sep
from IPython.html.widgets import interact, interactive, fixed

#from skimage.filter import threshold_otsu

#from thunder import RegistrationModel, Registration, KMeansModel, KMeans, ICA, PCA, NMF
#from thunder import Colorize
#image = Colorize.image


import sys
sys.path.insert(0, 'Y:/Avinash/Code/Python/code/codeFromNV')
sys.path.insert(0, 'Y:/Avinash/Code/Python/code/util')

#sc.addPyFile('/groups/koyama/home/pujalaa/code/codeFromNV/importEphys.py')
#sc.addPyFile('/groups/koyama/home/pujalaa/code/codeFromNV/conv.py')
#sc.addPyFile('/groups/koyama/home/pujalaa/code/codeFromNV/volTools.py')
#sc.addPyFile('/groups/koyama/home/pujalaa/code/codeFromNV/tifffile.py')
#sc.addFile('/groups/koyama/home/pujalaa/code/codeFromNV/tifffile.c')


import AnalyzeEphysData as aed
import volTools as volt
import tifffile as tif
#import analysisTools as ant
import time
import scipy.io as sio
# import cellBasedToSpark as cbs

# import seaborn as sns
#sns.set_context('notebook')

%matplotlib inline

#%% Loading Ephys Data

#Loading raw data
startTime = time.time()
data = aed.import10ch(epDir + fileName)
print 'Loaded raw data'
print int(time.time()-startTime), 'sec'

#Loading data preprocessed by matlab
startTime = time.time()
import h5py
f = h5py.File(epDir + 'pyData.mat')
pyData = f['pyData']
print 'Pointing to processed data on disk'

# Append pyData keys to data
for key in pyData.keys():
    data[key] = pyData[key]  

# Collecting some important variables
data['samplingRate'] = 6000;

stim ={}
stim['inds'] = np.round(pyData['stim']['inds'][:]-1).astype(int)
stim['amps'] = pyData['stim']['amp'][:]
stim['times'] = pyData['stim']['inds'][:]

print int(time.time()-startTime), 'sec'

#%% Loading cell data

f = h5py.File(regDir + 'cell_resp3.mat')
cellResp = np.transpose(np.array(f['cell_resp3'][:])) # cellResp array similar to matlab equivalent
f.close()
data['cells'] = {}
data['cells']['raw'] = cellResp


cellInfo = np.squeeze(sio.loadmat(regDir + 'cell_info_processed.mat')['cell_info'])
cKeys = cellInfo.dtype.fields.keys()

stack = {}
stack['inds'] = {}
stack['inds']['all'] = aed.stackInits(data)
stack['int'] = np.round(np.float(np.int(np.median(np.diff(stack['inds']['all']))))/data['samplingRate'], decimals = 2)
stack['times'] = data['t'][stack['inds']['all']]
stack['avg'] = tif.imread(regDir + 'ave.tif')
stack['dim'] = list(np.shape(stack['avg']))
stack['dim'].append(len(stack['inds']))

print 'Finding stacks coincident with stimuli'
stack['inds']['shock'] = np.zeros((len(stim['inds']),1)).astype(int)
for shock, val in enumerate(stim['inds']):
    print shock
    stack['inds']['shock'][shock]= np.argmin(np.abs(stack['inds']['all']-stim['inds'][shock]))
   # print b
    stack['inds']['shock'] = np.unique(stack['inds']['shock'])
    
    
#%% Converting raw cell traces to dF/F & filterning a bit
nBaselineStacks = int(baselinePeriod/stack['int'])
baselineInds= []



print 'Done!'
