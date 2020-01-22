# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:56:57 2015

@author: pujalaa
"""

#%% Specifying Data Directories
# Specify directories with data
rawDir = 'Y:\SPIM\Avinash\Registered\May 2015\5-9-2015_relaxed_dpf4\Ephys\Fish2/'
epDir = 'Z:/SPIM/Avinash/Registered/May 2015/5-9-2015_relaxed_dpf4/Ephys/Fish2/'
fileName = 'G50_T1_IPI60.10chFlt'

procDir = rawDir + 'proc/'
serDir = rawDir + 'series/'
regDir = 'Z:/SPIM/Avinash/Registered/May 2015/5-9-2015_relaxed_dpf4/Fish2/G50_T1_IPI60_20150509_162303/'


#%% Specifying fixed variables
exposureTime = 18e-3


#%% Importing Requisite Functions
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import split,join,sep
#from IPython.html.widgets import interact, interactive, fixed

#from skimage.filter import threshold_otsu

#from thunder import RegistrationModel, Registration, KMeansModel, KMeans, ICA, PCA, NMF
#from thunder import Colorize
#image = Colorize.image


import sys
sys.path.insert(0, 'P:/pujalaa/code/codeFromNV')
sys.path.insert(0, 'P:/pujalaa/code/util')

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
f = h5py.File(epDir + 'pyData.mat', driver = 'core')
pyData = f['pyData']
print 'Pointing to processed data on disk'

# Append pyData keys to data
for key in pyData.keys():
    data[key] = pyData[key]  

# Collecting some important variables
data['samplingRate'] = np.float64(6000)
data['baselinePeriod'] = np.float64(10)
data['exposureTime'] = np.float64(18e-3)

stim ={}
stim['inds'] = np.squeeze(np.round(pyData['stim']['inds'][:]-1).astype(int))
stim['amps'] = pyData['stim']['amp'][:]
stim['times'] = pyData['stim']['times'][:]

print int(time.time()-startTime), 'sec'



#%% Loading cell data created by Takashi's scripts
startTime = time.time()
f = h5py.File(regDir + 'cell_resp3.mat')
cellResp = np.transpose(np.array(f['cell_resp3'][:])) # cellResp array similar to matlab equivalent
f.close()
cells ={}
cells['raw'] = cellResp

cellInfo = np.squeeze(sio.loadmat(regDir + 'cell_info_processed.mat')['cell_info'])
cKeys = cellInfo.dtype.fields.keys()
cells['info'] = cellInfo
cells['infoKeys'] = cKeys
print 'Loaded TK-style cell data'
print int(time.time()-startTime), 'sec'





#%% Getting stack information
stack = {}
stack['inds'] = {}
stack['inds']['all'] = aed.stackInits(data)
stack['int'] = np.round(np.int(np.median(np.diff(stack['inds']['all'])))*(1/data['samplingRate']),decimals = 2)
stack['times'] = data['t'][stack['inds']['all']]
stack['avg'] = tif.imread(regDir + 'ave.tif')
stack['dim'] = list(np.shape(stack['avg']))
stack['dim'].append(len(stack['inds']['all']))

# Ignore stims and stacks that don't allow for enough of a pre-stimulus period
data['preStimPeriod'] = data['preStimPeriod'][:]
outImgTrls = []
if stim['times'][0] < data['preStimPeriod']:
    print 'Excluded 1st trial because stim occurs too early w.r.t imaging period'
    outImgTrls.append(0)
    for key in stim.keys():
        stim[key] = np.delete(stim[key],0)
        
if (stim['times'][-1] + data['postStimPeriod']) >= stack['times'][-1]:
    outImgTrls.append(len(stim['times']))
    print 'Excluded last trial because stim occurs too late w.r.t imaging period'
    for key in stim.keys():
        stim[key] = np.delete(stim[key],-1)       
    
        
stack['inds']['shock'] = np.zeros((len(stim['inds']),1)).astype(int)
stack['inds']['shock'] = np.zeros((len(stim['inds']),1)).astype(int)
for shock, val in enumerate(stim['inds']):
    stack['inds']['shock'][shock]= np.argmin(np.abs(stack['inds']['all']-stim['inds'][shock]))

# Checking stack detection
data['camTrigger'] = np.squeeze(data['camTrigger'][:])
subT = data['t'][stack['inds']['all'][1]:stack['inds']['all'][4]]
subCT = data['camTrigger'][stack['inds']['all'][1]:stack['inds']['all'][4]]
plt.figure(figsize = (12,6))
plt.plot(subT,subCT)
plt.hold(True)
plt.plot(data['t'][stack['inds']['all'][1:4]],data['camTrigger'][stack['inds']['all'][1:4]], 'ro')
plt.xlim((data['t'][stack['inds']['all']][1]-0.1,data['t'][stack['inds']['all']][4]))
plt.ylim((0-0.2,4.2));






#%% Finding trials that are outside the imaging period of the expt and excluding them from analysis
tData = dict(pyData['tData'].items())
data['postStimPeriod'] = np.squeeze(pyData['postStimPeriod'][:])
tData['outImgTrls'] = np.squeeze(np.where(stim['inds'] > (stack['inds']['all'][-1]-data['postStimPeriod']/stack['int'])))

trialSegData = dict(pyData['trialSegData'].items())
for key in trialSegData.keys():
    b = np.delete(trialSegData[key][0],tData['outImgTrls'])
    for trl in np.arange(len(b)):         
        trialSegData[key][0] = b
        
# Eliminate trials outside imaging period        
lastTrlTime = pyData[trialSegData['stim'][0,-1]]['time'][0,0]
if (lastTrlTime + data['postStimPeriod']) > stack['times'][-1]:
    for key in trialSegData.keys():
        trialSegData[key] = np.delete(trialSegData[key],-1,axis = 1)
#trialSegData2  = pyData['trialSegData'].items()






#%% Finding stacks coincident with wk and strng swms
wkSwmTrls = np.squeeze(tData['wkSwmTrls'][:].astype(int))
wkSwmTrls = np.delete(wkSwmTrls,np.where(wkSwmTrls >= len(stim['times'])))
wkSwmInds = stim['inds'][wkSwmTrls]
stack['inds']['wkSwm'] = stack['inds']['shock'][wkSwmTrls]

strngSwmTrls = np.squeeze(tData['strngSwmTrls'][:].astype(int))
strngSwmTrls = np.delete(strngSwmTrls,np.where(strngSwmTrls >= len(stim['times'])))
wkSwmInds = stim['inds'][strngSwmTrls]
stack['inds']['strngSwm'] = stack['inds']['shock'][strngSwmTrls]




#%% Converting cell responses to dF/F and filtering a bit
# Extracting baseline inds
nBaselineFrames = (np.ceil(data['baselinePeriod']*stack['int'])).astype(int)
baselineInds = []
for shock,_ in enumerate(stack['inds']['shock']):
    baselineInds = np.r_[baselineInds,stack['inds']['shock'][shock]-nBaselineFrames:\
                         stack['inds']['shock'][shock]-1]
baselineInds  = (np.delete(baselineInds,np.where(baselineInds <=0))).astype(int)
baselineF = np.tile(np.r_['c',np.mean(cells['raw'][:,baselineInds],axis = 1)],\
                    (1,np.shape(cells['raw'])[1]))
# Computing dFF and highpassing a bit
startTime = time.time()
cells['dFF']= np.array((cells['raw']-baselineF)/baselineF)
cells['dFF'] = spt.ChebFilt(cells['dFF'],stack['int'],(1./200),btype = 'highpass',axis = 1)
print int(time.time()-startTime), 'sec'



#%% Segmenting cell responses into trials by stimulus events
def SegmentDataByEvents(X,eventIndices,nPreEventPts,nPostEventPts, axis = 1):
    Y = list(np.nan*np.zeros([len(eventIndices)]))
    for idx,evt in enumerate(eventIndices):    
        inds = np.r_[evt-nPreEventPts:evt+nPostEventPts]      
        delInds = np.setdiff1d(np.arange(np.shape(X)[axis]),inds)       
        Y[idx] = np.delete(X,delInds,axis = axis)     
    return Y

nPreEventPts = (data['baselinePeriod']/stack['int']).astype(int)
nPostEventPts = np.min([55,(data['postStimPeriod']/stack['int']).astype(int)])
trialSegData['cells'] = {}
trialSegData['cells']['raw'] = SegmentDataByEvents(cells['raw'],stack['inds']['shock'],\
                                                   nPreEventPts,nPostEventPts,axis =1)
trialSegData['cells']['dFF'] = list(np.arange(len(trialSegData['cells']['raw'])))

shockStackIdx = nPreEventPts + 1;
baselineInds = np.r_[0:shockStackIdx-2]
totFrames = len(np.r_[-nPreEventPts:nPostEventPts])
trialSegData['cells']['time'] = np.r_[-nPreEventPts:nPostEventPts]
totFrames = len(np.r_[-nPreEventPts:nPostEventPts])
for trl in np.arange(len(stack['inds']['shock'])):
    baselineF = np.abs(np.mean(trialSegData['cells']['raw'][trl][:,baselineInds],axis = 1))
    baselineF = np.r_['c',baselineF]
    baselineF = np.tile(baselineF,(1,len(trialSegData['cells']['time'])))
    trialSegData['cells']['dFF'][trl] =\
    np.array((trialSegData['cells']['raw'][trl]-baselineF)/baselineF)
    
    



#%% Saving stuff
data['tData'] = tData
data['trialSegData'] = trialSegData




