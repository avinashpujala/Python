

#%% Importing some basics
import numpy as np
import matplotlib.pyplot as plt
import os, time, scipy, sys
import importlib
import pandas as pd
import scipy as sp


codeDir = r'C:\Users\pujalaa\Documents\Code\Python\code'
sys.path.insert(1,codeDir)

import nvCode.tifffile as tff
import apCode.SignalProcessingTools as spt
import apCode.volTools as volt
import apCode.spim.imageAnalysis as ia
import apCode.AnalyzeEphysData as aed
import seaborn as sns

#%matplotlib inline
time.ctime()


#%% Data directory
inputDir = r'S:\Avinash\SPIM\Alx\10-3-2015_AlxRG-relaxed-dpf4\Fish1\g50-t1-ipi6-f1_20151003_172727\pyData'
fName = 'Cell data for clustering_20170630T025141_.mat'

eCh = 1 # ephys channel to use

#%% Point to pyData on disk
import h5py
import apCode.FileTools as ft
#pyDir = r'S:\Avinash\SPIM\Alx\8-9-2015_Alx RG x 939_4dpf\Ephys\Fish2'
pyFileName = ft.findAndSortFilesInDir(inputDir,ext = 'mat')[-1]

pyData_file = h5py.File(os.path.join(inputDir,pyFileName),'r')
pyData_disk = pyData_file['data']

pyData = {}
pyData['stim'],pyData['swim'] = {},{}
pyData['stim']['amps'] = pyData_disk['stim']['amps'][:]
pyData['stim']['inds'] = pyData_disk['stim']['inds'][:].ravel().astype(int)-1
pyData['swim']['startInds'] = pyData_disk['swim']['startInds'][:].astype(int)-1
pyData['swim']['distFromLastStim'] = pyData_disk['swim']['distFromLastStim'][:]
pyData['time'] = pyData_disk['t'][:]
pyData['smooth'] = np.transpose(pyData_disk['smooth']['burst'][:])

pyData['samplingRate'] = np.int(1/(pyData['time'][1]-pyData['time'][0]))

pyData['cells'] = pyData_file['cells']

alreadyRegressed = False
if np.any(np.array(list(pyData_file.keys()))=="regr"):
    alreadyRegressed = True
    pyData['regr'] = pyData_file['regr']
    regr = {}
    for key in pyData['regr'].keys():   
        if isinstance(pyData['regr'][key],h5py.Dataset):
            regr[key] = pyData['regr'][key][:]
        else:
            regr[key] = pyData['regr'][key]

#%% Read cell data
tic = time.time()
print('Reading...')
sys.stdout.flush()
data = {}
for key in pyData['cells'].keys():    
    print(key),sys.stdout.flush()       
    if 'dataset' in str(pyData['cells'][key]).lower():        
        data[key] = pyData['cells'][key][:]
    else:
        data[key] = {}
        for key_key in pyData['cells'][key].keys():           
            print('\t', key_key), sys.stdout.flush()
            if isinstance(pyData['cells'][key][key_key],h5py.Group):
                data[key][key_key] = pyData['cells'][key][key_key]
            else:
                data[key][key_key] = pyData['cells'][key][key_key][...]
blah = {}
print('Reading object references...')
for key in data['info']:
    item = data['info'][key]
    blah[key]=[]
    print(key)
    #sys.stdout.flush()
    for val in item:
        if isinstance(val[0],h5py.Reference):
            blah[key].append(pyData['cells'][val[0]][0])
    try:
        blah[key] = np.array(blah[key]).ravel()
    except:
        pass
data['info'] = blah

print('Converting from matlab to python indices...')
for cNum, cInds in enumerate(data['info']['inds']):
    data['info']['inds'][cNum] = cInds.astype(int)-1
data['info']['center'] = data['info']['center'].astype(int)-1
data['info']['slice'] = data['info']['slice'].astype(int)-1
data['info']['x_minmax'] = data['info']['x_minmax'].astype(int)-1
data['info']['y_minmax'] = data['info']['y_minmax'].astype(int)-1
data['inds']['inMask'] = data['inds']['inMask'].astype(int)[0]-1

if alreadyRegressed:
    data['regr']= pyData['regr']
data['pathToData'] = os.path.join(inputDir,pyFileName)
print('\n', int(time.time()-tic),'sec')
print(time.ctime())

print('Geting indices for stacks in ephys data and stims in optical data...')
data['inds']['stim'] = spt.nearestMatchingInds(pyData['time'].ravel()[pyData['stim']['inds']],data['time'])
data['inds']['stackInit'] = spt.nearestMatchingInds(data['time'],pyData['time'],processing = 'parallel')

# Detrending dF/F signals
print('Detrending dF/F signals...')
for cc in np.arange(np.shape(data['dFF'])[1]):
    data['dFF'][:,cc] = ia.detrendCa(data['dFF'][:,cc],data['inds']['stim'])

# An adjustment for some datasets
data['avg'] = np.transpose(data['avg'],[0,2,1])
np.shape(data['avg'])

#%% Check to see if stimulus onset indices have been detected correctly
stimInd = 4
periTime_ephys = 0.1
periTime_opt = 8

t_ephys = pyData['time'].ravel()
y_ephys = spt.standardize(pyData['smooth'][:,eCh])
stimInds_ephys = pyData['stim']['inds']
periInd_ephys = np.ceil(periTime_ephys*pyData['samplingRate']).astype(int)
tInds_ephys = np.arange(stimInds_ephys[stimInd]-periInd_ephys,stimInds_ephys[stimInd]+ periInd_ephys) 


t_opt = data['time']
y_opt = spt.standardize(np.mean(data['dFF'],axis = 1))
stimInds_opt = data['inds']['stim']
periInd_opt = np.ceil(periTime_opt/(t_opt[1]-t_opt[0]))
tInds_opt = np.arange(stimInds_opt[stimInd]-periInd_opt+1, stimInds_opt[stimInd] + periInd_opt).astype(int)


plt.style.use(['dark_background','seaborn-colorblind','seaborn-poster'])
plt.figure(figsize = (16,10))
plt.subplot(211)
plt.plot(t_ephys[tInds_ephys],y_ephys[tInds_ephys], label = 'ephys')
plt.axvline(x = t_ephys[stimInds_ephys[stimInd]], linestyle = '--', color = 'm', 
            alpha = 0.5, label = 'stim onset - ephys')
plt.axvline(x = t_opt[stimInds_opt[stimInd]], linestyle = '--', color = 'y', alpha = 0.5, 
            label = 'stim onset - imaging')
tLim = t_ephys[tInds_ephys][[0,-1]]
plt.xlim(tLim)
plt.ylim(0,1)
plt.legend(loc = 'upper left')

plt.subplot(212)
plt.plot(t_opt[tInds_opt],y_opt[tInds_opt],  label ='mean $ \Delta F/F  $')
plt.axvline(x = t_ephys[stimInds_ephys[stimInd]], linestyle = '--', color = 'm', 
            alpha = 0.5, label = 'stim onset - ephys')
plt.axvline(x = t_opt[stimInds_opt[stimInd]], linestyle = '--', color = 'y', alpha = 0.5, 
            label = 'stim onset - imaging')
tLim = t_opt[tInds_opt][[0,-1]]
plt.xlim(tLim)
plt.ylim(0,1)
plt.legend(loc = 'upper left')
plt.xlabel('Time (sec)')
plt.suptitle('Checking stimulus onsets in relation to activity, stim ind =  ' + str(stimInd), fontsize = 16);

#%% Agglomerative clustering 
n_clusters = 10
tic = time.time()
X = spt.standardize(data['dFF'][:,data['inds']['inMask']].T,axis =1)
inds = np.round(np.linspace(0,np.shape(X)[0]-10,100)).astype(int)
X = X[inds,:]
ac = ia.agglomerativeClustering(X,n_clusters,linkage = 'ward')
print(int(time.time()-tic),'sec')

color_idx = np.linspace(0,0.9,n_clusters)
colors = plt.cm.hsv(color_idx)
ephys = spt.standardize(pyData['smooth'][:,eCh])**(0.5)
ia.plt.plotCentroids(data['time'],ac.cluster_centers_,data['inds']['stim'],pyData['time'],
              ephys, colors = colors)

print(time.ctime())

#%%
nSamples = np.shape(X)[0]
foo = np.zeros_like(ac.children_)
for itemNum, item in enumerate(ac.children_):
    print(itemNum)
    if item[0] <= nSamples:
        foo[itemNum,0] = ac.labels_[item[0]]
    if item[1] <= nSamples:
        foo[itemNum,1] = ac.labels_[item[1]]


