# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%


import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
import importlib
import tifffile as tff
import ScanImageTiffReader as sitr
import imreg_dft as ird

sys.path.append(r'V:/code/python/code')
import apCode.volTools as volt
#import apCode.behavior.FreeSwimBehavior as fsb
import apCode.SignalProcessingTools as spt
import apCode.FileTools as ft


#%% Inputs
dir_imgs = r'Y:\Avinash\Head-fixed tail free\201906714_full expt\f1\t01_ca'

files_ca = ft.findAndSortFilesInDir(dir_imgs,ext = 'tif')

#%% Read images

tic = time.time()
with sitr.ScanImageTiffReader(os.path.join(dir_imgs,files_ca[0])) as reader:
    I_si = reader.data()
print("SI time = {}".format(time.time()-tic))



#%%
ch_camTrig = 'patch1'
ch_stim = 'patch3'
frameRate = 50

path_bas = os.path.join(os.path.split(dir_imgs)[0], 't01_bas.16ch')
import apCode.ephys as ephys
bas = ephys.importCh(path_bas, nCh=16)
print(bas.keys())

inds_stim = spt.levelCrossings(bas[ch_stim], thr = 2)[0]
dInds = np.diff(bas['t'][inds_stim])
inds_del = np.where(dInds<15)[0]+1
inds_stim = np.delete(inds_stim, inds_del)
inds_camTrig = spt.levelCrossings(bas[ch_camTrig], thr = 2)[0]
dInds = np.diff(bas['t'][inds_camTrig])
inds_del = np.where(dInds<=(0.5/frameRate))[0]+1
inds_camTrig = np.delete(inds_camTrig, inds_del)
# plt.plot(bas[ch_camTrig])
# plt.plot(inds_camTrig,bas[ch_camTrig][inds_camTrig],'o')
print('# stims = {}, # of camera triggers = {}'.format(len(inds_stim), len(inds_camTrig)))

#%%
#%%
maxAllowedTimeBetweenStimAndCamTrig = 0.5 # In sec
t_preStim = 1
t_postStim = 20
Fs_bas = 6000
Fs_ca = 50
n_preStim = t_preStim*Fs_bas
n_postStim = t_postStim*Fs_bas
inds_camTrigNearStim = inds_camTrig[spt.nearestMatchingInds(inds_stim, inds_camTrig)]
t_stim = bas['t'][inds_stim]
t_camTrigNearStim = bas['t'][inds_camTrigNearStim]
inds_tooFar = np.where(np.abs(t_stim-t_camTrigNearStim)>maxAllowedTimeBetweenStimAndCamTrig)[0]


# plt.plot(bas['t'],bas[ch_stim])
# plt.plot(bas['t'],- bas[ch_camTrig])
# plt.plot(bas['t'][inds_camTrigNearStim], bas[ch_camTrig][inds_camTrigNearStim],'o')




