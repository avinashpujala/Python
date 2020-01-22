# -*- coding: utf-8 -*-
"""
Routine for extracting fish midlines from a series of images
Created on Sat Mar  9 17:31:24 2019

@author: pujalaa
"""

#%%
import numpy as np
import os, sys, time 
codeDir = r'v:/code/python/code'
sys.path.append(codeDir)
import apCode.behavior.FreeSwimBehavior as fsb
import apCode.volTools as volt
importlib.reload(fsb)
import matplotlib.pyplot as plt
import apCode.FileTools as ft


#%% Inputs
dir_imgs = r'F:\Avinash\Ablations & Behavior\RS neurons\M homologs\20190308\20190309_behavior\f3_abl_vibAmpOnly_amp_3\fastDir_03-14-19-065345'
headDiam = 1 # Approximate head diameter in mm (for determining head position by weighted average)


#%% Compute background image
print('Computing background...')
img_back = fsb.track.computeBackground(dir_imgs)

print('Estimating pixel size...')
pxlSize = fsb.getPxlSize(img_back)[0]

#%% Find fish position
imgNames = ft.findAndSortFilesInDir(dir_imgs, ext = 'bmp')
r = int(0.5*headDiam/pxlSize)

print('Estimating fish position...')
from sklearn.externals.joblib import Parallel, delayed
from skimage.io import imread
fp = Parallel(n_jobs=32, verbose = 1)(delayed(fsb.track.findFish)(imread(os.path.join(dir_imgs,imgName)),
                                      back_img = img_back, r = r) for imgName in imgNames)
fp = np.array(fp)

#%% Sanity check - Look at fish position trajectories
nFramesInTrl = 750
fp_trl = ft.sublistsFromList(fp,nFramesInTrl)
plt.figure(figsize = (16,16))
plt.imshow(img_back, cmap = 'gray')
for trl, fp_ in enumerate(fp_trl):
    fp_ = np.array(fp_)
    plt.plot(fp_[:,0], fp_[:,1],'.-', markersize = 4, color = plt.cm.tab20(trl))


