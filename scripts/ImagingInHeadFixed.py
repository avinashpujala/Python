# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%


import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.append(r'V:/code/python/code')
import apCode.volTools as volt
import apCode.behavior.FreeSwimBehavior as fsb
import importlib
import apCode.geom as geom
importlib.reload(fsb)
importlib.reload(geom)
#import h5py
import apCode.SignalProcessingTools as spt
from apCode.spectral import wavelet
import apCode.FileTools as ft
importlib.reload(wavelet)

#%% Blah
#imgDir = r'Y:\Avinash\Ablations and Behavior\M-homologs\20190521_fsb\f1_ctrl5\fastDir_05-24-19-041823'
imgDir = r'Y:\Avinash\Ablations and Behavior\M-homologs\20190521_fsb\f3_abl4\fastDir_05-24-19-041917'
fsb.sortIntoTrls(imgDir, 750)

#%%
blah = geom.smoothen_curve(fp.T)

