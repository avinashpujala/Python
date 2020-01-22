# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 11:01:04 2018

@author: pujalaa
"""
import apCode.FreeSwimBehavior as fsb
import importlib
importlib.reload(fsb)
#imgDir_fast = r'S:\Avinash\Ablations and behavior\Alx\2018\Mar\20180321\f5_ctrl'


fishLevelDir = r'S:\Avinash\Ablations and behavior\Alx\2018\Mar\20180321\toBeSorted'
fsb.sortFastAndSlowVibAndDark(fishLevelDir)

#%%  Test Parallel

from joblib import Parallel, delayed
import numpy as np

Parallel(n_jobs = 10, verbose = 5)(delayed(np.sqrt)(jj) for jj in range(100))

