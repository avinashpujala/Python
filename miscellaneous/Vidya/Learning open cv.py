# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 11:01:04 2018

@author: pujalaa
"""

#
#from PyQt5 import QtWidgets
#import pyqtgraph as pg
import numpy as np
import os, sys
import importlib

codeDir = r'c:/users/pujalaa/Documents/Code/Python/code'
sys.path.append(codeDir)

import apCode.FreeSwimBehavior as fsb
import apCode.FileTools as ft

#%%
imgDir = r'S:\Avinash\Ablations and behavior\Alx\2018\Apr\20180411-AlxKade-conv42hpf-abl60hpf\f1_ctrl\spont_300Hz_pre_15 mins'

#--- Remove hanging files resulting from program crash

