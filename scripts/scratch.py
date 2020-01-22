# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 00:52:22 2019

@author: pujalaa
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import os
import sys
sys.path.append(r'v:/code/python/code')
import apCode.SignalProcessingTools as spt


#%%
n_samples = 100
wts = np.array([30, 60, 100, 0, 130])
wts = wts/wts.sum()

var = {}
var['gpa'] = np.random.choice(np.arange(3.0,4.0,0.1), size = n_samples, replace = True)
var['sat_score'] = np.random.choice(np.arange(1200,1601), size = n_samples, replace = True)
var['hours_of_research_experience'] = np.random.choice(np.arange(20,110), size = n_samples, 
   replace = True)
var['age_of_applicant'] = np.random.choice(np.arange(18,22),size = n_samples, replace = True)
var['harvard_entrance_test_score'] = np.random.choice(np.arange(70, 101), size = n_samples,\
   replace = True)

X = np.array([var['gpa'], var['sat_score'], var['hours_of_research_experience'], var['age_of_applicant'],\
     var['harvard_entrance_test_score']])
X = spt.standardize(X.T,axis = 0)

y = np.dot(X,wts)
y = y + np.random.rand(len(y))*np.std(y)
y = spt.standardize(y)*0.8 + 0.2

var['chances_of_acceptance_into_harvard'] = y

data = var

#%% 
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X,y)

print('Regression coefficients are {}, and intercept is {}'.format(reg.coef_, reg.intercept_))


#%%
dir_save = r'P:\pujalaa\Akash\20190701'
np.save(os.path.join(dir_save, 'harvardRegression.npy'), data)