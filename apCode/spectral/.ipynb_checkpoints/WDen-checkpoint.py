# -*- coding: utf-8 -*-
"""
Created on Sun May 19 03:56:40 2019

@author: pujalaa
"""

#!/home/liulab/shin/python/bin/python
# 
# WDen.py
#
# WDen.py contains functions for wavelet denoising The functions are named following 
# the naming conventions used by MATLAB Wavelet Toolbox.
#
# Programmed by Hyunjin Shin 
#

#from numpy import *
import numpy as np
#from math import *
import math as mth
import pywt
#import sys
      
#################################
#
# wden(x, tptr, sorh, scal, n, wname) does wavelet denoising. 
#
def wden(x, tptr = 'sqtwolog', sorh = 'soft', scal = 'mln', n = None,\
         wname = 'db2'):
    """
    # x, input signal to be denoised
    # tptr, threshold selection rule. See thselect.
    # sorh, threshold type. See wthresh
    # scal = 'one', for no threshold rescaling
    #      = 'sln', for rescaling using a single estimation of level noise based 
    #               on the first detail coefficients
    #      = 'mln', for rescaling done using level dependent estimation
    # wname, wavelet name
    #
    """
    # epsilon stands for a very small number
    eps = 2.220446049250313e-16
    
    # decompose the input signal. Symetric padding is given as a default.
    coeffs = pywt.wavedec(x, wname, mode = 'symmetric', level = n)
    
    # threshold rescaling coefficients
    if scal == 'one':
        s = 1
    elif scal == 'sln':
        s = wnoisest(coeffs)
    elif scal == 'mln':
        s = wnoisest(coeffs, level = n)
    else: 
        raise ValueError('Invalid value for scale, scal = %s' %(scal))
    
    # wavelet coefficients thresholding
    coeffsd = [coeffs[0]]
    for i in range(0, n):
        if tptr == 'sqtwolog' or tptr == 'minimaxi':
            th = thselect(x, tptr)
        else:
            if len(s) == 1:
                if s < np.sqrt(eps) * np.max(coeffs[1+i]): th = 0
                else: th = thselect(coeffs[1+i]/s, tptr)
            else:
                if s[i] < np.sqrt(eps) * np.max(coeffs[1+i]): th = 0
                else: th = thselect(coeffs[1+i]/s[i], tptr)
        
        ### DEBUG
#        print "threshold before rescaling:", th
        ###
        
        # rescaling
        if len(s) == 1: th = th *s
        else: th = th *s[i]
        
        #### DEBUG
#        print "threshold:", th
        ####
        
        coeffsd.append(np.array(wthresh(coeffs[1+i], sorh, th)))
        
    # wavelet reconstruction 
    xdtemp = pywt.waverec(coeffsd, wname, mode = 'symmetric')
    
    # get rid of the extended part for wavelet decomposition
    extlen = int(abs(len(x)-len(xdtemp))/2)
    xd = xdtemp[extlen:len(x)+extlen]
    
    return xd
    
    
#################################
#
# thselect(x, tptr) returns threshold x-adapted value using selection rule defined by string tptr.
# 
# tptr = 'rigrsure', adaptive threshold selection using principle of Stein's Unbiased Risk Estimate.
#        'heursure', heuristic variant of the first option.
#        'sqtwolog', threshold is sqrt(2*log(length(X))).
#        'minimaxi', minimax thresholding. 
        
def thselect(x, tptr):
    x = np.array(x) # in case that x is not an array, convert it into an array
    l = len(x)
    
    if tptr == 'rigrsure':
        sx2 = [sx*sx for sx in np.absolute(x)]
        sx2.sort()
        cumsumsx2 = np.cumsum(sx2)        
        risks = []
        for i in range(0, l):
            risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
        mini = np.argmin(risks)
        th = np.sqrt(sx2[mini])
    if tptr == 'heursure':
        hth = np.sqrt(2*np.log(l))
        
        # get the norm of x
        normsqr = np.dot(x, x)
        eta = 1.0*(normsqr-l)/l
        crit = (mth.log(l,2)**1.5)/np.sqrt(l)
        
       
        if eta < crit: th = hth
        else: 
            sx2 = [sx*sx for sx in np.absolute(x)]
            sx2.sort()
            cumsumsx2 = np.cumsum(sx2)        
            risks = []
            for i in range(0, l):
                risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
            mini = np.argmin(risks)
            
            
            ### DEBUG
#            print "risk:", risks[mini]
#            print "best:", mini
#            print "risks[222]:", risks[222]
            ###
            
            rth = np.sqrt(sx2[mini])
            th = np.min(hth, rth)     
    elif tptr == 'sqtwolog':
        th = np.sqrt(2*np.log(l))
    elif tptr == 'minimaxi':
        if l <32: 
            th = 0
        else: 
            th = 0.3936 + 0.1829*mth.log(l, 2)
    else:
        raise ValueError('Invalid value for threshold selection rule, tptr = %s' %(tptr))
    
    return th
    
#################################
#
# wthresh(x, sorh, t) returns the soft (sorh = 'soft') or hard (sorh = 'hard') 
# thresholding of x, the given input vector. t is the threshold.
# sorh = 'hard', hard trehsholding
# sorh = 'soft', soft thresholding
#    
    
def wthresh(x, sorh, t):    
    if sorh == 'hard':
        y = [e*(abs(e) >= t) for e in x]
    elif sorh == 'soft':
        y = [((e<0)*-1.0 + (e>0))*((abs(e)-t)*(abs(e) >= t)) for e in x]
    else:
        raise ValueError('Invalid value for thresholding type, sorh = %s' %(sorh))
    
    return y

#################################
#
# wnoisest(coeffs, level = None) estimates the variance(s) of the given detail(s)
#
# coeffs = [CAn, CDn, CDn-1, ..., CD1], multi-level wavelet coefficients
# level, decomposition level. None is the default.
#

def wnoisest(coeffs, level= None):
    
    l = len(coeffs) - 1
    
    if level == None:
        sig = [np.abs(s) for s in coeffs[-1]]
        stdc = median(sig)/0.6745
    else:
        stdc = []
        for i in range(0, l):
            sig = [np.abs(s) for s in coeffs[1+i]]
            stdc.append(median(sig)/0.6745)
    
    return stdc

#################################
#
# median(data) returns the median of data
#
# data, a list of numbers.
#       
 
def median(data):
        
    temp = data[:]
    temp.sort()
    dataLen = len(data)
    if dataLen % 2 == 0: # even number of data points
        med = (temp[int(dataLen/2 -1)] + temp[int(dataLen/2)])/2.0
    else:
        med = temp[int(dataLen/2)]
        
    return med   