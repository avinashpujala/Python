# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 09:28:56 2014
Convolution of signal with calcium impulse response function (CIRF)
Convolution of signal with Gaussian kernel of given width
@author: nvladimus
"""
import numpy as np
def conv1exp(x,tau):
 """ Convolution with single-exponent kernel, tau is decay time constant, time in stacks 
 """
 kerWindow = 50 
 r = np.linspace(0, kerWindow) 
 ker1exp = np.r_[np.zeros(kerWindow),np.exp(-r/tau)]
 y = np.convolve(ker1exp/ker1exp.sum(),x,mode='same')
 return y

def conv2exp(x,tau1,tau):
 """ Convolution with double-exponent kernel, time in stacks """
 kerWindow = 50 
 r = np.linspace(0, kerWindow) 
 ker2exp = np.r_[np.zeros(kerWindow-1),np.exp(-r/tau) - np.exp(-r/tau1)]
 y = np.convolve(ker2exp/ker2exp.sum(),x,mode='same')
 return y
 
def gauss(x,window = 50):
     """ Gaussian convolution with sigma = window/6 """
     sigma = window/6.
     r = np.linspace(-window/2., window/2., window + 1)
     w = np.exp(-r**2/(2*sigma**2))
     y = np.convolve(w/w.sum(),x,mode='same')
     return y