# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 00:39:12 2018

@author: pujalaa
"""

def oknotP(n,k,L,etamax):
    import numpy as np
    mid = np.linspace(-etamax-L,etamax,n-k+2)
    delta = np.sum(L)/(n-k+2-1)
    beg = mid[0] + np.arange(-1*(k-1),0)*delta
    endd= mid[-1] + np.arange(1,k)*delta    
    return np.r_[beg, mid, endd]