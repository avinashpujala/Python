# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:02:12 2020

@author: pujalaa
"""
from sklearn.preprocessing import StandardScaler
import numpy as np


class Scaler(StandardScaler):
    """
    Class for scaling data (features) during preprocessing. Essentially, a
    wrapper around sklearn.preprocessing.StandardScaler with the additional
    option of standardizing (i.e., mapping values from 0 to 1).
    """

    def __init__(self, with_mean=True, with_std=True,
                 standardize: bool = False, **kwargs):
        """
        Parameters
        ----------
        with_mean, with_std: see StandardScaler
        standardize: bool
            If True, standardizes instead of normalizing, i.e. values are
            mapped from 0 to 1 based on min and max values.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.standardize_ = standardize
        if standardize:
            self.fit = self._fit
            self.transform = self._transform
            self.fit_transform = self._fit_transform
        else:
            StandardScaler.__init__(self, with_mean=with_mean,
                                    with_std=with_std, **kwargs)

    def _fit(self, X):
        if np.ndim(X) == 1:
            X = X[:, None]
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def _transform(self, X):
        if np.ndim(X) == 1:
            X = X[:, None]
        X = X-self.min_[None, :]
        X = X/(self.max_-self.min_)[None, :]
        return np.squeeze(X)

    def _fit_transform(self, X):
        X = self._fit(X)._transform(X)
        return X
