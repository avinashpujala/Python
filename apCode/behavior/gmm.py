# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 03:32:06 2020

@author: pujalaa
"""

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np
import sys
import dask
sys.path.append(r'v:/code/python/code')
import apCode.behavior.headFixed as hf  # noqa: E402
import apCode.SignalProcessingTools as spt  # noqa: E402


def max_min_envelopes(x):
    if np.ndim(x) == 1:
        x_env = spt.emd.envelopesAndImf(x)['env']
        x_env = np.c_[x_env['crests'], x_env['troughs']].T
    else:
        x_env = [dask.delayed(max_min_envelopes)(x_) for x_ in x]
        x_env = np.array(dask.compute(*x_env))
        x_env = np.concatenate(x_env, axis=0)
    return x_env


def predict_on_tailAngles(ta, fitter, subSample=1):
    """
    Given a tail angles array, returns a feature array that can be directly
    input to a trained GMM
    Parameters
    ----------
    ta: array, (nTailPoints, nTimePoints)
        Tail angles array
    scaler: sklearn StandardScaler object pre-fitted to multi-fish data
    nKer: int
        Kernel length in points for convolution of features
    n_pca: int
        Number of PCA components to reduce the feature array to
    Returns
    -------
    labels: array, (nSamples,)
    X: array, (nSamples, nFeatures)
        Feature array ready to be fed into a fitted GMM model.
    """
    scaler, pca, gmm, nKer = fitter['scaler'], fitter['pca'], fitter['gmm'],\
        fitter['nKer']

    df_features = hf.swimEnvelopes_multiLoc(ta)
    arr_feat = np.array(df_features)
    arr_feat_scaled = scaler.transform(arr_feat)
    if nKer is not None:
        arr_feat_scaled =\
            [dask.delayed(spt.causalConvWithSemiGauss1d)(x, nKer*2)
             for x in arr_feat_scaled.T]
        arr_feat_scaled = np.array(dask.compute(*arr_feat_scaled)).T
    if pca is not None:
        X = pca.transform(arr_feat_scaled[::subSample, :])
    else:
        X = arr_feat_scaled[::subSample, :]
    labels = gmm.predict(X)
    return labels, X


def predict_on_tailAngles_svd(ta, fitter, subSample=1):
    """
    Given a tail angles array, returns a feature array that can be directly
    input to a trained GMM
    Parameters
    ----------
    ta: array, (nTailPoints, nTimePoints)
        Tail angles array
    scaler: sklearn StandardScaler object pre-fitted to multi-fish data
    nKer: int
        Kernel length in points for convolution of features
    n_pca: int
        Number of PCA components to reduce the feature array to
    Returns
    -------
    labels: array, (nSamples,)
    X: array, (nSamples, nFeatures)
        Feature array ready to be fed into a fitted GMM model.
    """
    scaler, svd, gmm = fitter['scaler'], fitter['svd'], fitter['gmm']
    use_envelopes = fitter['use_envelopes']
    V = svd.transform(ta.T)
    dv = np.gradient(V)[0]
    ddv = np.gradient(dv)[0]
    X = np.c_[V, dv, ddv]
    if use_envelopes:
        X = max_min_envelopes(X.T).T
    X = scaler.transform(X)
    labels = gmm.predict(X)
    return labels, X


def tailAngles_to_svd_featureArray(ta, n_svd=3, use_envelopes=True):
    """
    Given tail angles array, returns svd-based feature array as well as
    the svd object. The feature array can be used for posture classification
    by GMM or some other clustering algorithm
    Parameters
    ----------
    ta: array, (nPointsAlongTail, nTimePoints)
        Tail angles array
    n_svd: int
        Number of SVD components to use. 3 components typically explains > ~95%
        variance. Test before deciding.
    Returns
    -------
    X_svd: array, (nTimePoints, n_svd*3)
        SVD-based feature array. In addition to the timeseries of the n_svd
        components, this array includes upto the 2nd derivative of these
        timeseries.
    svd: object
        Fitted SVD object
    """
    svd = TruncatedSVD(n_components=n_svd, random_state=143).fit(ta.T)
    V = svd.transform(ta.T)
    dv = np.gradient(V)[0]
    ddv = np.gradient(dv)[0]
    X = np.c_[V, dv, ddv]
    if use_envelopes:
        X = max_min_envelopes(X.T).T
    return X, svd


def train_on_tailAngles(ta, nKer=None, n_gmm=15, pca_percVar=0.98, subSample=1,
                        onOffThr=None, scaler_withMean=False):
    """
    Given a tail angles array returns fitted scaler, pca and gmm models
    resulting from the specified input parameters
    Parameters
    ----------
    ta: array, (nTailPoints, nTimePoints)
        Tail angles array
    nKer: int
        Kernel length in points for convolution of features
    pca_percVar: scalar or None
        Percent of variance (0 to 1) to be explained by PCA in reducing feature
        dimensions. If None, skips PCA.
    onOffThr: scalar
        Threshold to use on swim envelopes in determining suprathreshold
        activity.
        This is used to feed only the suprathreshold activity to PCA and GMM
        for speeding and other considerations.
    subSample: int
        Sub-sampling interval for the feature array. Used to reduce number
        of samples fed into PCA and GMM for purpose of computational speed.
    scaler_withMean: boolean
        If False, then in fitting the scaler, 0 is treated as the mean.
    Returns
    -------
    fitter: dict
        Dictionary with the following keys
        scaler, pca, gmm: sklearn objects
            Fitted StandardScaler, PCA, and GMM model objects.
        nKer: int
            Kernel length for convolving feature array during preprocessing.
    """

    df_features = hf.swimEnvelopes_multiLoc(ta)
    arr_feat = np.array(df_features)

    scaler = StandardScaler(with_mean=scaler_withMean).fit(arr_feat)
    arr_feat_scaled = scaler.transform(arr_feat)
    if nKer is not None:
        arr_feat_scaled =\
            [dask.delayed(spt.causalConvWithSemiGauss1d)(x, nKer*2)
             for x in arr_feat_scaled.T]
        arr_feat_scaled = np.array(dask.compute(*arr_feat_scaled)).T

    if onOffThr is not None:
        env_max = spt.emd.envelopesAndImf(ta[-1])['env']['max']
        inds_supraThresh = np.where(env_max > onOffThr)[0]
        arr_supra = arr_feat_scaled[inds_supraThresh, :]
    else:
        arr_supra = arr_feat_scaled

    X = arr_supra[::subSample, :]

    if pca_percVar is not None:
        print('Performing PCA to reduce feature dimensions...')
        pca = PCA(n_components=pca_percVar, random_state=143)
        X_pca = pca.fit_transform(X)
        print(f'Reduced features to {pca.n_components_} dimensions')
    else:
        pca = None
        X_pca = X

    print('Fitting GMM...')
    gmm = GMM(n_components=n_gmm, covariance_type='full').fit(X_pca)
    fitter = dict(gmm=gmm, pca=pca, scaler=scaler, nKer=nKer)
    return fitter


def train_on_tailAngles_svd(ta, n_gmm=15, n_svd=3, subSample=1,
                            scaler_withMean=False, use_envelopes=True):
    """
    Given a tail angles array returns fitted scaler, pca and gmm models
    resulting from the specified input parameters
    Parameters
    ----------
    ta: array, (nTailPoints, nTimePoints)
        Tail angles array
    n_svd: int
        Number of svd components
    subSample: int
        Sub-sampling interval for the feature array. Used to reduce number
        of samples fed into PCA and GMM for purpose of computational speed.
    scaler_withMean: boolean
        If False, then in fitting the scaler, 0 is treated as the mean.
    n_gmm: int
        Number of GMM components
    Returns
    -------
    fitter: dict
        Dictionary with the following keys
        scaler, svd, gmm: sklearn objects
            Fitted StandardScaler, SVD, and GMM model objects.
    """
    X, svd = tailAngles_to_svd_featureArray(ta, n_svd=n_svd,
                                            use_envelopes=use_envelopes)

    scaler = StandardScaler(with_mean=scaler_withMean).fit(X)
    X_scaled = scaler.transform(X)

    X_svd = X_scaled[::subSample, :]

    print('Fitting GMM...')
    gmm = GMM(n_components=n_gmm, covariance_type='full').fit(X_svd)
    fitter = dict(gmm=gmm, svd=svd, scaler=scaler,
                  use_envelopes=use_envelopes)
    return fitter
