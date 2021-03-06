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
import matplotlib.pyplot as plt
sys.path.append(r'v:/code/python/code')
import apCode.SignalProcessingTools as spt  # noqa: E402
from apCode.machineLearning.preprocessing import Scaler  # noqa: E402


def max_min_envelopes(x):
    if np.ndim(x) == 1:
        x_env = spt.emd.envelopesAndImf(x)['env']
        x_env = np.c_[x_env['crests'], x_env['troughs']].T
    else:
        x_env = [dask.delayed(max_min_envelopes)(x_) for x_ in x]
        x_env = np.array(dask.compute(*x_env))
        x_env = np.concatenate(x_env, axis=0)
    return x_env


class SvdGmm(object):
    def __init__(self, n_gmm=20, n_svd=3, svd=None, use_envelopes=True,
                 scaler_withMean=False, pca_percVar=None, pk_thr=5,
                 random_state=143, covariance_type='full', **gmm_kwargs):
        """Class for fiting Gaussian Mixture Model on SVD-based features
        extracted from tail angles.
        Parameters
        ----------
        n_gmm: int
            Number of Gaussian Mixture components to use
        n_svd: int
            Number of svd components to use in representing tail angles.
            Empirically, 3 is a good value because it explains ~95% of
            variance in the data.
        svd: Scikit-learn's TruncatedSVD class or None:
            If None, then initializes a naive SVD object using n_svd
            components, otherise uses in the provided SVD oject to fit data,
            in which case, n_svd is ignored. No the SVD object must be
            fit to pre-fit.
        use_envelopes: bool
            If True, uses the envelopes (crests and troughs) of the SVD
            component timeseries for generating features.
        scaler_withMean: bool
            If True, computes the mean for the SVD-based features and uses this
            when scaling, else uses 0 as the mean and scales only using the
            standard deviation.
        pca_percVar: float or None
            If None, then does not perform PCA on SVD-based features to reduce
            dimensionality. If float, then uses as many PCA compoments as will
            explain pca_percVar*100 percent of the total variance.
        pk_thr: scalar or None
            If not None, then uses this value as threshold for detecting
            peaks in the total tail angles and then only fits the GMM to
            these values. If None, then computes the GMM using all time points
        random_state: int
            Random state of the RNGs.
        """
        self.n_gmm_ = n_gmm
        self.n_svd_ = n_svd
        self.svd = svd
        self.use_envelopes_ = use_envelopes
        self.scaler_withMean_ = scaler_withMean
        self.pca_percVar_ = pca_percVar
        self.pk_thr_ = pk_thr
        self.random_state_ = random_state
        self.covariance_type_ = covariance_type
        self.gmm_kwargs_ = gmm_kwargs

    def fit(self, ta):
        """Fit model to tail angles. This includes preprocessing wherein
        SVD-based feature extraction is performed, followed by PCA for
        dimensionality reduction, if specfied.
        Parameters
        ----------
        self: object
            Instance of initiated SvdGmm class
        ta: array, (nPointsAlongTail, nTimePoints)
            Tail angles array
        Returns
        -------
        self: object
            Trained SvdGmm model.
        """
        if self.svd is None:
            svd = TruncatedSVD(n_components=self.n_svd_,
                               random_state=self.random_state_).fit(ta.T)
        else:
            svd = self.svd
        V = svd.transform(ta.T)
        dv = np.gradient(V)[0]
        ddv = np.gradient(dv)[0]
        X = np.c_[V, dv, ddv]
        if self.use_envelopes_:
            features = max_min_envelopes(X.T).T
        scaler = StandardScaler(with_mean=self.scaler_withMean_).fit(features)
        features = scaler.transform(features)
        if self.pk_thr_ is not None:
            y = ta[-1]
            pks = spt.findPeaks(y, thr=self.pk_thr_, thrType='rel',
                                pol=0)[0]
            print(f'Peaks are {round(100*len(pks)/len(y), 1)}% of all samples')
            features = features[pks, :]
        if self.pca_percVar_ is not None:
            pca = PCA(n_components=self.pca_percVar_,
                      random_state=self.random_state_).fit(features)
            features = pca.transform(features)
            pca.n_components_ = features.shape[1]
        else:
            pca = None
        print('Fitting GMM..')
        gmm = GMM(n_components=self.n_gmm_, random_state=self.random_state_,
                  covariance_type=self.covariance_type_, **self.gmm_kwargs_)
        gmm = gmm.fit(features)
        self.svd = svd
        self.scaler = scaler
        self.pca = pca
        self.gmm = gmm
        return self

    def to_features(self, ta):
        """
        Given tail angles array, returns svd-based feature array as well as
        the svd object. The feature array can be used for posture
        classification by GMM or some other clustering algorithm. Must be
        run after fit because it requires a fit SVD object
        Parameters
        ----------
        ta: array, (nPointsAlongTail, nTimePoints)
            Tail angles array
        Returns
        -------
        features: array, (nTimePoints, n_svd*(nDerivatives+1=3)*(nEnvelopes=2))
            SVD-based feature array. In addition to the timeseries of the n_svd
            components, this array includes upto the 2nd derivative of these
            timeseries.
        """
        V = self.svd.transform(ta.T)
        dv = np.gradient(V)[0]
        ddv = np.gradient(dv)[0]
        features = np.c_[V, dv, ddv]
        if self.use_envelopes_:
            features = max_min_envelopes(features.T).T
        features = self.scaler.transform(features)
        if self.pca is not None:
            features = self.pca.transform(features)
        return features

    def plot_with_labels(self, ta, cmap='tab20', figSize=(60, 10),
                         marker_size=100, line_alpha=0.2):
        """Plot tail angles overlaid with labels in different colors
        and numbers as markers.
        Parameters
        ----------
        ta: array, (nPointsAlongFish, nTimePoints)
            Tail angles
        cmap: str or matplotlib.colors.ListedColormap
            Colormap for plotting marker classes
        figSize: tuple, (2,)
            Figure size
        marker_size: scalar
            Marker size
        line_alpha: scalar
            Alpha value for lineplot of total tail angle timeseseries
        Returns
        --------
        fh: object
            Figure handle
        """
        if isinstance(cmap, str):
            cmap = eval(f'plt.cm.{cmap}')
        labels, features = self.predict(ta)
        labels_all = np.arange(self.n_gmm_)
        scaler = Scaler(standardize=True).fit(labels_all)
        labels_norm = scaler.transform(labels)
        clrs = cmap(labels_norm)
        fh = plt.figure(figsize=figSize)
        x = np.arange(ta.shape[1])
        plt.plot(ta[-1], c='k', alpha=line_alpha)
        for lbl in np.unique(labels):
            inds = np.where(labels == lbl)[0]
            clrs_now = clrs[inds].reshape(-1, 4)
            plt.scatter(x[inds], ta[-1][inds], c=clrs_now, s=marker_size,
                        marker=f"${str(lbl)}$")
        return fh

    def plot_with_labels_interact(self, ta, x=None, cmap='tab20',
                                  figsize=(20, 10), marker_size=10,
                                  line_alpha=0.2, ylim=(-150, 150),
                                  title='Tail angles with GMM labels'):

        import plotly.graph_objs as go
        if isinstance(cmap, str):
            cmap = eval(f'plt.cm.{cmap}')
        labels, features = self.predict(ta)
        if x is None:
            x = np.arange(ta.shape[1])
        y = ta[-1]
        if self.pk_thr_ is not None:
            pks = spt.findPeaks(y, thr=self.pk_thr_, pol=0, thrType='rel')[0]
        else:
            pks = np.arange(len(y))

        line = go.Scatter(x=x, y=y, mode = 'lines', opacity = line_alpha,
                          marker = dict(color='black'), name='ta')
        scatters = []
        scatters.append(line)
        for iLbl, lbl in enumerate(np.unique(labels)):
            clr = f'rgba{cmap(lbl/self.n_gmm_)}'
            inds= np.where(labels==lbl)[0]
            inds = np.intersect1d(inds, pks)
            scatter = go.Scatter(x=x[inds], y=y[inds], mode='markers',
                                 marker=dict(color=clr, symbol=lbl,
                                             size=marker_size),
                                 name=f'Lbl-{lbl}')
            scatters.append(scatter)
        fig = go.Figure(scatters)
        if ylim is not None:
            ylim = np.array(ylim)
            ylim[0] = np.minimum(ylim[0], y.min())
            ylim[1] = np.maximum(ylim[1], y.max())
            fig.layout.yaxis.range = ylim
        fig.layout.xaxis.range = [x[0], x[-1]]
        fig.update_layout(title=title)
        # fig.show()
        # figName = f'Fig-{util.timestamp()}_trl-{iTrl}.html'
        # fig.write_html(os.path.join(figDir,figName))
        return fig

    def predict(self, ta):
        """Use trained SvdGmm model to predict labels on an array of tail
         angles.
         Parameters
         ----------
         self: Trained SvdGMM model object
         ta: array, (nPointsAlongTail, nTimePoints)
            Tail angles
        Returns
        --------
        labels: array, (nTimePoints,)
            Predicted labels.
        features: array, (nTimePoints, nFeatures)
            Feature array
        """
        features = self.to_features(ta)
        return self.gmm.predict(features), features


def _plot_with_labels(model, ta, cmap='tab20', figSize=(60, 10),
                      marker_size=100, line_alpha=0.2):
    """Plot tail angles overlaid with labels in different colors
    and numbers as markers.
    Parameters
    ----------
    ta: array, (nPointsAlongFish, nTimePoints)
        Tail angles
    cmap: str or matplotlib.colors.ListedColormap
        Colormap for plotting marker classes
    figSize: tuple, (2,)
        Figure size
    marker_size: scalar
        Marker size
    line_alpha: scalar
        Alpha value for lineplot of total tail angle timeseseries
    Returns
    --------
    fh: object
        Figure handle
    """
    if isinstance(cmap, str):
        cmap = eval(f'plt.cm.{cmap}')
    labels, features = model.predict(ta)
    labels_all = np.arange(model.n_gmm_)
    scaler = Scaler(standardize=True).fit(labels_all)
    labels_norm = scaler.transform(labels)
    clrs = cmap(labels_norm)
    fh = plt.figure(figsize=figSize)
    x = np.arange(ta.shape[1])
    plt.plot(ta[-1], c='k', alpha=line_alpha)
    for lbl in np.unique(labels):
        inds = np.where(labels == lbl)[0]
        clrs_now = clrs[inds].reshape(-1, 4)
        plt.scatter(x[inds], ta[-1][inds], c=clrs_now, s=marker_size,
                    marker=f"${str(lbl)}$")
    return fh
