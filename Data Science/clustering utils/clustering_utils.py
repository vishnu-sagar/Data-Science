import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)

import os
import pandas as pd
import seaborn as sns
import numpy as np
import math
from itertools import product, cycle
from sklearn import covariance, preprocessing, tree, svm, neighbors, metrics, linear_model, manifold, linear_model
from sklearn_pandas import DataFrameMapper,CategoricalImputer
from sklearn import model_selection, ensemble, preprocessing, decomposition, feature_selection
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles, make_moons, make_classification,make_blobs
import matplotlib.cm as cm
from classification_utils import *


def grid_search_plot_one_parameter_curves_clustering(estimator, grid, X, scoring):
    name = str(estimator)
    items = sorted(grid.items())
    keys, values = zip(*items)
    params = []
    scores = []
    for v in product(*values):
        params.append(dict(zip(keys, v)))
    for param in params:
        estimator.set_params(**param)
        if(name.startswith('Gaussian')):
            labels = estimator.fit_predict(X)
        else:
            estimator.fit(X)
            labels = estimator.labels_
            
        if scoring == 's_score':
            score = metrics.silhouette_score(X, labels, metric='euclidean')
        elif scoring == 'ch_score':
            score =  metrics.calinski_harabaz_score(X, labels)
        else:
            print(str(scoring) +' metric not supported')
            return
        scores.append(score)
    
    plt.figure()
    plt.plot(params, scores, marker="D")
    plt.xlabel('nclusters')
    plt.ylabel(str(scoring))

def grid_search_best_model_clustering(estimator, grid, X, scoring):
    name = str(estimator)
    items = sorted(grid.items())
    keys, values = zip(*items)
    params =[]
    for v in product(*values):
        params.append(dict(zip(keys, v)))
    n = len(params)
    best_param = None
    best_score = 0.0
    for param in params:
        estimator.set_params(**param)
        if(name.startswith('Gaussian')):
            labels = estimator.fit_predict(X)
        else:
            estimator.fit(X)
            labels = estimator.labels_
        if scoring == 's_score':
            score = metrics.silhouette_score(X, labels, metric='euclidean')
        elif scoring == 'ch_score':
            score =  metrics.calinski_harabaz_score(X, labels)
        else:
            print(scoring+' metric not supported')
            break
        if score > best_score :
            best_score = score
            best_param = param
    if best_param is not None:
        estimator.set_params(**best_param)
        estimator.fit(X)
        print("Best score:" + str(best_score))
        return estimator
    else:
        return None

def grid_search_plot_models_2d_clustering(estimator, grid, X, xlim=None, ylim=None):
    plt.style.use('seaborn')
    items = sorted(grid.items())
    keys, values = zip(*items)
    params =[]
    for v in product(*values):
        params.append(dict(zip(keys, v)))
    n = len(params)
    fig, axes = plt.subplots(int(math.sqrt(n)), math.ceil(math.sqrt(n)), figsize=(20, 20), dpi=80)
    axes = np.array(axes)
    for ax, param in zip(axes.reshape(-1), params):
        estimator.set_params(**param)
        estimator.fit(X)  
        plot_model_2d_clustering(estimator, X, ax, xlim, ylim, str(param), False)
    plt.tight_layout()

def grid_search_plot_models_3d_clustering(estimator, grid, X, xlim=None, ylim=None, zlim=None):
    plt.style.use('seaborn')
    items = sorted(grid.items())
    keys, values = zip(*items)
    params =[]
    for v in product(*values):
        params.append(dict(zip(keys, v)))
    n = len(params)
    fig, axes = plt.subplots(int(math.sqrt(n)), math.ceil(math.sqrt(n)), figsize=(20, 20), dpi=80, subplot_kw=dict(projection='3d') )
    axes = np.array(axes)
    for ax, param in zip(axes.reshape(-1), params):
        estimator.set_params(**param)
        estimator.fit(X)  
        plot_model_3d_clustering(estimator, X, ax, xlim, ylim, zlim, str(param), False)
    plt.tight_layout()

def plot_model_3d_clustering(estimator, X, ax = None, xlim=None, ylim=None, zlim=None, title=None, new_window=True, rotation=False):
    name = str(estimator)
    if(name.startswith('Gaussian')):
        y = estimator.fit_predict(X)
    else:
        y = estimator.labels_
    
    ax = plot_data_3d_classification(X, y, ax, xlim, ylim, zlim, title, new_window, rotation)
    if hasattr(estimator, 'cluster_centers_'):
        centers = estimator.cluster_centers_
        plot_data_3d(centers, ax, new_window=False, title=title, s=200)

def plot_model_2d_clustering(estimator, X, ax = None, xlim=None, ylim=None, title=None, new_window=True):
    name = str(estimator)
    if(name.startswith('Gaussian')):
        y = estimator.fit_predict(X)
    else:
        y = estimator.labels_
    
    ax = plot_data_2d_classification(X, y, ax, xlim, ylim, title, new_window)
    if hasattr(estimator, 'cluster_centers_'):
        centers = estimator.cluster_centers_
        plot_data_2d(centers, ax, new_window=False, title = title, s=200)


def generate_synthetic_data_2d_clusters(n_samples, n_centers, cluster_std) :
    return make_blobs(n_samples=n_samples, centers=n_centers,
                       cluster_std=cluster_std, random_state=0)

def generate_synthetic_data_3d_clusters(n_samples, n_centers, cluster_std) :
    return make_classification(n_samples = n_samples,
                                       n_features = 3,
                                       n_informative = 3,
                                       n_clusters_per_class=1,
                                       n_redundant = 0,
                                       n_classes = n_centers,
                                       random_state=100)