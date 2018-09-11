import matplotlib.pyplot as plt

import numpy as np
from glob import glob
import os
from scipy.signal import resample
from collections import Counter

import random

from pyorbital import tlefile

# from keras.models import Sequential, Model
# from keras.layers import Dense, Dropout
# from keras.utils import to_categorical
# from keras.optimizers import SGD, Adam
# from keras import regularizers
# from keras.callbacks import EarlyStopping
# from keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

import pandas as pd
import seaborn as sns

#################################################################
# GENERAL UTILS
#################################################################

def visualize_embedding(X, y, y_names=None):

    # Fit each embedding
    pca_embedding = PCA(n_components=2).fit_transform(X)
    tsne_embedding = TSNE(n_components=2).fit_transform(X)
    umap_embedding = umap.UMAP().fit_transform(X)

    # Choose a color scheme
    colors = ['red', 'blue', 'yellow', 'purple', 'green', 'pink'] * 10
    colors = [ colors[ndx] for ndx in y ]

    # Plot the Embeddings
    embeddings = [
        ('TSNE', tsne_embedding),
        ('UMAP', umap_embedding),
        ('PCA',  pca_embedding)
    ]
    fig, axes = plt.subplots(ncols=len(embeddings), figsize=(20,7))
    for ndx, (title, xy) in enumerate(embeddings):
        ax = axes[ndx]
        ax.scatter(xy[:,0], xy[:,1], c=colors, alpha=0.3, s=50)
        ax.set_title(title)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.tight_layout()

def normalize_features(X):
    return (X - np.mean(X, keepdims=True)) / np.std(X, keepdims=True)

def plot_confusion_matrix(model,X,y,title,names):
    yhat = model.predict(normalize_features(X))
    cm = confusion_matrix(y, np.argmax(yhat, axis=1))
    a = pd.DataFrame(cm)
    a.columns = names
    a.index = names
    acc = np.trace(cm)/np.sum(cm)
    title_with_acc = '{} (acc: {:.3f})'.format(title, acc)
    sns.heatmap(a, cmap=None, annot=True, linewidths=.1).set_title(title_with_acc)

#################################################################
# TLE DATA
#################################################################

def load_tle_data(data_dir):
    data = []
    for tle_fn in glob(os.path.join(data_dir, '*.txt')):
        group_name, _ = os.path.splitext(os.path.basename(tle_fn))
        with open(tle_fn, 'r') as fh:
            while True:
                try:
                    platform = next(fh)
                    line1 = next(fh)
                    line2 = next(fh)
                    x = tlefile.read(platform, line1=line1, line2=line2)
                    data.append({
                        'group' : group_name,
                        'arg_perigee' : x.arg_perigee,
                        'bstar' : x.bstar,
                        'excentricity' : x.excentricity,
                        'inclination' : x.inclination,
                        'mean_anomaly' : x.mean_anomaly,
                        'mean_motion' : x.mean_motion,
                        'mean_motion_derivative' : x.mean_motion_derivative,
                        #'mean_motion_sec_derivative' : x.mean_motion_sec_derivative,
                        'orbit' : x.orbit,
                        'right_ascension' : x.right_ascension
                    })
                except StopIteration:
                    break       
    df = pd.DataFrame(data)
    return df

#####################################################################
# ALCDEF DATA
#####################################################################

def resample_light_curve(timestamps, intensities, nb_samples=100):
    '''Resample light curve to a given number of samples.'''
    r_intensities, r_timestamps = resample(intensities, num=nb_samples, t=timestamps)
    return r_timestamps, r_intensities

def to_float(v):
    try:
        return float(v)
    except:
        return None

def load_alcdef_data(data_dir, min_samples=1, resample_to=100, reduce_to_top=None):
    fns = glob(os.path.join(data_dir, '*.txt'))
    data = []
    for item in parse_alcdef_files(fns):
        if len(item['DATA']) < min_samples:
            continue
        # Resample item before moving on
        intensities = item['DATA'][:,1]
        timestamps = item['DATA'][:,0]
        r_timestamps, r_intensities = resample_light_curve(timestamps, intensities, nb_samples=min_samples)
        item['DATA_RESAMPLED'] = np.array([r_timestamps, r_intensities]).T
        data.append(item)

    if reduce_to_top is not None:
        c = Counter([item['OBJECTNAME'] for item in data])
        names = [ name for name, cnt in c.most_common(reduce_to_top) ]
        data = [ item for item in data if item['OBJECTNAME'] in names ]

    return data
    
def parse_alcdef_files(fns):
    all_objects = []
    item = None
    for fn in fns:
        with open(fn, 'r') as fh:
            item = {}
            for line in fh.readlines():
                line = line.strip()
                if line == 'ENDDATA':
                    item['DATA'] = np.array(item['DATA'])
                    yield item
                elif line == 'STARTMETADATA':
                    item = {}
                elif line == 'ENDMETADATA':
                    item['DATA'] = []
                elif line.startswith('DATA='):
                    values = line.strip().split('=')[1].split('|')
                    values = list(map(to_float, values))
                    item['DATA'].append(values)
                else:
                    try:
                        split_ndx = line.index('=')
                        k = line[:split_ndx]
                        v = line[split_ndx+1:]
                    except:
                        print(line)
                        raise
                    item[k] = v

def plot_alcdef_examples(data, nrows=6, ncols=8):

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(2*ncols,2*nrows))
    items = []

    # Collect the examples to display so that the same object gets displayed across a row
    object_names = list(set([ item['OBJECTNAME'] for item in data ]))
    assert nrows <= len(object_names), 'You cannot displace more rows than you have distinct object names'
    object_names = np.random.choice(object_names, size=nrows, replace=False)
    for object_name in object_names:
        examples = [ item for item in data if item['OBJECTNAME'] == object_name ]
        examples = np.random.choice(examples, size=ncols, replace=False)
        items.extend(examples)

    for ndx, item in enumerate(items):
        r = ndx//8
        c = ndx%8
        ax = axes[r,c]
        x = item['DATA_RESAMPLED'][:,0]
        y = item['DATA_RESAMPLED'][:,1]
        ax.scatter(x=x,y=y)
        ax.set_title('{} ({})'.format(item['OBJECTNAME'], item['MAGBAND']))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.axis('off')
    plt.tight_layout()    