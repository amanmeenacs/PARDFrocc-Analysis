import numpy as np
import os
import pandas as pd
import sys
import sklearn
import scipy.sparse as sp
import sklearn.datasets as skds
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler, Normalizer

def blood(majority=["tested_negative"], frac=1, scaled=True, normalized=True):
    import sklearn.datasets as skds

    datadir = os.path.expanduser(f"~/data1")

    if os.path.isdir(datadir):

        x_test = np.load(os.path.join(datadir, "x_test.npy"),allow_pickle=True)
        x_val = np.load(os.path.join(datadir, "x_val.npy"),allow_pickle=True)
        y_train = np.load(os.path.join(datadir, "y_train.npy"),allow_pickle=True)
        y_train2 = np.load(os.path.join(datadir, "y_train2.npy"), allow_pickle=True)
        y_test = np.load(os.path.join(datadir, "y_test.npy"),allow_pickle=True)
        y_val = np.load(os.path.join(datadir, "y_val.npy"),allow_pickle=True)
        x_train = np.load(os.path.join(datadir, "x_train.npy"))
        x_train2 = np.load(os.path.join(datadir, "x_train2.npy"))

    else:
        # print(22)
        x,y = skds.fetch_openml(data_id=1464, return_X_y=True)
        # print(x)
        y.replace(['1','2'],[0,1], inplace=True)
        x_train, x_test, y_train, y_test = tts(x, y, test_size=0.4)
        x_train, x_train2, y_train, y_train2 = tts(x, y, test_size=0.5)
        x_val, x_test, y_val, y_test = tts(x_test, y_test, test_size=0.6)
        # print(5)
        if scaled:
            scalar = StandardScaler().fit(x_train)
            x_train = scalar.transform(x_train)
            x_train2 = scalar.transform(x_train2)
            x_val = scalar.transform(x_val)
            x_test = scalar.transform(x_test)
        # print(6)
        if normalized:
            normalizer = Normalizer().fit(x_train)
            x_train = normalizer.transform(x_train)
            x_train2 = normalizer.transform(x_train2)
            x_val = normalizer.transform(x_val)
            x_test = normalizer.transform(x_test)
        # print(7)
        os.mkdir(datadir)
        np.save(os.path.join(datadir, "x_train"), x_train)
        np.save(os.path.join(datadir, "x_train2"), x_train)
        np.save(os.path.join(datadir, "x_val"), x_val)
        np.save(os.path.join(datadir, "x_test"), x_test)
        np.save(os.path.join(datadir, "y_train"), y_train)
        np.save(os.path.join(datadir, "y_train2"), y_train)
        np.save(os.path.join(datadir, "y_val"), y_val)
        np.save(os.path.join(datadir, "y_test"), y_test)
    return x_train, y_train, x_train2, y_train2, x_val, y_val, x_test, y_test

x_train, y_train, x_train2, y_train2, x_val, y_val, x_test, y_test = blood(majority=[0], frac=1, scaled=True, normalized=True)