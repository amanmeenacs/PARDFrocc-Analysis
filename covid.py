import numpy as np
import os
import pandas as pd
import sys
import sklearn
import scipy.sparse as sp
import sklearn.datasets as skds
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler, Normalizer

def covid(majority=["tested_negative"], frac=1, scaled=True, normalized=True):
    import sklearn.datasets as skds

    datadir = os.path.expanduser(f"~/data5")

    if os.path.isdir(datadir):

        x_test = np.load(os.path.join(datadir, "x_test.npy"),allow_pickle=True)
        x_val = np.load(os.path.join(datadir, "x_val.npy"),allow_pickle=True)
        y_train = np.load(os.path.join(datadir, "y_train.npy"),allow_pickle=True)
        y_test = np.load(os.path.join(datadir, "y_test.npy"),allow_pickle=True)
        y_val = np.load(os.path.join(datadir, "y_val.npy"),allow_pickle=True)
        x_train = np.load(os.path.join(datadir, "x_train.npy"))

    else:
        # print(22)
        x,y = skds.fetch_openml(data_id=1504, return_X_y=True)
        # print(x)
        categories = np.array([0, 1])

        y.replace(['1', '2'],[0, 1], inplace=True)
        for i, _y in enumerate(y):
            if _y not in majority:
                y[i] = 0
            else:
                y[i] = 1
        y = y.astype("int")
        x_train, x_test, y_train, y_test = tts(x, y, test_size=0.4)
        # print(4)
        x_val, x_test, y_val, y_test = tts(x_test, y_test, test_size=0.5)
        # print(5)


        if scaled:
            scalar = StandardScaler().fit(x_train)
            x_train = scalar.transform(x_train)
            x_val = scalar.transform(x_val)
            x_test = scalar.transform(x_test)
        # print(6)
        if normalized:
            normalizer = Normalizer().fit(x_train)
            x_train = normalizer.transform(x_train)
            x_val = normalizer.transform(x_val)
            x_test = normalizer.transform(x_test)
        # print(7)
        os.mkdir(datadir)
        np.save(os.path.join(datadir, "x_train"), x_train)
        np.save(os.path.join(datadir, "x_val"), x_val)
        np.save(os.path.join(datadir, "x_test"), x_test)
        np.save(os.path.join(datadir, "y_train"), y_train)
        np.save(os.path.join(datadir, "y_val"), y_val)
        np.save(os.path.join(datadir, "y_test"), y_test)

        print(8)

    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = covid(majority=[0], frac=1, scaled=True, normalized=True)