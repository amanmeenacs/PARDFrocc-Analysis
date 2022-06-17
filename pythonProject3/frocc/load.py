# https://www.cs.toronto.edu/~kriz/cifar.html
import pardfrocc
import sys
from steel import x_train,x_train2,y_train2, y_train, x_val, y_val, x_test, y_test
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score, precision_score, recall_score, f1_score, accuracy_score
import os
import sklearn
from timeit import default_timer as timer
import scipy.sparse as sp
import sklearn.datasets as skds
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler, Normalizer

# def mnist(majority=["0"], frac=1, scaled=True, normalized=True):
#     import sklearn.datasets as skds
#
#     datadir = os.path.expanduser(f"~/data")
#
#     if os.path.isdir(datadir):
#         x_train = np.load(os.path.join(datadir, "x_train.npy"),allow_pickle=True)
#         x_test = np.load(os.path.join(datadir, "x_test.npy"),allow_pickle=True)
#         x_val = np.load(os.path.join(datadir, "x_val.npy"),allow_pickle=True)
#         y_train = np.load(os.path.join(datadir, "y_train.npy"),allow_pickle=True)
#         y_test = np.load(os.path.join(datadir, "y_test.npy"),allow_pickle=True)
#         y_val = np.load(os.path.join(datadir, "y_val.npy"),allow_pickle=True)
#         # print(x_train)
#     else:
#         # print(22)
#         x, y = skds.fetch_openml("mnist_784", return_X_y=True)
#         # print(x)
#
#         # for i, _y in enumerate(y):
#         #     if _y not in majority:
#         #         y[i] = 0
#         #     else:
#         #         y[i] = 1
#         # y = y.astype("int")
#         # print(3)
#         x_train, x_test, y_train, y_test = tts(x, y, test_size=0.4)
#         # print(4)
#         x_val, x_test, y_val, y_test = tts(x_test, y_test, test_size=0.5)
#         # print(5)
#
#         if scaled:
#             scalar = StandardScaler().fit(x_train)
#             x_train = scalar.transform(x_train)
#             x_val = scalar.transform(x_val)
#             x_test = scalar.transform(x_test)
#         # print(6)
#         if normalized:
#             normalizer = Normalizer().fit(x_train)
#             x_train = normalizer.transform(x_train)
#             x_val = normalizer.transform(x_val)
#             x_test = normalizer.transform(x_test)
#         # print(7)
#
#         os.mkdir(datadir)
#         np.save(os.path.join(datadir, "x_train"), x_train)
#         np.save(os.path.join(datadir, "x_val"), x_val)
#         np.save(os.path.join(datadir, "x_test"), x_test)
#         np.save(os.path.join(datadir, "y_train"), y_train)
#         np.save(os.path.join(datadir, "y_val"), y_val)
#         np.save(os.path.join(datadir, "y_test"), y_test)
#         print(8)
#     return x_train, y_train, x_val, y_val, x_test, y_test


# def cifar(majority=["0"], frac=1, scaled=True, normalized=True):
#     import sklearn.datasets as skds
#
#     datadir = os.path.expanduser(f"~/data")
#
#     if os.path.isdir(datadir):
#         x_train = np.load(os.path.join(datadir, "x_train.npy"),allow_pickle=True)
#         x_test = np.load(os.path.join(datadir, "x_test.npy"),allow_pickle=True)
#         x_val = np.load(os.path.join(datadir, "x_val.npy"),allow_pickle=True)
#         y_train = np.load(os.path.join(datadir, "y_train.npy"),allow_pickle=True)
#         y_test = np.load(os.path.join(datadir, "y_test.npy"),allow_pickle=True)
#         y_val = np.load(os.path.join(datadir, "y_val.npy"),allow_pickle=True)
#     else:
#         x, y = skds.fetch_openml("cifar_10", return_X_y=True)
#
#         # for i, _y in enumerate(y):
#         #     if _y not in majority:
#         #         y[i] = 0
#         #     else:
#         #         y[i] = 1
#         # y = y.astype("int")
#
#         x_train, x_test, y_train, y_test = tts(x, y, test_size=0.4)
#         x_val, x_test, y_val, y_test = tts(x_test, y_test, test_size=0.5)
#
#         if scaled:
#             scalar = StandardScaler().fit(x_train)
#             x_train = scalar.transform(x_train)
#             x_val = scalar.transform(x_val)
#             x_test = scalar.transform(x_test)
#         if normalized:
#             normalizer = Normalizer().fit(x_train)
#             x_train = normalizer.transform(x_train)
#             x_val = normalizer.transform(x_val)
#             x_test = normalizer.transform(x_test)
#
#         os.mkdir(datadir)
#         np.save(os.path.join(datadir, "x_train"), x_train)
#         np.save(os.path.join(datadir, "x_val"), x_val)
#         np.save(os.path.join(datadir, "x_test"), x_test)
#         np.save(os.path.join(datadir, "y_train"), y_train)
#         np.save(os.path.join(datadir, "y_val"), y_val)
#         np.save(os.path.join(datadir, "y_test"), y_test)
#
#     return x_train, y_train, x_val, y_val, x_test, y_test
if __name__ == '__main__':
    # x_train, y_train, x_val, y_val, x_test, y_test = data_gen.himoon(n_samples=1000, n_dims=1000) #or data_gen.mmgauss()
    # print(x_train,x_test,y_test)
    # x_test = sp.csc_matrix(x_test, dtype=np.float32)
    # x_train = sp.csc_ffffmatrix(x_train, dtype=np.float32)
    # y_test = sp.csc_matrix(y_test, dtype=np.float32)
    # print(x_test)
    # x_train, y_train, x_val, y_val, x_test, y_test = cifar(majority=["0"], frac=1, scaled=True, normalized=True)


    # np.set_printoptions(threshold=sys.maxsize)
    t1,t2 = x_train,y_train
    a,b,c =0,0,0
    prc = 0
    f1 = 0
    i = 0
    k = np.count_nonzero(y_train == 1)
    print( x_train.shape)
    x_train_new = np.zeros(shape=(k,54))
    y_train_new = np.zeros(shape=(k))
    k = 0
    # print(y_train)
    for i in range(len(y_train)):
        if y_train[i] == 1:
            x_train_new[k] = x_train[i]
            y_train_new[k] = 1
            k += 1
    x_train = x_train_new
    y_train = y_train_new
    x_test = sp.csc_matrix(x_test, dtype=np.float32)
    x_train = sp.csc_matrix(x_train, dtype=np.float32)
    clf = pardfrocc.ParDFROCC(epsilon=0.1)
    # print(x_train)
    clf.fit(x_train)
    # print(x_test.shape,y_test.shape)
    scores = clf.decision_function(x_test)
    predictions = clf.predict(x_test)
    roc = roc_auc_score(y_test, scores)
    prc = precision_score(y_test,predictions,pos_label=0)
    f1 = f1_score(y_test, predictions,pos_label=0)
    a+=roc
    b+=prc
    c+=f1
    print("Sir,wait a minute Please")
    j= "{:.3f}".format(roc)
    k2 = "{:.3f}".format(prc)
    l = "{:.3f}".format(f1)


    prc = 0
    f1 = 0
    i = 0
    k = np.count_nonzero(y_train2 == 1)
    x_train_new = np.zeros(shape=(k,54))
    y_train_new = np.zeros(shape=(k))
    k = 0
    # print(y_train)
    for i in range(len(y_train2)):
        if y_train2[i] == 1:
            x_train_new[k] = x_train2[i]
            y_train_new[k] = 1
            k += 1
    x_train2 = x_train_new
    y_train2 = y_train_new
    x_test = sp.csc_matrix(x_test, dtype=np.float32)
    x_train2 = sp.csc_matrix(x_train2, dtype=np.float32)
    clf = pardfrocc.ParDFROCC(epsilon=0.1)
    # print(x_train)
    clf.fit(x_train2)
    # print(x_test.shape,y_test.shape)
    scores = clf.decision_function(x_test)
    predictions = clf.predict(x_test)
    roc = roc_auc_score(y_test, scores)
    prc = precision_score(y_test,predictions,pos_label=0)
    f1 = f1_score(y_test, predictions,pos_label=0)
    a += roc
    b += prc
    c += f1
    p = "{:.3f}".format(roc)
    q = "{:.3f}".format(prc)
    r = "{:.3f}".format(f1)


    prc = 0
    f1 = 0
    i = 0
    k = np.count_nonzero(y_train == 1)
    x_train_new = np.zeros(shape=(k,54))
    y_train_new = np.zeros(shape=(k))
    x_train=t1
    y_train=t2
    k = 0
    # print(y_train)
    for i in range(len(y_train)):
        if y_train[i] == 1:
            x_train_new[k] = x_train[i]
            y_train_new[k] = 1
            k += 1
    x_train = x_train_new
    y_train = y_train_new
    x_val = sp.csc_matrix(x_val, dtype=np.float32)
    x_train = sp.csc_matrix(x_train, dtype=np.float32)
    clf = pardfrocc.ParDFROCC(epsilon=0.1)
    # print(x_train)
    clf.fit(x_train)
    # print(x_test.shape,y_test.shape)
    scores = clf.decision_function(x_val)
    predictions = clf.predict(x_val)
    roc = roc_auc_score(y_val, scores)
    prc = precision_score(y_val,predictions,pos_label=0)
    f1 = f1_score(y_val, predictions,pos_label=0)
    a += roc
    b += prc
    c += f1
    print(j,k2,l)
    print(p,q,r)
    print("{:.3f}".format(roc),"{:.3f}".format(prc),"{:.3f}".format(f1))
    print("{:.3f}".format(a/3),"{:.3f}".format(b/3),"{:.3f}".format(c/3))








    #
    # clf = pardfrocc.ParDFROCC(epsilon=0.1)
    # # print(x_train)
    # clf.fit(x_train)
    # # print(x_test.shape,y_test.shape)
    # scores = clf.decision_function(x_test)
    # predictions = clf.predict(x_test)
    # roc = roc_auc_score(y_test, scores)
    # prc = precision_score(y_test, predictions, pos_label=0)
    # f1 = f1_score(y_test, predictions, pos_label=0)
    # print("{:.3f}".format(roc), "{:.3f}".format(prc), "{:.3f}".format(f1))
    # print()
    # clf = pardfrocc.ParDFROCC(epsilon=0.5)
    # # print(x_train)
    # clf.fit(x_train)
    # # print(x_test.shape,y_test.shape)
    # scores = clf.decision_function(x_test)
    # predictions = clf.predict(x_test)
    # roc = roc_auc_score(y_test, scores)
    # prc = precision_score(y_test, predictions, pos_label=0)
    # f1 = f1_score(y_test, predictions, pos_label=0)
    # print("{:.3f}".format(roc), "{:.3f}".format(prc), "{:.3f}".format(f1))
    # print()
    # clf = pardfrocc.ParDFROCC(epsilon=0.9)
    # # print(x_train)
    # clf.fit(x_train)
    # # print(x_test.shape,y_test.shape)
    # scores = clf.decision_function(x_test)
    # predictions = clf.predict(x_test)
    # roc = roc_auc_score(y_test, scores)
    # prc = precision_score(y_test, predictions, pos_label=0)
    # f1 = f1_score(y_test, predictions, pos_label=0)
    # print("{:.3f}".format(roc), "{:.3f}".format(prc), "{:.3f}".format(f1))
    #
    # # print(y_test,predictions,scores)



