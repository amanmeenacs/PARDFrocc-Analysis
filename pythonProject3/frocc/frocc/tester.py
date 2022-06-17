
from mnist import x_train, y_train, x_val, y_val, x_test, y_test
import numpy as np
import pandas as pd
import sklearn.datasets as skds
x,y = skds.fetch_openml(data_id =  1053, return_X_y = True)
print(type(x))
# iris = skds.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# Y = iris.target  # print(x)
# print(X,Y)
# print(x.shape,x)
# print(y.shape,y)

# iris = skds.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# y = iris.target  # print(x)
# y = y.astype("int")
# for i, _y in enumerate(y):
#     if _y not in [0]:
#         y[i] = 0
#     else:
#         y[i] = 1
# print(y)