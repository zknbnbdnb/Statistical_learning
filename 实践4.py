from sklearn.neural_network import MLPRegressor
import os
from sklearn.metrics import classification_report
import numpy as np
import scipy.io as sio
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb



def readPIE(path):
    root = 'D:\pytorch\统计学习方法实践\PIE'
    fin_path = os.path.join(root, path)
    data = sio.loadmat(fin_path)
    X,y = data['fea'].astype(np.float64),data['gnd'].ravel()
    y = LabelEncoder().fit(y).transform(y).astype(int)
    return X,y


path = ['C05.mat', 'C07.mat', 'C09.mat', 'C27.mat', 'C29.mat']
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=3000)

X_train, y_train = readPIE(path[0])
X_test, y_test = readPIE(path[1])
model.fit(X_train, y_train)
pre = model.predict(X_test)
print(classification_report(pre, y_test))

# X, y = readPIE(path[0])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022)
# model.fit(X_train, y_train)
# pre = model.predict(X_test)
# print(classification_report(pre, y_test))