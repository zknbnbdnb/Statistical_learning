import random

import numpy as np
import imgaug.augmenters as iaa
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer, StandardScaler, Normalizer
import numpy as np
import scipy.io as sio
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn import svm

def readPIE(path):
    root = 'D:\pytorch\统计学习方法实践\PIE'
    fin_path = os.path.join(root, path)
    data = sio.loadmat(fin_path)
    X,y = data['fea'].astype(np.float64), data['gnd'].ravel()
    y = LabelEncoder().fit(y).transform(y).astype(int)
    return X,y

path = ['C05.mat', 'C07.mat', 'C09.mat', 'C27.mat', 'C29.mat']

X_t, y_t = [], []
for i in path:
    xtmp, ytmp = readPIE(i)
    X_t.append(xtmp)
    y_t.append(ytmp)
    info = [0 for i in range(68)]

X_train = X_t[0]
y_train = y_t[0]
X_test, y_test = X_t[0], y_t[1]

scaler = StandardScaler()
scaler.fit(X_train)
X_train,X_test = scaler.transform(X_train),scaler.transform(X_test)


X_train = np.hstack((np.ones((len(X_train),1)),X_train))
X_test = np.hstack((np.ones((len(X_test),1)),X_test))

negative = -1 * np.ones(X_train.shape[0])
positive = np.ones(X_test.shape[0])

XX,l = np.vstack((X_train,X_test)),np.hstack((negative,positive))
prob = LogisticRegression(C=.1).fit(XX,l).predict_proba(X_train)
weight = 1.0 * (X_train.shape[0] / X_test.shape[0]) * (prob[:,1] / prob[:,0])