import os
import random

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
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

def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))



class ANN:
    def predict(self, x):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(x, w) + b
            x = self.activation(z)
        return self.classes_[np.argmax(x, axis=1)]


class BP(ANN):
    def __init__(self, layers, batch):
        self.layers = layers
        self.num_layers = len(layers)
        self.batch = batch
        self.activation = logistic
        self.activation_deactivation = logistic_derivative
        self.biases = [np.random.rand(x) for x in layers[1:]]
        self.weights = [np.random.rand(x, y) for x, y in zip(layers[: -1], layers[1:])]


    def fit(self, x, y, lr, epochs):
        label_bin = LabelBinarizer()
        y = label_bin.fit_transform(y)
        self.classes_ = label_bin.classes_
        train_data = [(x, y) for x, y in zip(x, y)]
        n = len(train_data)
        for i in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k: k + self.batch] for k in range(0, n, self.batch)]
            for sub_batch in batches:
                sub_x = []
                sub_y = []
                for tmp_x, tmp_y in sub_batch:
                    sub_x.append(tmp_x)
                    sub_y.append(tmp_y)
                activations = [np.array(sub_x)]
                for w, b in zip(self.weights, self.biases):
                    res = np.dot(activations[-1], w) + b
                    output = self.activation(res)
                    activations.append(output)
                err = activations[-1] - np.array(sub_y)
                details = [err * self.activation_deactivation(activations[-1])]
                for i in range(self.num_layers - 2, 0, -1):
                    details.append(self.activation_deactivation(activations[i]) *
                                    np.dot(details[-1], self.weights[i].T))
                details.reverse()
                for j in range(self.num_layers - 1):
                    details = lr / self.batch * ((np.atleast_2d(activations[j].sum(axis=0)).T).dot(np.atleast_2d(
                        details[j].sum(axis=0)
                    )))
                    self.weights[j] -= details
                    details = lr / self.batch * details[j].sum(axis=0)
                    self.biases[j] -= details
        return self

path = ['C05.mat', 'C07.mat', 'C09.mat', 'C27.mat', 'C29.mat']

X_t, y_t = [], []
for i in path:
    xtmp, ytmp = readPIE(i)
    X_t.append(xtmp)
    y_t.append(ytmp)
    info = [0 for i in range(68)]
    for j in ytmp:
        info[j] += 1
    print(info)


org_tt_step, tt_step = 24, 17
new_X_test, new_y_test = X_t[1][:tt_step, :], y_t[1][:tt_step]
new_X, new_y = X_t[1][tt_step:org_tt_step, :], y_t[1][tt_step:org_tt_step]

for f in range(org_tt_step, len(X_t[1]), org_tt_step):
    new_X_test = np.vstack((new_X_test, X_t[1][f:f+tt_step, :]))
    new_X = np.vstack((new_X, X_t[1][f+tt_step:f+org_tt_step, :]))
    new_y_test = np.hstack((new_y_test, y_t[1][f:f+tt_step]))
    new_y = np.hstack((new_y, y_t[1][f+tt_step:f+org_tt_step]))

X_tr = np.vstack((X_t[0], new_X_test))
y_tr = np.hstack((y_t[0], new_y_test))

clf = BP([X_tr.shape[1], 68], 100).fit(x=X_tr, y=y_tr, lr=1e-1, epochs=3000)
pl = clf.predict(new_X)
print(classification_report(new_y, pl))


# model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=3000)
# model.fit(X_t[0], y_t[0])
# pl = model.predict(X_t[1])
# print(classification_report(pl, y_t[1]))

train_test = np.vstack([X_tr, new_X]) # new training data
lgb_data = lgb.Dataset(train_test, label=np.array([0]*len(X_tr)+[1]*len(new_X)))
params = {
        'booster': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 5,
        'min_data_in_leaf': 91,
        'max_bin': 205,
        'max_depth': 8,
        'num_leaves':20,
        'max_bin':50,
        "learning_rate": 0.01,
        'feature_fraction': 0.6,
        "bagging_fraction": 1.0,  # 每次迭代时用的数据比例
        'bagging_freq': 45,
        'min_split_gain': 1.0,
        'min_child_samples': 10,
        'lambda_l1': 0.3, 'lambda_l2': 0.6,
        'n_jobs': -1,
        'silent': True,  # 信息输出设置成1则没有信息输出
        'seed': 1000,
        'verbose': -1,
    }
result = lgb.cv(params, lgb_data, num_boost_round=1000, nfold=10, verbose_eval=20)
print('交叉验证中最优的AUC为 {:.5f}，对应的标准差为{:.5f}.'.format(
    result['auc-mean'][-1], result['auc-stdv'][-1]))




