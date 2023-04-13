import os
import random

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer, StandardScaler
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



from sklearn.tree import DecisionTreeClassifier
import numpy as np


class TrAdaboost:
    def __init__(self, base_classifier=DecisionTreeClassifier(), N=10):
        self.base_classifier = base_classifier
        self.N = N
        self.beta_all = np.zeros([1, self.N])
        self.classifiers = []

    def fit(self, x_source, x_target, y_source, y_target):
        x_train = np.concatenate((x_source, x_target), axis=0)
        y_train = np.concatenate((y_source, y_target), axis=0)
        x_train = np.asarray(x_train, order='C')
        y_train = np.asarray(y_train, order='C')
        y_source = np.asarray(y_source, order='C')
        y_target = np.asarray(y_target, order='C')

        row_source = x_source.shape[0]
        row_target = x_target.shape[0]

        # 初始化权重
        weight_source = np.ones([row_source, 1]) / row_source
        weight_target = np.ones([row_target, 1]) / row_target
        weights = np.concatenate((weight_source, weight_target), axis=0)

        beta = 1 / (1 + np.sqrt(2 * np.log(row_source / self.N)))

        result = np.ones([row_source + row_target, self.N])
        for i in range(self.N):
            weights = self._calculate_weight(weights)
            self.base_classifier.fit(x_train, y_train, sample_weight=weights[:, 0])
            self.classifiers.append(self.base_classifier)

            result[:, i] = self.base_classifier.predict(x_train)
            error_rate = self._calculate_error_rate(y_target,
                                                    result[row_source:, i],
                                                    weights[row_source:, :])

            print("Error Rate in target data: ", error_rate, 'round:', i, 'all_round:', self.N)

            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                self.N = i
                print("Early stopping...")
                break
            self.beta_all[0, i] = error_rate / (1 - error_rate)

            # 调整 target 样本权重 正确样本权重变大
            for t in range(row_target):
                weights[row_source + t] = weights[row_source + t] * np.power(self.beta_all[0, i], -np.abs(result[row_source + t, i] - y_target[t]))
            # 调整 source 样本 错分样本变大
            for s in range(row_source):
                weights[s] = weights[s] * np.power(beta, np.abs(result[s, i] - y_source[s]))

    def predict(self, x_test):
        result = np.ones([x_test.shape[0], self.N + 1])
        predict = []

        i = 0
        for classifier in self.classifiers:
            y_pred = classifier.predict(x_test)
            result[:, i] = y_pred
            i += 1

        for i in range(x_test.shape[0]):
            left = np.sum(result[i, int(np.ceil(self.N / 2)): self.N] *
                          np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)):self.N]))

            right = 0.5 * np.sum(np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)): self.N]))

            if left >= right:
                predict.append(1)
            else:
                predict.append(0)
        return predict

    def predict_prob(self, x_test):
        result = np.ones([x_test.shape[0], self.N + 1])
        predict = []

        i = 0
        for classifier in self.classifiers:
            y_pred = classifier.predict(x_test)
            result[:, i] = y_pred
            i += 1

        for i in range(x_test.shape[0]):
            left = np.sum(result[i, int(np.ceil(self.N / 2)): self.N] *
                          np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)):self.N]))

            right = 0.5 * np.sum(np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)): self.N]))
            predict.append([left, right])
        return predict

    def _calculate_weight(self, weights):
        sum_weight = np.sum(weights)
        return np.asarray(weights / sum_weight, order='C')

    def _calculate_error_rate(self, y_target, y_predict, weight_target):
        sum_weight = np.sum(weight_target)
        return np.sum(weight_target[:, 0] / sum_weight * np.abs(y_target - y_predict))

path = ['C05.mat', 'C07.mat', 'C09.mat', 'C27.mat', 'C29.mat']

X_t, y_t = [], []
for i in path:
    xtmp, ytmp = readPIE(i)
    X_t.append(xtmp)
    y_t.append(ytmp)
    info = [0 for i in range(68)]


X_train, X_test, y_train, y_test = X_t[0], X_t[1], y_t[0], y_t[1]
X_train_s, fin_test_x, y_train_s, fin_test_y = train_test_split(X_test, y_test, test_size=0.9, random_state=2022)
X_train = np.vstack((X_train, X_train_s))
y_train = np.hstack((y_train, y_train_s))
model = MLPClassifier(hidden_layer_sizes=(512,256), max_iter=100)
model.fit(X_train, y_train)
pl = model.predict(fin_test_x)
print(classification_report(pl, fin_test_y))