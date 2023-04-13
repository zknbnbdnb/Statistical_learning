import os
import random
import time
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import scipy.io as sio
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import pandas as pd


def readPIE(path):
    root = 'D:\pytorch\统计学习方法实践\PIE'
    fin_path = os.path.join(root, path)
    data = sio.loadmat(fin_path)
    X,y = data['fea'].astype(np.float64),data['gnd'].ravel()
    y = LabelEncoder().fit(y).transform(y).astype(int)
    return X,y


def readBooks(path):
    root = 'D:\pytorch\统计学习方法实践\Amazon Reviews'
    file = os.path.join(root, path)
    X, y = load_svmlight_files([file])
    X = X.toarray()
    y = LabelEncoder().fit(y).transform(y).astype(np.float64)
    #X = scale(X)
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


    def fit(self, x, y, lr, epochs, weight=weight):
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
                err = activations[-1] * weight - np.array(sub_y)
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

def PIE_eval(X, y, res, weights):
    X_t, y_t = X, y
    params = {
        'booster': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 5,
        'min_data_in_leaf': 91,
        'max_bin': 205,
        'max_depth': 8,
        'num_leaves': 20,
        'max_bin': 50,
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
    for i in range(len(X_t)):
        if weights:
            org_tr_step, tr_step = len(X_t[i]) // len(set(y_t[i])), len(X_t[i]) // len(set(y_t[i])) // 2

            tmp_nda = X_t[i][: org_tr_step, :]
            random.shuffle(tmp_nda)
            new_X_train = tmp_nda[: tr_step, :]
            new_y_train = y_t[i][: tr_step]

            for f in range(org_tr_step, len(X_t[i]), org_tr_step):
                tmp_nda = X_t[i][f: f + org_tr_step, :]
                random.shuffle(tmp_nda)

                new_X_train = np.vstack((new_X_train, tmp_nda[: tr_step, :]))
                new_y_train = np.hstack((new_y_train, y_t[i][f: f + tr_step]))

            for j in range(len(X_t)):
                X_test, y_test = X_t[j], y_t[j]
                org_tt_step, tt_step = len(X_t[j]) // len(set(y_t[j])), len(X_t[j]) // len(set(y_t[j])) // 2

                tmp_nda = X_t[j][: org_tt_step, :]
                random.shuffle(tmp_nda)
                new_X_test = tmp_nda[: tt_step, :]
                new_y_test = y_t[j][: tt_step]

                for f in range(org_tt_step, len(X_t[j]), org_tt_step):
                    tmp_nda = X_t[j][f: f + org_tt_step, :]
                    random.shuffle(tmp_nda)

                    new_X_test = np.vstack((new_X_test, tmp_nda[: tt_step, :]))
                    new_y_test = np.hstack((new_y_test, y_t[j][f: f + tt_step]))

                fin_X_train, fin_y_train = np.vstack((new_X_train, new_X_test)), np.hstack((new_y_train, new_y_test))

                train_test = np.vstack([fin_X_train, X_test])  # new training data
                lgb_data = lgb.Dataset(train_test, label=np.array([0] * len(fin_X_train) + [1] * len(X_test)))
                result = lgb.cv(params, lgb_data, num_boost_round=100, nfold=10, verbose_eval=20)

                clf = BP([fin_X_train.shape[1], 68], 100).fit(x=fin_X_train, y=fin_y_train, lr=1e-1, epochs=3000)
                predict_label = clf.predict(X_test)

                dic = classification_report(predict_label, y_test, output_dict=True)
                res[i].append((dic['accuracy'], min(result['auc-mean'])))
                print(classification_report(predict_label, y_test))
        else:
            X_train, y_train = X_t[i], y_t[i]
            clf = BP([X_train.shape[1], 68], 100).fit(x=X_train, y=y_train, lr=1e-1, epochs=3000)
            for j in range(len(X_t)):
                X_test, y_test = X_t[j], y_t[j]

                train_test = np.vstack([X_train, X_test])  # new training data
                lgb_data = lgb.Dataset(train_test, label=np.array([0] * len(X_train) + [1] * len(X_test)))
                result = lgb.cv(params, lgb_data, num_boost_round=100, nfold=10, verbose_eval=20)

                predict_label = clf.predict(X_test)

                dic = classification_report(predict_label, y_test, output_dict=True)
                res[i].append((dic['accuracy'], min(result['auc-mean'])))
                print(classification_report(predict_label, y_test))

def Book_eval(X, y, res, weights):
    X_t, y_t = X, y
    params = {
        'booster': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 5,
        'min_data_in_leaf': 91,
        'max_bin': 205,
        'max_depth': 8,
        'num_leaves': 20,
        'max_bin': 50,
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
    for i in range(len(X_t)):
        if weights:
            new_X_train, new_y_train = np.vstack((X_t[i][:500, :], X_t[i][1000:1500, :])), \
                                        np.hstack((y_t[i][:500], y_t[i][1000:1500]))
            for j in range(len(X_t)):
                X_test, y_test = X_t[j], y_t[j]
                new_X_test, new_y_test = np.vstack((X_test[:500, :], X_test[1000:1500, :])), \
                                        np.hstack((y_test[:500], y_test[1000:1500]))

                fin_X_train, fin_y_train = np.vstack((new_X_train, new_X_test)), np.hstack((new_y_train, new_y_test))

                train_test = np.vstack([fin_X_train, X_test])  # new training data
                lgb_data = lgb.Dataset(train_test, label=np.array([0] * len(fin_X_train) + [1] * len(X_test)))
                result = lgb.cv(params, lgb_data, num_boost_round=100, nfold=10, verbose_eval=20)

                clf = BernoulliNB().fit(fin_X_train, fin_y_train)
                predict_label = clf.predict(X_test)

                dic = classification_report(predict_label, y_test, output_dict=True)
                res[i].append((dic['accuracy'], min(result['auc-mean'])))
                print(classification_report(predict_label, y_test))
        else:
            X_train, y_train = X_t[i], y_t[i]
            clf = BernoulliNB().fit(X_train, y_train)
            for j in range(len(X_t)):
                X_test, y_test = X_t[j], y_t[j]

                train_test = np.vstack([X_train, X_test])  # new training data
                lgb_data = lgb.Dataset(train_test, label=np.array([0] * len(X_train) + [1] * len(X_test)))
                result = lgb.cv(params, lgb_data, num_boost_round=100, nfold=10, verbose_eval=20)

                predict_label = clf.predict(X_test)

                dic = classification_report(predict_label, y_test, output_dict=True)
                res[i].append((dic['accuracy'], min(result['auc-mean'])))
                print(classification_report(predict_label, y_test))


if __name__ == '__main__':
    start = time.time()
    PIE_path = ['C05.mat', 'C07.mat', 'C09.mat', 'C27.mat', 'C29.mat']
    Book_path = ['books.svmlight', 'dvd.svmlight', 'electronics.svmlight', 'kitchen.svmlight']

    X_t, y_t = [], []
    X_t_b, y_t_b = [], []
    for i in PIE_path:
        xtmp, ytmp = readPIE(i)
        X_t.append(xtmp)
        y_t.append(ytmp)
    for i in Book_path:
        xtmp, ytmp = readBooks(i)
        X_t_b.append(xtmp)
        y_t_b.append(ytmp)

    PIE_res = [[], [], [], [], []]
    PIE_res_w = [[], [], [], [], []]
    Book_res = [[], [], [], []]
    Book_res_w = [[], [], [], []]

    PIE_eval(X_t, y_t, PIE_res_w, weights=True)
    PIE_eval(X_t, y_t, PIE_res, weights=False)

    Book_eval(X_t_b, y_t_b, Book_res, weights=False)
    Book_eval(X_t_b, y_t_b, Book_res_w, weights=True)
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)

    PIE_df = pd.DataFrame(PIE_res, columns=['1-Other', '2-Other', '3-Other', '4-Other', '5-Other'],
                        index=['ACC&AUC' for _ in range(5)])
    print(PIE_df)
    PIE_df_w = pd.DataFrame(PIE_res_w, columns=['1-Other', '2-Other', '3-Other', '4-Other', '5-Other'],
                        index=['acc' for _ in range(5)])
    print(PIE_df_w)
    Book_df = pd.DataFrame(Book_res, columns=['1-Other', '2-Other', '3-Other', '4-Other'],
                        index=['acc' for _ in range(4)])
    print(Book_df)
    Book_df_w = pd.DataFrame(Book_res_w, columns=['1-Other', '2-Other', '3-Other', '4-Other'],
                        index=['acc' for _ in range(4)])
    print(Book_df_w)

    end = time.time()
    print(end - start)



