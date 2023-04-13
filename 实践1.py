import os
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import numpy as np
import scipy.linalg as la
import scipy.io as sio
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report
import lightgbm as lgb

def readBooks(path):
    root = 'D:\pytorch\统计学习方法实践\Amazon Reviews'
    file = os.path.join(root, path)
    X, y = load_svmlight_files([file])
    X = X.toarray()
    y = LabelEncoder().fit(y).transform(y).astype(np.float64)
    #X = scale(X)
    return X,y

if __name__ == '__main__':
    path = ['books.svmlight', 'dvd.svmlight', 'electronics.svmlight', 'kitchen.svmlight']

    X_t, y_t = [], []
    for i in path:
        xtmp, ytmp = readBooks(i)
        X_t.append(xtmp)
        y_t.append(ytmp)
        info = [0 for i in range(2)]
        for j in ytmp:
            info[int(j)] += 1
        print(info)
    model = BernoulliNB()
    for i in range(len(X_t)):
        new_X_train, new_y_train = np.vstack((X_t[i][:500, :], X_t[i][1000:1500,:])), \
                                    np.hstack((y_t[i][:500],y_t[i][1000:1500]))
        for j in range(len(y_t)):
            new_X_test, new_y_test = np.vstack((X_t[j][:500, :], X_t[j][1000:1500,:])), \
                                    np.hstack((y_t[j][:500],y_t[j][1000:1500]))
            fin_X_train = np.vstack((new_X_train, new_X_test))
            fin_y_train = np.hstack((new_y_train, new_y_test))
            train_test = np.vstack([fin_X_train, X_t[j]])  # new training data
            lgb_data = lgb.Dataset(train_test, label=np.array([0] * len(fin_X_train) + [1] * len(X_t[j])))
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
            result = lgb.cv(params, lgb_data, num_boost_round=100, nfold=10, verbose_eval=20)
            print('交叉验证中最优的AUC为 {:.5f}，对应的标准差为{:.5f}.'.format(
                result['auc-mean'][-1], result['auc-stdv'][-1]))
            model.fit(fin_X_train, fin_y_train)
            pre = model.predict(X_t[j])
            print(classification_report(pre, y_t[j]))


