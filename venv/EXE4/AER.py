import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import math
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


class AER:

    def __init__(self, cov_type='tied', lambda_A=0.6, max_steps=5, max_loss=0.001, K=5, xgb_params=None, random_state=2020):
        self.classes_ = None
        if xgb_params is None:
            xgb_params = {}
        self.w = [0] * K
        self.f = []
        for _ in range(2 * K):
            self.f.append(xgb.XGBClassifier(**xgb_params))

        self.random_state = random_state
        self.params = {'cov_type': cov_type,
                       'lambda_A': lambda_A,
                       'max_steps': max_steps,
                       'max_loss': max_loss,
                       'K': K,
                       'xgb_params': xgb_params
                       }

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def build_global_geometry_dataset(self, i, predicted_probas, X_n, X_k):
        best_part_idx = predicted_probas.sort_values(by='likelihood_{}'.format(i), ascending=False).head(
            int(X_n.shape[0] / self.params['K'])).index
        return pd.concat([X_n.iloc[best_part_idx], X_k])

    @staticmethod
    def build_local_geometry_dataset(i, predicted_probas, X_n, X_k):
        best_part_idx = predicted_probas.sort_values(by='likelihood_{}'.format(i), ascending=False).head(
            int(X_k.shape[0])).index
        return pd.concat([X_n.iloc[best_part_idx], X_k, X_n.sample(int(0.5 * X_k.shape[0]))])

    ####################################
    # X_n - majority dataset
    # X_k - minority dataset
    # K - number of gaussian distributions (later will be also number of components of the ensemble)
    ####################################
    def build_datasets(self, X_n, X_k):
        G_datasets = []
        L_datasets = []

        best_gmm = None
        best_bic = 0
        X = pd.concat([X_n, X_k])
        # find the best model with lowest bayesian information criterion (bic)
        for i in range(5):
            gmm = GaussianMixture(n_components=self.params['K'],
                                  covariance_type=self.params['cov_type'], max_iter=200,
                                  random_state=(self.random_state + i))
            gmm.fit(X)
            bic = gmm.bic(X)
            if best_gmm is None or best_bic > bic:
                best_bic = bic
                best_gmm = gmm

        # For each gaussian centroid form two types of data sets:
        # 1. Build first data set by:
        #    1.1 Find len(X_n) / K closest samples from X_n
        #    1.2. Add X_k
        # 2. Build second data set by:
        #    2.1 Find len(X_k) closest samples from X_n
        #    2.2 Add floor(len(X_k) / 2) random samples from X_n
        #    3.2 Add X_k
        predicted_probas = pd.DataFrame(best_gmm.predict_proba(X_n),
                                        columns=['likelihood_{}'.format(i) for i in range(self.params['K'])])

        for i in range(self.params['K']):
            G_datasets.append(self.build_global_geometry_dataset(i, predicted_probas, X_n, X_k))
            L_datasets.append(self.build_local_geometry_dataset(i, predicted_probas, X_n, X_k))
        return G_datasets, L_datasets, best_gmm

    @staticmethod
    def S(x):
        return (x - min(x)) / (max(x) - min(x))

    # Eq 17
    def initial_weights(self, model_data, gmm):
        weight = []
        for md in model_data:
            if md[1].empty:
                continue
            bic = gmm.bic(md[1])
            aic = gmm.aic(md[1])
            weight.append(1.0 / (self.params['lambda_A'] * aic + (1 - self.params['lambda_A']) * bic))
        return weight / sum(weight)

    # Eq 11
    @staticmethod
    def loss(predicts, Y, w):
        s2 = predicts.mul(w).sum(axis=1)
        zero_idx = s2.eq(Y)
        one_idx = ~zero_idx & ((s2 <= 0) | (s2 >= 1))
        other_idx = (~zero_idx) & (~one_idx)
        s2[zero_idx] = 0
        s2[one_idx] = 1
        s2[other_idx] = Y[other_idx] * np.log(s2[other_idx]) + (1 - Y[other_idx]) * np.log(1 - s2[other_idx])
        s1 = s2.sum()
        return -1.0 * s1 / (2.0 * len(Y))

    # Eq 12
    @staticmethod
    def gradient(predicts, Y, w):
        to_mul = (1 - predicts.mul(w).sum(axis=1) - Y)
        to_mul[to_mul != 0] = 1.0 / to_mul[to_mul != 0]
        s1 = predicts.mul(to_mul, axis=0)
        return s1.sum() / len(Y)

    ####################################
    # f - list of 2K classifiers
    # X - list of 2K corresponding datasets
    # y - list of 2K corresponding labels should be (0, 1)
    # gmm - best GMM previously fitted
    ####################################
    def compute_weights(self, f, X, y, gmm):
        model_data = list(zip(f, X, y))
        w = self.initial_weights(model_data, gmm)

        X = pd.concat(X)
        y = pd.concat(y).reset_index(drop=True)
        predicts = pd.DataFrame()
        # get predictions for all f's
        for k in range(len(w)):
            f = model_data[k][0]
            predicts[str(k)] = f.predict(X)

        for t in range(self.params['max_steps']):
            D_L = self.gradient(predicts, y, w)
            lambda_t = 1 / (2 * math.sqrt(sum(D_L ** 2)))
            upper = self.S(w - lambda_t * D_L)
            lower = sum(upper)
            w = upper / lower
            model_loss = self.loss(predicts, y, w)
            if model_loss < self.params['max_loss']:
                break
        return w

    def fit(self, X, y):
        self.classes_, _ = np.unique(y, return_inverse=True)
        X_n = X[y != 0]
        X_k = X[y == 0]

        G_datasets, L_datasets, best_gmm = self.build_datasets(X_n, X_k)
        X_set = G_datasets + L_datasets
        Y_set = []
        i = 0

        for x in X_set:
            y_i = y[x.index]
            Y_set.append(y_i)
            self.f[i].fit(x, y_i)
            i += 1

        self.w = self.compute_weights(self.f, X_set, Y_set, best_gmm)

    def predict(self, X):
        D = self.predict_proba(X)
        return self.classes_[np.argmax(D, axis=1)]

    def predict_proba(self, X):
        ret = [0] * X.shape[0]
        for ensemble in list(zip(self.f, self.w)):
            ret += ensemble[0].predict(X) * ensemble[1]
        return np.stack(list(zip(1 - ret, ret)))
