import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn import preprocessing
import itertools


class NaiveBayesNominal:
    def __init__(self):
        self.classes_ = None
        self.model = {}
        self.y_prior = {}

    def fit(self, X, y):
        le = preprocessing.LabelEncoder()
        le.fit(y)
        self.classes_ = le.classes_
        for p in y:
            self.model[p] = self.model.get(p, 0) + 1
        for i, p in enumerate(X):
            for f, v in enumerate(p):
                pair = ((f, v), y[i])
                self.model[pair] = self.model.get(pair, 0) + 1
        for k, v in self.model.items():
            if type(k) == np.int64:
                self.y_prior[k] = v / len(y)
            else:
                self.y_prior[k] = v / self.model[k[1]]

    def predict_proba(self, X):
        proba = {}

        tmp2 = 0
        for cc in self.classes_:
            tmp3 = self.y_prior[cc]
            for ff, vv in enumerate(X):
                tmp3 *= self.y_prior[((ff, vv), cc)]
            tmp2 += tmp3

        for c in self.classes_:
            proba[c] = 0
            tmp1 = self.y_prior[c]
            for f, v in enumerate(X):
                tmp1 *= self.y_prior[((f, v), c)]
            proba[c] = tmp1 / tmp2
        return proba

    def predict(self, X):
        cc = []
        for p in X:
            max = 0
            c = None
            for k, v in self.predict_proba(p).items():
                if v > max:
                    max = v
                    c = k
            cc.append(c)
        return cc


class NaiveBayesGaussian:
    def __init__(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        raise NotImplementedError

    def fit(self, X, yy):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
