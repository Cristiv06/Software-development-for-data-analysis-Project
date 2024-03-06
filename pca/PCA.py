import numpy as np
import pandas as pd

class PCA:
    def __init__(self, X):
        self.X = X
        Cov = np.cov(m=X, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(a=Cov)
        k_desc = [k for k in reversed(np.argsort(self.eigenvalues))]
        self.alpha = self.eigenvalues[k_desc]
        self.A = self.eigenvectors[:, k_desc]

        for j in range(self.A.shape[1]):
            minCol = np.min(a=self.A[:, j], axis=0)
            maxCol = np.max(a=self.A[:, j], axis=0)
            if np.abs(minCol) > np.abs(maxCol):
                self.A[:, j] = (-1) * self.A[:, j]

        self.C = self.X @ self.A
        self.Rxc = self.A * np.sqrt(self.alpha)
        self.C2 = self.C * self.C

    def getAlpha(self):
        return self.alpha

    def getA(self):
        return self.A

    def getPrinComp(self):
        return self.C

    def getFactorLoadings(self):
        return self.Rxc

    def getScores(self):
        return self.C / np.sqrt(self.alpha)

    def getQualObs(self):
        SL = np.sum(self.C2, axis=1)
        return np.transpose(self.C2.T / SL)

    def getContribObs(self):
        return self.C2 / (self.X.shape[0] * self.alpha)

    def getCommun(self):
        Rxc2 = np.square(self.Rxc)
        return np.cumsum(a=Rxc2, axis=1)
