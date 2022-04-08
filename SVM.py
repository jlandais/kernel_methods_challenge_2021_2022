import numpy as np
from tqdm import tqdm
from itertools import combinations
from SVC import KernelSVC

class SVM:

    def __init__(self, C, kernel, epsilon = 1e-3):
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        self.X = None
        self.y = None
        self.num_classes = None
        self.models = {}
    
    def fit(self, X, y):
        self.num_classes = np.max(y)+1
        comb = []
        for i,j in combinations(range(self.num_classes), 2):
            comb.append((i,j))
        for i,j in tqdm(comb):
            X_ij = X[np.argwhere( (y == i) | (y == j))]
            X_ij = X_ij.reshape((X_ij.shape[0],-1))
            y_ij = np.where(y[np.argwhere( (y == i) | (y == j))] == i , -1 ,1).reshape((-1,))
            self.models[str(i)+str(j)] = KernelSVC(C=self.C, kernel=self.kernel, epsilon = 1e-1)
            self.models[str(i)+str(j)].fit(X_ij, y_ij)
    
    def predict(self, X):
        preds = {}
        y = np.zeros((X.shape[0],self.num_classes))
        for i,j in combinations(range(self.num_classes), 2):
            preds[str(i)+str(j)] = self.models[str(i)+str(j)].predict(X)
            for k,x in enumerate(preds[str(i)+str(j)]):
                if x == -1:
                    y[k][i] += 1
                else:
                    y[k][j] += 1

        return np.argmax(y, axis = 1)