import numpy as np
import pandas as pd
from kernels import RBF
from hog_descriptor import projection_HOG
from SVC import KernelSVC
from SVM import SVM

if __name__ == '__main__':
    print("loading data...")
    y = np.asarray(np.genfromtxt('Ytr.csv', delimiter=',')[1:,1],dtype=int)
    X = projection_HOG(np.genfromtxt('Xtr.csv', delimiter=',')[:,:-1])
    X_test = projection_HOG(np.genfromtxt('Xte.csv', delimiter=',')[:,:-1])

    model = SVM(1.0, RBF(3.4).kernel)
    model.fit(X,y)
    pred = model.predict(X_test)

    linear_pred = pd.DataFrame({"Prediction":pred})
    linear_pred.insert(0, "Id", linear_pred.index)
    linear_pred["Id"] = linear_pred["Id"]+1
    linear_pred.to_csv('Yte.csv',index=False)