import copy
import numpy as np
from sklearn.svm import SVC

def ovr_svm_multipro(X_train, y_train, kernel, probability, i):
    y_train_ = copy.deepcopy(y_train)

    y_train_[y_train_== i] = -1
    y_train_[y_train_!= -1] = 0
    y_train_[y_train_== -1] = 1

    model = SVC(kernel=kernel, gamma='auto', probability=probability)
    model.fit(X_train, y_train_)
    print("Done Class:", i)
    return model

def ovo_svm_multipro(X_train, y_train, kernel, i, j, gamma='auto'):
    X_train_ = []
    y_train_ = []

    for x, y in zip(X_train, y_train):
        if y == i or y == j:
            X_train_.append(x)
            y_train_.append(y)

    y_train_ = np.array(y_train_)

    y_train_[y_train_== i] = -1
    y_train_[y_train_!= -1] = 1
    y_train_[y_train_== -1] = 0

    model = SVC(kernel=kernel, gamma=gamma)
    model.fit(X_train_, y_train_)
    return model
