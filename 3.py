import pandas as pd
from sklearn.ensemble import IsolationForest
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
def main():

    train_age =  np.random.randint(18,60,[1000,1])
    train_salary  = np.random.randint(30,90,[1000,1])
    #sex = np.random.randint(1,3,[100,1])
    train = np.concatenate((train_age,train_salary), axis=1)

    test_age =  np.random.randint(18,60,[100,1])
    test_salary  = np.random.randint(30,90,[100,1])
    #sex = np.random.randint(1,3,[100,1])
    test = np.concatenate((test_age,test_salary), axis=1)

    outliers_age =  np.random.randint(1,10,[100,1])
    outliers_salary  = np.random.randint(10,20,[100,1])
    #sex = np.random.randint(1,3,[100,1])
    outliers = np.concatenate((outliers_age,outliers_salary), axis=1)

    outliers1_age =  np.random.randint(61,100,[10,1])
    outliers1_salary  = np.random.randint(100,200,[10,1])
    #sex = np.random.randint(1,3,[100,1])
    outliers1 = np.concatenate((outliers1_age,outliers1_salary), axis=1)


    clf = IsolationForest(max_samples=100, contamination=0.05)
    clf.fit(train)

    xx, yy = np.meshgrid(np.linspace(1, 100, 50), np.linspace(1, 100, 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    b1 = plt.scatter(train[:, 0], train[:, 1], c='white')
    b2 = plt.scatter(test[:, 0], test[:, 1], c='green')
    c = plt.scatter(outliers[:, 0], outliers[:, 1], c='red')


    plt.show()

if __name__ == '__main__':
    main()

