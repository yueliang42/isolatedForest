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

    outliers1_age =  np.random.randint(61,100,[100,1])
    outliers1_salary  = np.random.randint(100,200,[100,1])
    #sex = np.random.randint(1,3,[100,1])
    outliers1 = np.concatenate((outliers1_age,outliers1_salary), axis=1)


    clf = IsolationForest(max_samples=100, contamination=0.01)
    clf.fit(train)


    Z = clf.predict(train)
    z_neg = np.zeros(shape=(1,2))
    for i in range(0,len(Z)):
        if(Z[i]<0):
            z_neg =  np.row_stack((z_neg, train[i]))
    z_neg = np.delete(z_neg,0,axis=0)


    Z1 = clf.predict(test)
    z1_neg = np.zeros(shape=(1,2))
    for i in range(0,len(Z1)):
        if(Z1[i]<0):
            z1_neg =  np.row_stack((z1_neg, test[i]))
    z1_neg = np.delete(z1_neg,0,axis=0)

    Z2 = clf.predict(outliers)
    z2_neg = np.zeros(shape=(1,2))
    for i in range(0,len(Z2)):
        if(Z2[i]<0):
            z2_neg =  np.row_stack((z2_neg, outliers[i]))
    z2_neg = np.delete(z2_neg,0,axis=0)


    Z3 = clf.predict(outliers1)
    z3_neg = np.zeros(shape=(1,2))
    for i in range(0,len(Z3)):
        if(Z3[i]<0):
            z3_neg =  np.row_stack((z3_neg, outliers1[i]))
    z3_neg = np.delete(z3_neg,0,axis=0)





    xx, yy = np.meshgrid(np.linspace(1, 100, 50), np.linspace(1, 200, 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    b1 = plt.scatter(z_neg[:, 0], z_neg[:, 1], c='red')
    b2 = plt.scatter(z1_neg[:, 0], z1_neg[:, 1], c='green')
    c = plt.scatter(z2_neg[:, 0], z2_neg[:, 1], c='blue')
    d = plt.scatter(z3_neg[:, 0], z3_neg[:, 1], c='black')
    plt.show()

if __name__ == '__main__':
    main()

