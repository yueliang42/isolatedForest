import pandas as pd
from sklearn.ensemble import IsolationForest
import random
import numpy as np

def main():
    ilf = IsolationForest(n_estimators=100,
                          n_jobs=-1,          # 使用全部cpu
                          verbose=2,
        )

    age =  np.random.randint(1,100,[100,1])
    salary  = np.random.randint(3000,20000,[100,1])
    sex = np.random.randint(1,3,[100,1])
    a = np.concatenate((age,salary,sex), axis=1)

    #data = pd.read_csv('data.csv', index_col="id")
    #data = data.fillna(0)
    # 选取特征，不使用标签(类型)
    #X_cols = ["age", "salary", "sex"]
    #print(data.shape)

    # 训练
    model = ilf.fit(a)

    shape = a.shape[0]
    print(shape)
    batch = 5
    
    all_pred = []
    for i in range(0,shape/batch):
        start = i * batch
        end = (i+1) * batch
        test = a[start:end]
        # 预测
        pred = model.predict(test)
        all_pred.extend(pred)
    
    a['pred'] = all_pred
    a.to_csv('outliers.csv', columns=["pred",], header=False)

if __name__ == '__main__':
    main()

