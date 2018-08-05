import numpy as np
import data_map
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import time

def data_clean(data_in):
    data_out = data_in.fillna('moderate')
    return data_out

def data_trans(data_in):
    data_out = data_in.copy()
    data_out['Sex'] = data_in['Sex'].map(data_map.sex)
    data_out['Housing'] = data_in['Housing'].map(data_map.housing)
    data_out['Saving accounts'] = data_in['Saving accounts'].map(data_map.saving)
    data_out['Checking account'] = data_in['Checking account'].map(data_map.checking)
    data_out['Purpose'] = data_in['Purpose'].map(data_map.purpose)
    data_out['Risk'] = data_in['Risk'].map(data_map.risk)

    X = data_out[data_map.feature].values
    y = data_out[data_map.label].values
    return X, y

def dataset_view(train_data, test_data):
    print('查看数据：')
    print('训练数据有{}'.format(len(train_data)))
    print('测试数据有{}'.format(len(test_data)))
    plt.figure()
    ax1 = plt.subplot(1,2,1)
    sns.countplot(x='Risk', data=train_data)
    plt.subplot(1,2,2,sharey=ax1)
    sns.countplot(x='Risk', data=test_data)
    plt.tight_layout()
    plt.savefig(os.path.join(data_map.dataout, '查看数据.png'))
    #plt.show()

def train_test_model(X_train, y_train, X_test, y_test, parametre, model_name):
    models = []
    accuracy = []
    times =[]
    for para in parametre:
        #print(para)
        if model_name == 'kNN':
            #print('训练kNN(k={})'.format(para))
            model = KNeighborsClassifier(n_neighbors=para)
        elif model_name == 'LR':
            #print('训练LR(c={})'.format(para))
            model = LogisticRegression(C=para)

        starttime = time.time()
        model.fit(X_train, y_train)
        endtime = time.time()
        duration = endtime - starttime
        #print('耗时{}s'.format(duration))
        acc = model.score(X_test,y_test)
        #print(acc)

        models.append(model)
        accuracy.append(acc)
        times.append(duration)

    mean_duration = np.mean(times)
    best_index = np.argmax(accuracy)
    best_acc = accuracy[best_index]
    best_model = models[best_index]

    return best_acc,best_model,mean_duration
