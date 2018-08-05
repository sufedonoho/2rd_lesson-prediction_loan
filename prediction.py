import os
import pandas as pd
import data_map
import data_process
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def main():
    print("have a test")
    print(os.path)
    print(os.getcwd())
    data_in = pd.read_csv(os.path.join(data_map.datain, 'german_credit_data.csv'))
    print(data_in.info())
    print(data_in.head())

    clean_data = data_process.data_clean(data_in)
    print(clean_data.info())
    print(clean_data.head())
    #clean_data.to_csv(os.path.join(data_map.dataout, 'clean_data.csv'), index=False,encoding='utf-8')
    """
    X, y = data_process.data_trans(clean_data)
    print(X.shape)
    print(y.shape)
    """
    train_data, test_data = train_test_split(clean_data, test_size=0.25, random_state=10)
    data_process.dataset_view(train_data,test_data)

    X_train, y_train = data_process.data_trans(train_data)
    X_test, y_test = data_process.data_trans(test_data)
    model_name_para_dic = {'kNN':[5,10,15],
                           'LR':[0.01,1,100]}
    for model_name, para in model_name_para_dic.items():
        print(model_name,para)
        duration = data_process.train_test_model(X_train,y_train,X_test,y_test,para,model_name)
        print(duration)

if __name__=="__main__":
    main()
