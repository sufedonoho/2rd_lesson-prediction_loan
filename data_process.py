import data_map
import pandas as pd

def data_clean(data_in):
    data_out = data_in.fillna('moderate')
    return data_out

def data_trans(data_in):
    data_out = data_in.copy()
    data_out['Sex'] = data_in['Sex']
