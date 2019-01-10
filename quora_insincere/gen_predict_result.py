# !/usr/bin/env python3

import os
import pandas as pd
import numpy as np

tsv_file = './test_results.tsv'
csv_file = './data/test.csv'

result_list = None
with open(tsv_file) as f:
    result_list = f.readlines()
    print(len(result_list))

csv = pd.read_csv(csv_file)
print(len(csv))

flag_list = []
for index, rows in csv.iterrows():
    txt = result_list[index]
    txt = txt.strip()
    ss = txt.split('\t')
    flag = 1
    if float(ss[0]) > float(ss[1]):
        flag = 1
    else:
        flag = 0
    flag_list.append(flag)

flag_arr = np.array(flag_list)
print(csv.columns)

df_csv = csv['qid'].copy()

id_arr = np.array(csv['qid'])

# df_csv = pd.DataFrame(columns=['qid','flag'], data=np.reshape([id_arr, flag_arr],(-1,2)))

csv_data = {
    'qid':id_arr,
    'flag':flag_arr
}

df_csv = pd.DataFrame(data=csv_data)


df_csv.to_csv('./predict.csv')
# df_csv['flag'] = flag_arr
print('hello world!')