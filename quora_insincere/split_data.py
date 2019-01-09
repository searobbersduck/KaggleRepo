# !/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd

infile = './data/train.csv'
outfile_train = './data/train_s.csv'
outfile_test = './data/test_s.csv'
infile_csv = pd.read_csv(infile)

print(infile_csv.columns)
print(infile_csv.dtypes)

infile_csv0 = infile_csv[infile_csv['target'] == 0]
infile_csv1 = infile_csv[infile_csv['target'] == 1]

mask = np.random.rand(len(infile_csv0)) < 0.8
infile_csv0_train = infile_csv0[mask]
infile_csv0_test = infile_csv0[~mask]

mask = np.random.rand(len(infile_csv1)) < 0.8
infile_csv1_train = infile_csv1[mask]
infile_csv1_test = infile_csv1[~mask]

train_csv = pd.concat([infile_csv0_train, infile_csv1_train])
test_csv = pd.concat([infile_csv1_train, infile_csv1_test])

print('train set total number: {}'.format(len(train_csv)))
print('test set total number: {}'.format(len(test_csv)))

train_csv = train_csv.sample(frac=1).reset_index(drop=True)
test_csv = test_csv.sample(frac=1).reset_index(drop=True)

print('train set total number after shuffle: {}'.format(len(train_csv)))
print('test set total number after shuffle: {}'.format(len(test_csv)))

train_cnt = len(train_csv)
test_cnt = len(test_csv)
flag0_cnt = len(infile_csv0)
flag1_cnt = len(infile_csv1)
total_cnt = flag0_cnt + flag1_cnt

print('flag 0/flag 1\t=\t{:.3f}/{:.3f}'.format(flag0_cnt/total_cnt, flag1_cnt/total_cnt))

train_csv.to_csv(outfile_train)
test_csv.to_csv(outfile_test)