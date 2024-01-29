# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:46:21 2023

@author: JC TU
"""


import time
from os import path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import tree as miptree
from sklearn import tree



train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

seed=1


df=pd.read_csv(r'data\monk2.csv')
x=df.drop('target',axis=1)
y=df['target']
x = np.array(x.values)
y = np.array(y.values)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=seed)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                test_size=test_ratio/(test_ratio+val_ratio), random_state=seed)


brtree = miptree.f1booleanoptimalDecisionTreeClassifier(max_depth=2, min_samples_split=1, alpha=0.01,N=3, warmstart=True, timelimit=300, output=True)

tick = time.time()
brtree.fit(x_train, y_train)
tock = time.time()
y_train_pred = brtree.predict(x_train)
y_test_pred = brtree.predict(x_test)
y_val_pred = brtree.predict(x_val)

train_time = tock - tick
train_acc = accuracy_score(y_train, brtree.predict(x_train))
val_acc = accuracy_score(y_val, brtree.predict(x_val))
test_acc = accuracy_score(y_test, brtree.predict(x_test))


f1_train= f1_score(y_train,y_train_pred)

f1_test=  f1_score(y_test,y_test_pred)

f1_val=   f1_score(y_val,y_val_pred)


print('f1_train:', f1_train, 'f1_val:', f1_val, 'f1_test:',f1_test)



