# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:31:18 2023

@author: JC TU
"""



import time
from os import path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
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

#
if sum(y)> len(y) -sum(y):
    c1=1;
    c0=sum(y)/(len(y) -sum(y))
    
else:
    c0=1;
    c1=(len(y) -sum(y))/sum(y)
    
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=seed)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                test_size=test_ratio/(test_ratio+val_ratio), random_state=seed)


brtree = miptree.costbooleanoptimalDecisionTreeClassifier(max_depth=2, min_samples_split=1, alpha=0.01,N=3, warmstart=True, timelimit=300, c0=c0,c1=c1, output=True)

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


def totalcost(y_test,y_test_pred):
    n10=0
    n01=0
    for i in range(len(y_test)):
        if y_test[i]-y_test_pred[i]>=0.2:
            n10=n10+1
        elif y_test[i]-y_test_pred[i]<=-0.2:
            n01=n01+1
    
    tcost=c1* n10 + c0 * n01
    return tcost

cost_train= totalcost(y_train,y_train_pred)

cost_test=  totalcost(y_test,y_test_pred)

cost_val=   totalcost(y_val,y_val_pred)



print('cost_train:', cost_train, 'cost_test:', cost_test, 'cost_val:',cost_val)



