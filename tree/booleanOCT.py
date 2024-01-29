# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:11:09 2023

@author: JC TU
"""



from collections import namedtuple
import numpy as np
from scipy import stats
import gurobipy as gp
from gurobipy import GRB
from sklearn import tree

class booleanoptimalDecisionTreeClassifier:
    """
    optimal classification tree
    """
    def __init__(self, max_depth=3, min_samples_split=1, alpha=0.01,N=5, warmstart=True, timelimit=600, output=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.N = N
        self.warmstart = warmstart
        self.timelimit = timelimit
        self.output = output
        self.trained = False
        self.optgap = None

        # node index
        self.n_index = [i+1 for i in range(2 ** (self.max_depth + 1) - 1)]
        self.b_index = self.n_index[:-2**self.max_depth] # branch nodes
        self.l_index = self.n_index[-2**self.max_depth:] # leaf nodes

    def fit(self, x, y):
        """
        fit training data
        """
        # data size
        self.n, self.p = x.shape
        if self.output:
            print('Training data include {} instances, {} features.'.format(self.n,self.p))

        # labels
        self.labels = np.unique(y)

        # scale data
        self.scales = np.max(x, axis=0)
        self.scales[self.scales == 0] = 1

        # solve MIP
        # m, a, b, c, d, l = self._buildMIP(x/self.scales, y)
        m, a, b, c, d, l = self._buildMIP(x, y)
        if self.warmstart:
            self._setStart(x, y, a, c, d, l)
        m.optimize()
        self.optgap = m.MIPGap

        # get parameters
        self._a = {ind:a[ind].x for ind in a}
        self._b = {ind:b[ind].x for ind in b}
        self._c = {ind:c[ind].x for ind in c}
        self._d = {ind:d[ind].x for ind in d}

        self.trained = True
    




    
        

    def predict(self, x):
        """
        model prediction
        """
        if not self.trained:
            raise AssertionError('This optimalDecisionTreeClassifier instance is not fitted yet.')
        
        def getleaf(tb):
            t=tb
            if tb % 2 == 1:
            
                t=2*t+1
            else:
              
                    t=2*t
            
            return t

        # leaf label
        labelmap = {}
        for t in self.l_index:
            for k in self.labels:
                if self._c[k,t] >= 1e-2:
                    labelmap[t] = k

        y_pred = []
        for xi in x:
            t = 1
            while t not in self.l_index:
                right = (sum([self._a[j,t] * xi[j] for j in range(self.p)]) + 1e-9 >= self._b[t] +1)
                if self._d[t] == 1:
                    if right:
                       t = 2 * t + 1
                       
                    else:
                         t = 2 * t
                     
                else:
                       t = getleaf(t)
            
            # label
            y_pred.append(labelmap[t])

        return np.array(y_pred)

    def _buildMIP(self, x, y):
        """
        build MIP formulation for Optimal Decision Tree
        """
        
        def get_l(t):
            lls=[]
            lrs=[]
            left=(t % 2 == 0)
            right=(t % 2 == 1)
            
            if t>=2 and left:
                while (t % 2 == 0):
                     lls.append(t)
                     t=t//2
                lls.append(t)
                lls.pop(0)
            
            if t>=3 and right:
                while (t % 2 == 1) and (t>=3):
                     lrs.append(t)
                     t=t//2
                lrs.append(t)
                lrs.pop(0)
            
            if left:
                return lls
            else:
                return lrs


        # create a model
        m = gp.Model('m')

        # output
        m.Params.outputFlag = self.output
        m.Params.LogToConsole = self.output
        # time limit
        m.Params.timelimit = self.timelimit
        
        m.params.Heuristics=0.2
        

        # model sense
        m.modelSense = GRB.MINIMIZE

        # variables
        a = m.addVars(self.p, self.b_index, vtype=GRB.BINARY, name='a') # splitting feature
        b = m.addVars(self.b_index, lb=0, ub=self.N, vtype=GRB.INTEGER, name='b') # splitting threshold
        c = m.addVars(self.labels, self.l_index, vtype=GRB.BINARY, name='c') # node prediction
        d = m.addVars(self.b_index, vtype=GRB.BINARY, name='d') # splitting option
        z = m.addVars(self.n, self.l_index, vtype=GRB.BINARY, name='z') # leaf node assignment
        #zc = m.addVars(self.n, self.labels, vtype=GRB.BINARY, name='z') # leaf node assignment
        l = m.addVars(self.l_index, vtype=GRB.BINARY, name='l') # leaf node activation
        L = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='L') # leaf node misclassified
        M = m.addVars(self.labels, self.l_index, vtype=GRB.CONTINUOUS, name='M') # leaf node samples with label
        N = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='N') # leaf node samples
        

        # objective function
        error=L.sum() / self.n 
        obj = error + self.alpha * gp.quicksum(a[j,t] for j in range(self.p) for t in self.b_index)
        m.setObjective(obj)

        m.addConstr(d.sum() >= 1)

        m.addConstrs(a.sum('*', t) <= self.N* d[t] for t in self.b_index)

        m.addConstrs(a.sum('*', t) >= d[t] for t in self.b_index)

        m.addConstrs(b[t] <= (self.N-1)* d[t] for t in self.b_index)

        #m.addConstrs(b[t] <= a.sum('*', t)-1 for t in self.b_index)

        m.addConstrs(d[t] <= d[t//2] for t in self.b_index if t != 1)

        m.addConstrs(c.sum('*', t) == l[t] for t in self.l_index)

        m.addConstrs(z.sum(i, '*') == 1 for i in range(self.n))



        m.addConstrs(z[i,t] <= l[t] for t in self.l_index for i in range(self.n))

        m.addConstrs(z.sum('*', t) >= self.min_samples_split * l[t] for t in self.l_index)

        for t in self.l_index:
            left = (t % 2 == 0)
            ta = t // 2
            while ta != 0:
                if left:
                    m.addConstrs(gp.quicksum(a[j,ta] * x[i,j]  for j in range(self.p))
                                 <=b[ta]+ self.N*(1 - z[i,t])
                                 for i in range(self.n))
                else:
                    m.addConstrs(gp.quicksum(a[j,ta] * x[i,j] for j in range(self.p))
                                  >=b[ta] +d[ta] - self.N*(2 - z[i,t]  - d[ta])
                                  for i in range(self.n))
                left = (ta % 2 == 0)
                ta //= 2
                
         # (11)
        m.addConstrs(gp.quicksum((y[i] == k) * z[i,t] for i in range(self.n)) == M[k,t]
                                  for t in self.l_index for k in self.labels)
         # (12)
        m.addConstrs(z.sum('*', t) == N[t] for t in self.l_index)
        # constraints
        # (13)
        m.addConstrs(L[t] >= N[t] - M[k,t] - self.n * (1 - c[k,t]) for t in self.l_index for k in self.labels)
        # (14)
        m.addConstrs(L[t] <= N[t] - M[k,t] + self.n * c[k,t] for t in self.l_index for k in self.labels)
        
            
        m.addConstrs(l[t]<=gp.quicksum(d[s] for s in get_l(t)) for t in self.l_index)

        m.addConstrs(self.max_depth*l[t]>=gp.quicksum(d[s] for s in get_l(t)) for t in self.l_index)

       
       
       
        
        

        return m, a, b, c, d, l

    @staticmethod
    def _calBaseline(y):
        """
        obtain baseline accuracy by simply predicting the most popular class
        """
        mode = stats.mode(y)[0][0]
        return np.sum(y == mode)

    @staticmethod
    def _calMinDist(x):
        """
        get the smallest non-zero distance of features
        """
        min_dis = []
        for j in range(x.shape[1]):
            xj = x[:,j]
            # drop duplicates
            xj = np.unique(xj)
            # sort
            xj = np.sort(xj)[::-1]
            # distance
            dis = [1]
            for i in range(len(xj)-1):
                dis.append(xj[i] - xj[i+1])
            # min distance
            min_dis.append(np.min(dis) if np.min(dis) else 1)
        return min_dis

    def _setStart(self, x, y, a, c, d, l):
        """
        set warm start from CART
        """
        # train with CART
        if self.min_samples_split > 1:
            clf = tree.DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        else:
            clf = tree.DecisionTreeClassifier(max_depth=self.max_depth)
        clf.fit(x, y)

        # get splitting rules
        rules = self._getRules(clf)

        # fix branch node
        for t in self.b_index:
            # not split
            if rules[t].feat is None or rules[t].feat == tree._tree.TREE_UNDEFINED:
                d[t].start = 0
                for f in range(self.p):
                    a[f,t].start = 0
            # split
            else:
                d[t].start = 1
                for f in range(self.p):
                    if f == int(rules[t].feat):
                        a[f,t].start = 1
                    else:
                        a[f,t].start = 0

        # fix leaf nodes
        for t in self.l_index:
            # terminate early
            if rules[t].value is None:
                l[t].start = int(t % 2)
                # flows go to right
                if t % 2:
                    t_leaf = t
                    while rules[t].value is None:
                        t //= 2
                    for k in self.labels:
                        if k == np.argmax(rules[t].value):
                            c[k, t_leaf].start = 1
                        else:
                            c[k, t_leaf].start = 0
                # nothing in left
                else:
                    for k in self.labels:
                        c[k, t].start = 0
            # terminate at leaf node
            else:
                l[t].start = 1
                for k in self.labels:
                    if k == np.argmax(rules[t].value):
                        c[k, t].start = 1
                    else:
                        c[k, t].start = 0

    def _getRules(self, clf):
        """
        get splitting rules
        """
        # node index map
        node_map = {1:0}
        for t in self.b_index:
            # terminal
            node_map[2*t] = -1
            node_map[2*t+1] = -1
            # left
            l = clf.tree_.children_left[node_map[t]]
            node_map[2*t] = l
            # right
            r = clf.tree_.children_right[node_map[t]]
            node_map[2*t+1] = r

        # rules
        rule = namedtuple('Rules', ('feat', 'threshold', 'value'))
        rules = {}
        # branch nodes
        for t in self.b_index:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(clf.tree_.feature[i], clf.tree_.threshold[i], clf.tree_.value[i,0])
            rules[t] = r
        # leaf nodes
        for t in self.l_index:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(None, None, clf.tree_.value[i,0])
            rules[t] = r

        return rules


