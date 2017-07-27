#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 00:13:10 2017

@author: KINGSHUK
"""
from __future__ import division
import sys
import decimal
import numpy as np
import pandas as pd
#import math
from scipy.stats import itemfreq



X_train=np.matrix(pd.read_csv(sys.argv[1],header=None))
Y_train=np.array(pd.read_csv(sys.argv[2],header=None))
X_test=np.array(pd.read_csv(sys.argv[3],header=None))

"""

def P_C_given_D(x,mu,sigma2):
    p= (1/(np.sqrt(2*np.pi*sigma2))) * (np.exp((-(x-mu)**2)/(2*sigma2)))
    return p

def Naive_Bayes(X_train,Y_train,X_test):
    IF=itemfreq(Y_train)
    N=len(Y_train)
    Dict={}
    for i in range(0,len(IF)):
        Dict[i]=IF[i][1]/N
    D=pd.DataFrame(np.concatenate((X_train,Y_train),axis=1))
    mu_table=D.groupby(D.shape[1]-1).mean()
    sigma_table=D.groupby(D.shape[1]-1).var()
    
    shape=(X_test.shape[0],len(Dict))
    a = np.empty(shape,dtype=decimal.Decimal)

    for n in range(0,X_test.shape[0]):
        prob_dict={}
        for i in range(0,len(Dict)):
            M=Dict[i]
            for col in range(0,D.shape[1]-1):
                M=M*P_C_given_D(X_test[n][col],mu_table.loc[i][col],sigma_table.loc[i][col])
            prob_dict[i]=M
        a[n]=np.array(list(prob_dict.values()))
    probs=pd.DataFrame(a)
    probs_N=probs.div(probs.sum(axis=1), axis=0)
    return probs_N

"""   

def Naive_Bayes_QDA(X_train,Y_train,X_test):
    IF=itemfreq(Y_train)
    N=len(Y_train)
    Dict={}
    for i in range(0,len(IF)):
        Dict[i]=IF[i][1]/N
    D=pd.DataFrame(np.concatenate((X_train,Y_train),axis=1))
    mu=D.groupby(D.shape[1]-1).mean()
    cov=D.groupby(D.shape[1]-1).cov()
    shape=(X_test.shape[0],len(Dict))
    a = np.empty(shape,dtype=decimal.Decimal)
    d=D.shape[1]-1
    for n in range(0,X_test.shape[0]):
        prob_dict={}
        for i in range(0,len(Dict)):
            M=(X_test[n]-np.matrix(mu.loc[i])) * np.matrix(cov.loc[i]).I * (X_test[n]-np.matrix(mu.loc[i])).T
            Den=((2*np.pi)**(d/2.0)) * (np.sqrt(abs(np.linalg.det (cov.loc[i]))))
            P=(1/Den)*np.exp(-0.5*M)
            prb= P*Dict[i]
            prob_val=np.linalg.det(prb)
            prob_dict[i]=prob_val
        a[n]=np.array(list(prob_dict.values()))
    probs_mv=pd.DataFrame(a)
    probs_mv_N=probs_mv.div(probs_mv.sum(axis=1), axis=0)
    return probs_mv_N

#probs=Naive_Bayes(X_train,Y_train,X_test)

probs_QDA=Naive_Bayes_QDA(X_train,Y_train,X_test)

#probs.to_csv('probs_test.csv',header=False,index=False)

probs_QDA.to_csv('probs_test.csv',header=False,index=False)
"""
X_train=np.array(pd.read_csv(sys.argv[1],header=None))
Y_train=np.array(pd.read_csv(sys.argv[2],header=None))
X_test=np.array(pd.read_csv(sys.argv[3],header=None))


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train,Y_train)
probs_scikit=clf.predict_proba(X_test)
probs_sci=pd.DataFrame(probs_scikit)
probs_sci_N=probs_sci.div(probs_sci.sum(axis=1), axis=0)


probs_sci_N.to_csv('probs_test.csv',header=False,index=False)
"""

