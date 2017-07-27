# -*- coding: utf-8 -*-
"""
Spyder Editor
Author--KINGSHUK BOSE
This is a temporary script file.
"""

#-Part I, Ridge Regression

import sys
import math
import numpy as np
import pandas as pd


X=np.matrix(pd.read_csv(sys.argv[3],header=None))
Y=np.matrix(pd.read_csv(sys.argv[4],header=None))
lamda=float(sys.argv[1])



def Ridge(X,Y,lamda):
    In=np.identity(np.shape(X)[1])
    W=(((X.T * X)+lamda*In).I)*X.T*Y
    return W

W=Ridge(X,Y,lamda)
#W=np.around(W,2)
W_D=pd.DataFrame(W)

Filename='wRR_'+ str(int(math.floor(lamda))) +'.csv'

W_D.to_csv(Filename,header=False,index=False)


#Part II, Active Learning

X=np.matrix(pd.read_csv(sys.argv[3],header=None))
Y=np.matrix(pd.read_csv(sys.argv[4],header=None))
Test=np.matrix(pd.read_csv(sys.argv[5],header=None))
lamda=float(sys.argv[1])
sigma=float(sys.argv[2])

def Find_Best_Sigma(X_train,Y_train,lamda,sigma,X_Test):
    In=np.identity(np.shape(X_train)[1])
    Var=(lamda*In+ sigma**(-2) * np.dot(X_train.T,X_train)).I
    D={}   
    for i in range(0,np.shape(X_Test)[0]):
        sigma20=sigma**2+ np.dot(X_Test[i],np.dot(Var,X_Test[i].T))
        #Var=(Var.I +  sigma**(-2) * np.dot(X_Test[i],X_Test[i].T)).I
        D[i]=sigma20
    X=max(D, key=lambda i: D[i])
    return X
   
   
def return_index(data,vector):
    for i in range(0,np.shape(data)[0]):
        if np.array_equal(data[i],vector):
            index=i
            break
    return index
   
def Active_Learning(X_train,Y_train,lamda,sigma,X_Test):
  Test=X_Test
  L=np.shape(X_Test)[0]
  List=[]
  while len(List)< min(L,10):
   index=Find_Best_Sigma(X_train,Y_train,lamda,sigma,X_Test)
   org_index=return_index(Test,X_Test[index])
   X_train=np.append(X_train,X_Test[index],axis=0)
   X_Test=np.delete(X_Test,index,axis=0)
   List.append(org_index+1)
  return List

Active_List=Active_Learning(X,Y,lamda,sigma,Test) 
Ac_D=pd.DataFrame(np.matrix(Active_List))
Filename2='active_'+ str(int(math.floor(lamda)))+'_'+str(int(math.floor(sigma))) +'.csv'
Ac_D.to_csv(Filename2,header=False,index=False)