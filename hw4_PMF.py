import sys
import math
import numpy as np
import pandas as pd

ratings=(pd.read_csv(sys.argv[1],header=None))
ratings.columns=['user','item','ratings']
data=ratings.pivot(index='user',columns='item')['ratings']
R=np.matrix(data)

def find_cost(R1,P,Q,sigma2=0.1,lamda=2):
    cost=-(1/(2*sigma2))*(np.sum(np.square((R1-P*Q))))-(lamda/2.0)*(np.sum(np.square(P))+np.sum(np.square(Q)))
    return cost
  
def Update_P(R1,P,Q,K,sigma2=0.1,lamda=2):
    U=(((((lamda*sigma2) *np.matrix(np.identity(K)))+ (Q*Q.T)).I )*(R1*Q.T).T).T
    return U
  
def Update_Q(R1,P,Q,K,sigma2=0.1,lamda=2):
    V=((((lamda*sigma2) *np.matrix(np.identity(K)))+ (P.T*P)).I)*(P.T*R1)
    return V
   
K=5
R1=np.matrix(pd.DataFrame(R.copy()).fillna(0))
P = np.matrix(np.random.rand(R.shape[0],K))
Q = np.matrix(np.random.rand(K,R.shape[1]))

objective=[]
csv_iter=[9,24,49]

for iter in range(0,50):
 print('iter:'+str(iter))
 C=find_cost(R1,P,Q)
 objective.append(C)
 P=Update_P(R1,P,Q,K)
 Q=Update_Q(R1,P,Q,K)
 if iter in csv_iter:
  Filename1='U-'+str(int(math.floor(iter+1)))+'.csv'
  Filename2='V-'+str(int(math.floor(iter+1)))+'.csv'
  U=pd.DataFrame(P)
  U.to_csv(Filename1,header=False,index=False)
  V=pd.DataFrame(Q.T)
  V.to_csv(Filename2,header=False,index=False)

obj=pd.DataFrame(objective)
Filename3='objective.csv'
obj.to_csv(Filename3,header=False,index=False)
  

 
 
 