from __future__ import division
import sys
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import multivariate_normal as MV_N


X=(pd.read_csv(sys.argv[1],header=None))
X=X.fillna(X.mean())

## K-Means Clustering
def Find_Best_Cluster(X,k,init):
    N=len(X)
    Item_Class={}
    for row in range(0,N):
        List=[]
        for cls in range(0,k):
            Dist=distance.euclidean(X.loc[row],init.ix[cls])
            List.append(Dist)
        p=np.argmin(List)
        Item_Class[row]=p
    return Item_Class 
   
def Find_Centroids(X,k,number):
    N=len(X)
    init=X.sample(k).reset_index(drop=True)
    for iter in range(0,number):
        Cluster=Find_Best_Cluster(X,k,init)
        Data=pd.DataFrame.from_dict(Cluster,orient='index')
        X['class']=Data[0]
        init=X.groupby('class').mean()
        del X['class']
        Filename='centroids-'+ str(int(math.floor(iter))+1)+'.csv'
        init.to_csv(Filename,header=False,index=False)
        
Find_Centroids(X,5,10)

### EM-GMM Clustering
#X=(pd.read_csv('Data.csv',header=None))
#X.to_csv('/home/ccc_v1_w_daf2d_16855/asn12839_Clustering_ML_T_/asn12840_Clustering/work/Data.csv',header=False,index=False)                                                          
#missing=sum(X.isnull().values.ravel())
#print('missing:'+str(missing))
#X=X.fillna(X.mean())
#missing2=sum(X.isnull().values.ravel())
#print('missing2:'+str(missing2))
#def print_full(x):
    #pd.set_option('display.max_rows', len(x))
    #print(x)
    # pd.reset_option('display.max_rows')
    
#print_full(X)

X=(pd.read_csv(sys.argv[1],header=None))
X=X.fillna(X.mean())



### EM-GMM Clustering
#X=(pd.read_csv('Data.csv',header=None))
#X.to_csv('/home/ccc_v1_w_daf2d_16855/asn12839_Clustering_ML_T_/asn12840_Clustering/work/Data.csv',header=False,index=False)                                                          
#missing=sum(X.isnull().values.ravel())
#print('missing:'+str(missing))
#X=X.fillna(X.mean())
#missing2=sum(X.isnull().values.ravel())
#print('missing2:'+str(missing2))
#def print_full(x):
    #pd.set_option('display.max_rows', len(x))
    #print(x)
    # pd.reset_option('display.max_rows')
    
#print_full(X)

def dot_product(v,X):
    M=0
    X=X.fillna(1)
    for i in range (0,len(v)):
        M=M+v[i]*X.loc[i]
    return M

def Initialize(X,k):
    d=X.shape[1]
    mu_0=X.sample(k).reset_index(drop=True)
    #mu_0=pd.DataFrame(np.random.rand(k,d))
    sigma0=X.cov()
    prior=[1/k]*k
    columns=np.arange(k)
    df =pd.DataFrame(index=X.index,columns=columns)
    df= df.fillna(1/k)
    for row in range(0,len(X)):
        L=[]
        for i in range(0,k):
            g=MV_N.pdf(X.loc[row],mu_0.loc[i],sigma0)*prior[i]
            L.append(g)
        W = [float(i)/sum(L) for i in L]
        df.loc[row]=W
    return df,mu_0
   
def Maximum(X,k,df):
    prior=df.mean()
    e=np.exp(-100)
    #mu=Initialize(X,k)[1]
    #mu = pd.DataFrame(np.empty((k,X.shape[1],)))
    s=(k,X.shape[1])
    print('shape'+str(s))
    mu=pd.DataFrame(np.ones(s))
    #print(mu)
    for i in range(0,k):
        #mu.loc[i]=np.dot(df[i],X)/(sum(df[i])+e)
        #L=dot_product(df[i],X)
        df=df.fillna(0)
        X=X.fillna(X.mean())
        L=np.matrix(df[i].T)*np.matrix(X)
        print('L:'+str(L))
        mu.loc[i]=L/sum(df[i]+e)
        mu=mu.fillna(1)
    
    D={}
    for cls in range(0,k):
        M=0
        for i in range(0,len(X)):
            M=M+df.loc[i][cls]*(np.matrix(X.loc[i]-mu.loc[cls]).T * np.matrix(X.loc[i]-mu.loc[cls]))
        D[cls]=M
    
    return prior,mu,D

def pseudoinverse(S):
    return (S.T*S).I* S.T 

def MV_Gaussian(x,mu,cov):
    d=np.matrix(x).shape[1]
    e=np.exp(-100)
    M=(np.matrix(x-mu)) * (np.linalg.pinv(cov)) * (np.matrix((x-mu)).T)
    Den=((2*np.pi)**(d/2.0)) * (np.sqrt(abs(np.linalg.det(cov))))
    Den=Den+e
    p=((1/Den)*np.exp(-0.5*M))
    pdf=np.linalg.det(p)
    return pdf
k=5
number=10
df=Initialize(X,k)[0]
print(df)
for iter in range(0,number):
    print('iter:'+str(iter))
    prior=Maximum(X,k,df)[0]
    prior_df=pd.DataFrame(prior)
    FileName1='pi-'+str(int(math.floor(iter))+1)+'.csv'
    prior_df.to_csv(FileName1,header=False,index=False)
    mu=Maximum(X,k,df)[1]
    #print('mu:'+str(mu))
    FileName2='mu-'+str(int(math.floor(iter))+1)+'.csv'
    mu.to_csv(FileName2,header=False,index=False)
    print('mu:')
    print(mu)
    sigma=Maximum(X,k,df)[2]
    #print('sigma value:')
    #print(sigma)
    for cluster in range(0,len(sigma)):
        # print('cluster:'+str(cluster))
        data=pd.DataFrame(sigma[cluster])
        FileName3='Sigma-'+ str(int(math.floor(cluster))+1)+'-'+ str(int(math.floor(iter))+1)+'.csv'
        data.to_csv(FileName3,header=False,index=False)
        for row in range(0,len(X)):
            L=[]
            for i in range(0,k):
                # print ('Cluster:'+str(i))
                #g= MV_N.pdf(X.loc[row],mu.loc[i],sigma[i])* prior[i]
                g=MV_Gaussian(X.loc[row],mu.loc[i],sigma[i])* prior[i] 
                L.append(g)
            W = [i/sum(L) for i in L]
            #print('phi:-'+str(W))
        df.loc[row]=W
