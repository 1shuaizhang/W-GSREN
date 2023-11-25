import numpy as np 
import pandas as pd
from numpy.random import shuffle  
from sklearn.metrics import roc_auc_score

#%%
##example
def Calc_prob(y):
    p = np.exp(5*y) / (1 + np.exp(5*y))
    return p

def pred(predict):
    prediction = []
    for i in predict:
        if i>0.5:
            prediction.append(1)
        else:
            prediction.append(0) 
    return prediction

n_samples,n_features=300,1000 
m = 20
np.random.seed(88) 
X = np.random.normal(size=[n_samples,n_features])
w1 = np.zeros(int(n_features/m))
for i in range(10):
    w1[i] = i+1

siga = np.random.randn(n_samples)#randn生成服从标准正态分布
y =  X[:,:50].dot(w1) + X[:,50:100].dot(w1) + X[:,100:150].dot(w1) + 3*siga
w = np.hstack((np.hstack((w1,w1,w1)),np.zeros(850)))
groups = np.arange(X.shape[1]) // 50
p = Calc_prob(y)
clssify = pred(p)

idx_pos = []
idx_neg = []
for i in range(len(clssify)):
    if clssify[i] ==1:
        idx_pos.append(i)
    else:
        idx_neg.append(i)        
        
X1 = np.vstack((X[idx_pos,:],X[idx_neg,:]))     
"""original dataset [positive samples,negative samples]"""
np.random.seed(1) # shuffle the positive and negative samples respectively
shuffle(X1[:150])
shuffle(X1[150:])
Y = np.hstack((np.ones(152),np.zeros(148))) #generate labels of original dataset 

X_train = X1[30:270] #split training set
Y_train = Y[30:270] 
X_test = np.vstack((X1[:30,:],X1[270:,:])) # slpit independent test set
Y_test = np.hstack((Y[:30],Y[270:])) # label of independent test set 
x_pos = X_train[:120].copy()  # positive samples of training set 
x_neg = X_train[120:].copy() # negative samples of training set 

w2 = np.zeros(int(n_features/m))
for i in range(50):
    if i < 10:
        w2[i] = 0.5
    else:
        w2[i] = 1
    
w21 = np.hstack((np.hstack((w2,w2,w2)),np.ones(850)))
W_i = np.diag(w21)

W_g  = np.zeros(int(m))
for i in range(20):
    if i < 3:
        W_g[i] = (0.5/50**0.5)**(-1)
    else:
        W_g[i] = (0.05/50**0.5)**(-1)
W_g  = list(W_g)

#%%
import math
import time
import numpy as np
import pandas as pd
from numpy import mean
from sklearn import metrics
from numpy.random import shuffle
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score    

from _fista import FISTAProblem
from _singular_values import find_largest_singular_value
from _subsampling import Subsampler
from _group_lasso import GroupLasso

def soft_threshold(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)
'''
def soft_threshold_d(a, b):
    Std = a * soft_threshold(np.linalg.norm(a), b) / np.linalg.norm(a)
    return Std
'''


def soft_threshold_d(a, b):
    Std = []
    for i in range(len(a)):
        if a.all() == 0:
            Std.append(a.all()) 
        else:
            if a[i] == 0:
                Std.append(a[i])
            else:
                Std_ = a[i] * soft_threshold(np.linalg.norm(a[i]), b) / np.linalg.norm(a[i])
                Std.append(Std_ )                        
    return Std

'''
def soft_threshold_d(a, b):
    Std = []
    for i in range(len(a)):
        if a[i] == 0:
            Std.append(a[i]) 
        else:
            Std_ = a[i] * soft_threshold(np.linalg.norm(a[i]), b) / np.linalg.norm(a[i])
            Std.append(Std_ )
    return Std
'''
def select_fea(sgl_coef):
    idx = []
    for i in range(len(sgl_coef)):
        if (sgl_coef[i] != 0):
            idx.append([i])
    return idx



def calc_prob( X, betas):
    power = X.dot(betas)
    p = np.exp(power) / (1 + np.exp(power))
    return p

def pred(predict):
    prediction = []
    for i in predict:
        if i>0.5:
            prediction.append(1)
        else:
            prediction.append(0) 
    return prediction

gl = GroupLasso(groups=np.array(groups),group_reg=0.0055,l1_reg=0.02,frobenius_lipschitz=True,scale_reg="none",
                    subsampling_scheme=None,supress_warning=True,n_iter=1000,tol=1e-3,)
gl.fit(X_train, Y_train)
coefs_train = gl.coef_

def group_square_root_EN(X, y, groups, lambd1, lambd2, W_g,W_i,max_iter=1000, tol=1e-3):
    start_time = time.time()# 记录程序开始运行时间

    beta = np.array(coefs_train).flatten()
    
    w = 2
    n, p = X.shape
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    kesi = np.zeros(X.shape[1], dtype=X.dtype)
    #beta = np.ones(X.shape[1], dtype=X.dtype) / 2
    #beta = np.zeros(X.shape[1], dtype=X.dtype) 
    #beta = np.ones(X.shape[1], dtype=X.dtype) + np.ones(X.shape[1], dtype=X.dtype) / 2

    for _ in range(max_iter):
        perm = np.array(range(len(group_labels)))
        #perm = np.array(len(group_labels))
        beta_old = beta.copy()
        sita = W_i @ beta
        sita_old = sita.copy()
        
        for i in perm:
            group = group_labels[i]
            _X = y - np.dot(X, beta)
            L = np.linalg.norm(_X, ord=2)
            X_j = X[:, group]
            p = W_g[i]
            tmp = 1 + lambd2 * L
            kesi[group] = (1- w) * kesi[group] + w * ((sita[group] + X_j.T @ _X) / tmp)
            beta[group] = soft_threshold_d(kesi[group], lambd1 * p * L / tmp)
            
        #tt = np.linalg.norm(beta - beta_old)
        if np.linalg.norm(beta - beta_old) < tol:
            print(_)
            break
    end_time = time.time()  # 记录程序结束运行时间
    print('Took %f s' % (end_time - start_time))
    return beta


n = 10   #表示5折
m = len(x_pos)//n

lambda1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
           0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.0001,0.0002,0.0003,0.0004,
           0.0005,0.0006,0.0007,0.0008]
lambda1 = [0.061,0.062,0.063,0.064,0.065,0.066,0.067,0.068,0.069,0.07,0.071,0.072,0.073,
           0.074,0.075,0.076,0.077,0.078,0.079,0.08,0.081,0.082,0.083,0.084,0.085,0.086,
           0.087,0.088,0.089,0.09,0.091,0.092,0.093,0.094,0.095,0.096,0.097,0.098,0.1]
lambda1 = [0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.005,0.0055,0.006,0.0065,0.007,0.0075,0.0080,0.085,0.009,0.0095]
lambda1 = [0.061,0.062,0.063,0.064,0.065,0.066,0.067,0.068,0.069,0.07,0.071,
           0.072,0.073,0.074,0.075,0.076,0.077,0.078,0.079,0.08,0.081,0.082,
           0.083,0.084,0.085,0.086,0.087,0.088,0.089]
lambda2 = [200] *18
lambda1 = [0.001]
lambda2 = [200]

lambda1 = np.logspace(-4, -1, 60)
lambda2 = np.full(60,100)


a_l = []
for i in range(len(lambda1)):
    a_l.append([lambda1[i],lambda2[i]])
    
def calc_lambd(x_pos,x_neg,Y_train,m,n,a_l,groups):
    alpha_lambd = []
    Coef_modle = []
    for i in a_l:
        lambd1 = i[0]
        lambd2 = i[1]
        perfor_auc = [] ; 
        coef_val = [];
        # for lambd in lambda_values:    
        ACC = []; AUC = [];
        coef = []; 
        for k in range(n):  #10折,每折总24个样本，正负12个
            x_val = np.vstack((x_pos[k * m:(k+1)*m],x_neg[k*m:(k+1)*m]))
            x_pos_train = np.vstack((x_pos[:k*m], x_pos[(k+1)*m:]))
            x_neg_train = np.vstack((x_neg[:k*m], x_neg[(k+1)*m:]))
            x_train = np.vstack((x_pos_train,x_neg_train)) #训练集，用来训练基学习器,48
            y_val = Y_train[(n-1)*m: (n+1)*m]  #验证集，用于验证基学习器并生成元学习器的数据
            y_train = Y_train[m: (2*n-1)*m]
            K = np.linalg.norm(x_train) / math.sqrt(2)
            X = x_train / K
            y = y_train / K
                
            coefs_train =  group_square_root_EN(X, y, groups, lambd1, lambd2, W_g,W_i,max_iter=1000, tol=1e-3)
            pred_val =  calc_prob(x_val, coefs_train) #计算测试集概率
            predict_val = pred(pred_val) #计算分类为0或1
                
            acc_score = metrics.accuracy_score(y_val, predict_val) #acc
            auc_score = roc_auc_score(y_val, pred_val)
            coef.append(coefs_train) #每个lambda对应的五折训练结果
            ACC.append(acc_score)
            AUC.append(auc_score)
          
        coef_val = coef_val + coef  #保留所有lambda训练结果                   
        perfor_auc.append([np.mean(AUC),np.mean(ACC),lambd2]) #每次五折验证集auc的平均值
        Coef_modle.append(coef_val)  #选出平均acc最高的lambda对应的结果
        alpha_lambd.append([lambd1, perfor_auc[0][2], perfor_auc[0][0],perfor_auc[0][1]]) 
    return Coef_modle,alpha_lambd
C_M,A_L = calc_lambd(x_pos,x_neg,Y_train,m,n,a_l,groups)
a1,a2, = C_M,A_L


a3 = a1[0]
beta0 = a3[0]
beta1 = a3[1]
beta2 = a3[2]
beta3 = a3[3]
beta4 = a3[4]

index0 = select_fea(beta0)
index1 = select_fea(beta1)
index2 = select_fea(beta2)
index3 = select_fea(beta3)
index4 = select_fea(beta4)

K = np.linalg.norm(X_train) / math.sqrt(2)
X = X_train / K
y = Y_train / K

a_l2 = []
for i in a_l:
    ab = group_square_root_EN(X, y, groups,  i[0], i[1], W_g, W_i, max_iter=1000, tol=1e-3)

    index = select_fea(ab)
    predict = calc_prob( X_test, ab)
    y_pred = pred(predict)
    acc_score = metrics.accuracy_score(Y_test,y_pred)
    auc_score = roc_auc_score(Y_test, predict)
    a_l2.append([i[0],i[1],auc_score,acc_score])
    

#100次重复实验

#100次重复实验
ACC = []
AUC = []
for i in range(100):
    ab = group_square_root_EN(X, y, groups,  0.0005150678076168122, 100, W_g, W_i, max_iter=1000, tol=1e-3)
    predict = calc_prob( X_test, ab)
    y_pred = pred(predict)
    acc_score = metrics.accuracy_score(Y_test,y_pred)
    auc_score = roc_auc_score(Y_test, predict)
    ACC.append(acc_score)
    AUC.append(auc_score)

#FNR,FDR,F1
ab = group_square_root_EN(X, y, groups,  0.00022695105366946685, 100, W_g, W_i, max_iter=1000, tol=1e-3)
predict = calc_prob( X_test, ab)
y_pred = pred(predict)

cm=confusion_matrix(Y_test,y_pred)
TP=cm[0,0]
FN=cm[0,1]
FP=cm[1,0]
TN=cm[1,1]

FNR = FN/(TP+FN)
FDR = FP/(TP+FP)
F1_score=(2*TP)/(2*TP+FP+FN)

print("FNR:",FNR)
print("FDR:",FDR)
#print("recall:", recall)
print("f1_score:", F1_score)
