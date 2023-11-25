# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 22:26:12 2023

@author: admin
"""
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
y =  X[:,:50].dot(w1) + X[:,50:100].dot(w1) + X[:,100:150].dot(w1) + 1*siga
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
shuffle(X1[:149])
shuffle(X1[149:])
Y = np.hstack((np.ones(149),np.zeros(151))) #generate labels of original dataset 

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
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from numpy.random import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.linear_model import ElasticNet


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

def select_fea(sgl_coef):
    idx = []
    for i in range(len(sgl_coef)):
        if (sgl_coef[i] != 0):
            idx.append([i])
    return idx


def Extract(num, a):
    coef = np.maximum(a, -a)
    coef = list(coef)
    c_sorted = sorted(enumerate(coef), key=lambda x: x[1])
    c_list = []
    for i in range(len(c_sorted)):
        x = list(c_sorted[i])
        c_list.append(x)
    s_g = c_list[-num:]
    s = [i[0] for i in s_g]
    gene_select = []
    for i in s:
        gene_select.append(gene_name[i])  # lasso选出基因名
    return s, gene_select  

n = 10   #表示5折
m = len(x_pos)//n
alpha  = np.logspace(-3, -1, 60)
lambda1 = np.full(60,30)
a_l = []
for i in range(len(alpha)):
    a_l.append([alpha[i],lambda1[i]])   
    
def calc_lambd(x_pos,x_neg,Y_train,m,n,a_l,groups):
    alpha_lambd = []
    Coef_modle = []
    for i in a_l:
        alpha = i[0]
        lambd1 = i[1]
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

            start_time = time.time()# 记录程序开始运行时间
            reg = ElasticNet(lambd1,alpha, fit_intercept=False)
            reg.fit(x_train, y_train)
            coefs_train = reg.coef_
            end_time = time.time()  # 记录程序结束运行时间 
            print('Took %f s' % (end_time - start_time))

            
            pred_val =  calc_prob(x_val, coefs_train) #计算测试集概率
            predict_val = pred(pred_val) #计算分类为0或1
                
            acc_score = metrics.accuracy_score(y_val, predict_val) #acc
            auc_score = roc_auc_score(y_val, pred_val)
            coef.append(coefs_train) #每个lambda对应的五折训练结果
            ACC.append(acc_score)
            AUC.append(auc_score)
          
        coef_val = coef_val + coef  #保留所有lambda训练结果                   
        perfor_auc.append([np.mean(AUC),np.mean(ACC),alpha]) #每次五折验证集auc的平均值
        Coef_modle.append(coef_val)  #选出平均acc最高的lambda对应的结果
        alpha_lambd.append([lambd1, perfor_auc[0][2], perfor_auc[0][0],perfor_auc[0][1]]) 
    return Coef_modle,alpha_lambd

C_M,A_L = calc_lambd(x_pos,x_neg,Y_train,m,n,a_l,groups)

a,a1 = C_M,A_L

a_l2 = []
for i in a_l:
    reg = ElasticNet(i[1],i[0], fit_intercept=False)
    reg.fit(X_train, Y_train)
    coefs_train = reg.coef_

    ab = coefs_train
    index = select_fea(ab)
    predict = calc_prob( X_test, coefs_train)
    y_pred = pred(predict)
    acc_score = metrics.accuracy_score(Y_test,y_pred)
    auc_score = roc_auc_score(Y_test, predict)
    a_l2.append([i[0],i[1],auc_score,acc_score])

#100次重复实验
acc = []
auc = []
for i in range(100):
    reg = ElasticNet(30,0.002018760254679039, fit_intercept=False)
    reg.fit(X_train, Y_train)
    ab = reg.coef_
    predict = calc_prob( X_test, ab)
    y_pred = pred(predict)
    acc_score = metrics.accuracy_score(Y_test,y_pred)
    auc_score = roc_auc_score(Y_test, predict)
    acc.append(acc_score)
    auc.append(auc_score)

#FNR,FDR,F1
reg = ElasticNet(30,0.002018760254679039, fit_intercept=False)
reg.fit(X_train, Y_train)
ab = reg.coef_
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
