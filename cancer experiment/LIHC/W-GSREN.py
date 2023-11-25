# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:57:33 2023

@author: admin
"""
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

def read_LIHC():
    data = pd.read_csv("F:/张帅代码/LIHC/LIHC.csv", sep=",", index_col=0)
    data = data.T
    gene_name = data.index.tolist()
    data_scaler = preprocessing.scale(np.array(data).T) #标准化后的数据
    y = np.hstack((np.ones(50),np.zeros(50))) #50个正样本，41个负样本
    np.random.seed(2)
    np.random.shuffle(data_scaler[:50])  #打乱正样本
    np.random.shuffle(data_scaler[50:])
    
    X_train = data_scaler[20:80,:]
    Y_train = y[20:80]
    X_test = np.vstack((data_scaler[:20,:],data_scaler[80:,:])) #独立测试集
    Y_test = np.hstack((y[:20],y[80:]))
    x_pos = X_train[:30].copy()  #打乱正负样本，正样本30，负样本30
    np.random.seed(1)
    shuffle(x_pos)
    x_neg = X_train[30:].copy()
    np.random.seed(1)
    np.random.shuffle(x_neg)
    
    W_i = pd.read_csv('F:/张帅代码/LIHC/LIHC_W.csv', sep=",", index_col=0)
    W_i = W_i['P'].tolist()
    W_i = np.diag(W_i)
    
    
    set_size = pd.read_csv("F:/张帅代码/LIHC/LIHC_group.csv",sep = ",") 
    s = set_size['number of gene'].tolist()
    g = set_size['cor'].tolist()
    W_g = []
    groups = []
    for i in range(len(s)):
        a = [i] * int(s[i])
        ab = (g[i]/s[i]**0.5)**(-1)
        groups += a
        W_g.append(ab)
    return X_train,x_pos, x_neg, X_test, Y_train,Y_test, groups,gene_name,W_g,W_i

X_train,x_pos,x_neg, X_test,Y_train,Y_test, groups,gene_name,W_g,W_i = read_LIHC()

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

lambda1 = [0.001]
lambda2 = [200]
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

mybeta = group_square_root_EN(X, y, groups, 0.001,200, W_g, W_i, max_iter=1000, tol=1e-3)

a = mybeta
index = select_fea(a)
s, gene_select = Extract(100, a)



