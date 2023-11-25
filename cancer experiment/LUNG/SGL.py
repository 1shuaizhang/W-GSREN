# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:14:47 2023

@author: admin
"""
import math
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from numpy.random import shuffle
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from _fista import FISTAProblem
from _singular_values import find_largest_singular_value
from _subsampling import Subsampler
from _group_lasso import GroupLasso


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

def read_Lung():    
    data = pd.read_csv('F:/张帅代码/LUNG/LUNG2.csv', sep=",", index_col=0) 
    gene_name = data.index.tolist()
    data_scaler = preprocessing.scale(np.array(data).T) #标准化后的数据
    y = np.hstack((np.ones(500),np.zeros(500))) 
    np.random.seed(2) #1
    np.random.shuffle(data_scaler[:500])  #打乱正样本
    np.random.shuffle(data_scaler[500:])

    X_train = data_scaler[150:850,:]
    Y_train = y[150:850]
    X_test = np.vstack((data_scaler[:150,:],data_scaler[850:,:])) #独立测试集
    Y_test = np.hstack((y[:150],y[850:]))
    x_pos = X_train[:350].copy()  #打乱正负样本，正样本30，负样本30
    np.random.seed(1) #1
    shuffle(x_pos)
    x_neg = X_train[350:].copy()
    np.random.seed(1) #2
    np.random.shuffle(x_neg)
    
    W_i = pd.read_csv('F:/张帅代码/LUNG/LUNG_W2.csv', sep=",", index_col=0)
    W_i = W_i['P'].tolist()
    W_i = np.diag(W_i)

    set_size = pd.read_csv("F:/张帅代码/LUNG/LUNG_group2.csv",sep = ",") 
    s = set_size['number of gene'].tolist()
    g = set_size['cor'].tolist()
    W_g = []
    groups = []
    for i in range(len(s)):
        a = [i] * int(s[i])
        ab = (g[i]/s[i]**0.5)**(-1)
        groups += a
        W_g.append(ab)
    return X_train,x_pos, x_neg, X_test, Y_train,Y_test, groups,gene_name,W_i,W_g 

X_train,x_pos, x_neg, X_test, Y_train,Y_test, groups,gene_name,W_i,W_g  = read_Lung()



n = 10   #表示5折
m = len(x_pos)//n

alpha = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,
                0.7,0.75,0.8,0.85,0.9,0.95]
lambda1 = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
                 0.02,0.02,0.02,0.03,0.04]

alpha = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,
                0.7,0.75,0.8,0.85,0.9,0.95]
lambda1 = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
                 0.02,0.02,0.02,0.03,0.04]
alpha  = np.logspace(-4, -2, 30)
lambda1 = np.full(30,0.01)
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
            gl = GroupLasso(groups=groups,group_reg= lambd1,l1_reg=alpha,frobenius_lipschitz=True,scale_reg="none",
                            subsampling_scheme=None,supress_warning=True,n_iter=1000,tol=1e-3,)
            a = gl.fit(x_train, y_train)
            coefs_train = a.coef_
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
        alpha_lambd.append([perfor_auc[0][2],lambd1,  perfor_auc[0][0],perfor_auc[0][1]]) 
    return Coef_modle,alpha_lambd

C_M,A_L = calc_lambd(x_pos,x_neg,Y_train,m,n,a_l,groups)

a,a1 = C_M,A_L


a_l2 = []
for i in a_l:
    gl = GroupLasso(groups=groups,group_reg=i[1],l1_reg=i[0],frobenius_lipschitz=True,scale_reg="none",
                subsampling_scheme=None,supress_warning=True,n_iter=1000,tol=1e-3,)
    gl.fit(X_train, Y_train)
    coefs_train = gl.coef_


    ab = coefs_train
    index = select_fea(ab)
    predict = calc_prob( X_test, coefs_train)
    y_pred = pred(predict)
    acc_score = metrics.accuracy_score(Y_test,y_pred)
    auc_score = roc_auc_score(Y_test, predict)
    a_l2.append([i[0],i[1],auc_score,acc_score])



gl = GroupLasso(groups=groups,group_reg=0.01,l1_reg=0.008531678524172805,frobenius_lipschitz=True,scale_reg="none",
                subsampling_scheme=None,supress_warning=True,n_iter=1000,tol=1e-3,)
gl.fit(X_train, Y_train)
coefs_train = gl.coef_

ab = coefs_train
index = select_fea(ab)
s, gene_select = Extract(100, ab)

predict = calc_prob( X_test, coefs_train)
y_pred = pred(predict)
acc_score = metrics.accuracy_score(Y_test,y_pred)
auc_score = roc_auc_score(Y_test, predict)




'''
#lasso五倍交叉验证
def calc_lambd(x_pos,x_neg,Y_train,m,n,a_l,groups):
    alpha_lambd = []
    Coef_modle = []
    for i in range(len(a_l)):
        lambd = a_l[i]
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
            gl = GroupLasso(groups=groups,group_reg=lambd ,l1_reg=0,frobenius_lipschitz=True,scale_reg="none",
                            subsampling_scheme=None,supress_warning=True,n_iter=1000,tol=1e-3,)
            a = gl.fit(x_train, y_train)
            coefs_train = a.coef_
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
        perfor_auc.append([np.mean(AUC),np.mean(ACC)]) #每次五折验证集auc的平均值
        Coef_modle.append(coef_val)  #选出平均acc最高的lambda对应的结果
        alpha_lambd.append([lambd,  perfor_auc[0][0],perfor_auc[0][1]])
    
    return Coef_modle,alpha_lambd
'''