#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from sklearn.model_selection import KFold
import random    
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


# In[14]:


def create_u_v(m):
    u = np.full((m.shape[0],2), 1)
    v = np.full((2,m.shape[1]), 1)
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    return u,v


# In[15]:


def normalize_data(m): 
    Row_df = m.pivot(index = 'user_id', columns ='movie_id', values = 'rating')
    u_mean = Row_df.mean(axis=1)
    Row_df_array = Row_df.to_numpy()
    u_mean = u_mean.to_numpy()
    #creating a normal matrix to compare to our uv matrix
    normal = Row_df_array - u_mean.reshape(-1,1)
    N = normal
    return N,Row_df


# In[23]:


# updating u using the formula x =(Σj vsj (mrj−Σk̸=surkvkj))/Σjv^2sj

def update_u(u,v,N):
    sums = 0
    u_rk = u[r,:]
    v_kj = v[:,:]

    #to calculate the part of the matrices not affected by the value at index r 
    u_rk_del = np.delete(u_rk, s, 0)
    v_kj_del = np.delete(v_kj, s, 0)
    v_sj = v[s,:]
    v_sj_squared = v_sj ** 2       
    #create the matrix combination of u and v which would be subtracted from original matrix for error
    u_rk_v_kj = np.dot(u_rk_del, v_kj_del)
    m_rj = N[r,:]
    error = m_rj - u_rk_v_kj
    vsj_dot_er = v_sj * error
    sums = np.nansum(vsj_dot_er)
    v_sj_ssum = np.nansum((v_sj_squared) * (~np.isnan(m_rj)))
    newval_u = sums / v_sj_ssum
    u[r,s] = u[r,s] + ((newval_u - u[r,s]))
    return u,v
    


# In[24]:



#update v using the formula y = (Σiuir(mis−Σk̸=ruikvks))/Σiu^2ir
def update_v(u,v,N):
    sums = 0
    u_ik = u[:,:]
    v_ks = v[:,s]
    u_ik_del = np.delete(u_ik, r, 1)
    v_ks_del = np.delete(v_ks, r, 0)
    u_ir = u[:,r]
    u_ir_squared = u_ir ** 2
    u_ik_v_ks = np.dot(u_ik_del, v_ks_del)
    m_is = N[:,s]
    error = m_is - u_ik_v_ks
    uir_dot_er = u_ir * error
    sumsv = np.nansum(uir_dot_er)
    u_ir_ssum = np.nansum(u_ir_squared * (~np.isnan(m_is)))
    newval_v =  sumsv / u_ir_ssum
    v[r,s] = v[r,s] + ((newval_v - v[r,s]))
    return u,v


# In[25]:


def mae(dif):
    dif_abs= (np.absolute(dif))
        #converting all nan values to a zero value.
    dif_abs_0s = np.nan_to_num(dif_abs)
    dif_abs_sum = np.sum(dif_abs_0s,axis=0)
    sum_dif = dif_abs_sum.sum()
    non_0_count = np.count_nonzero(dif_abs_0s)
    MAE=sum_dif/non_0_count
    return MAE


# In[26]:


def rmse(dif):
    dif_sqr = dif ** 2
    dif_sqr_0s = np.nan_to_num(dif_sqr)
    dif_sqr_total= np.sum( dif_sqr_0s ,axis=0)
    sumz = dif_sqr_total.sum()
    non_0_count_sqr = np.count_nonzero( dif_sqr_0s )
    RMSE = sumz/ non_0_count_sqr
    return RMSE


# In[45]:


# UV Decomposition - Training

#input the path of ratings.dat file
RT = pd.read_csv('ratings.dat', engine='python', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])

#create a kfold function to divide the data into 5 random sets for cross validation
KF = KFold(n_splits=5, shuffle=True, random_state=9)
c = 2
i = 5
#start the iteration for each of the 5 folds
for train_index, test_index in KF.split(RT):
    RT_train, RT_test = RT.loc[train_index], RT.loc[test_index]
    #create a dataframe to store all ratings as values for each movie in a coloumn with every user id as index of the rows.
    normal,Row_df = normalize_data(RT_train)
    N = normal
    Row_df_array = Row_df.to_numpy()
    #creating uv matrix components with u having n X d and v having d X m ( where n = number of users, m = number of movies and d = 2)
    u,v = create_u_v(normal)
    uv = np.dot(u,v)
    print("Index:", train_index)
    for iterations in range(i):
        for r in range(6040):
            for s in range(c):
                u,v = update_u(u,v,N)
        for r in range(c):
            for s in range(Row_df_array.shape[1]):
                u,v = update_v(u,v,N)
        uv = np.dot(u,v)
        dif = uv-normal
        print("Iteration Number: ",iterations )
        MAE = mae(dif)
        print('MAE %.4f : ' %MAE)
        #calculating RMSE
        RMSE = rmse(dif)
        print('RMSE %.4f : '%RMSE)   


# In[44]:


# UV Decomposition - Test

#input the path of ratings.dat file
RT = pd.read_csv('ratings.dat', engine='python', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])

#create a kfold function to divide the data into 5 random sets for cross validation
KF = KFold(n_splits=5, shuffle=True, random_state=9)
c = 2
i = 5

#start the iteration for each of the 5 folds
for train_index, test_index in KF.split(RT):
    RT_train, RT_test = RT.loc[train_index], RT.loc[test_index]
    #create a dataframe to store all ratings as values for each movie in a coloumn with every user id as index of the rows.
    normal,Row_df = normalize_data(RT_test)
    N = normal
    Row_df_array = Row_df.to_numpy()
    #creating uv matrix components with u having n X d and v having d X m ( where n = number of users, m = number of movies and d = 2)
    u,v = create_u_v(normal)
    uv = np.dot(u,v)
    print("Index:", test_index)
  # updating u using the formula x =(Σj vsj (mrj−Σk̸=surkvkj))/Σjv^2sj
    for iterations in range(i):
        for r in range(1510):
            for s in range(c):
                u,v = update_u(u,v,N)
        #update v using the formula y = (Σiuir(mis−Σk̸=ruikvks))/Σiu^2ir
        for r in range(c):
            for s in range(Row_df_array.shape[1]):
                u,v = update_v(u,v,N)
        uv = np.dot(u,v)
        dif = uv-normal
        print("Iteration Number: ",iterations )
        MAE = mae(dif)
        print('MAE %.4f : ' %MAE)
        #calculating RMSE
        RMSE = rmse(dif)
        print('RMSE %.4f : '%RMSE)


# In[ ]:




