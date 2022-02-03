#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import tensorflow as tf
import numpy as np


# In[2]:


def ade(predAll,targetAll,count_):

    All = len(predAll)
    sum_all = 0 
    for s in range(All): 
        pred = np.swapaxes(predAll[s][:,:count_[s],:], 0 ,1) 
        target = np.swapaxes(targetAll[s][:,:count_[s],:], 0, 1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                sum_+=np.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
        
    return sum_all/All


# In[3]:


def fde(predAll,targetAll,count_):
    
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=np.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All


# In[4]:


def seq_to_nodes(seq_,max_nodes = 88):
    """

    """
    seq_ = np.squeeze(seq_)  
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return np.squeeze(V)


# In[5]:


def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return np.squeeze(nodes_)


# In[6]:


def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False


# In[7]:


def bivariate_loss(V_pred,V_trgt):

    normx = V_trgt[:,:,0]- V_pred[:,:,0]
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = tf.math.exp(V_pred[:,:,2]) #sx
    sy = tf.math.exp(V_pred[:,:,3]) #sy
    corr = tf.math.tanh(V_pred[:,:,4]) #corr
    
    sxsy = sx * sy

    # Numerator
    result = tf.math.exp(-((normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy))/(2*(1 - corr**2)))
    # Normalization factor
    denom = 2 * math.pi * (sxsy * tf.math.sqrt((1 - corr**2)))

    
    # Final PDF calculation
    result = result / denom

    epsilon = 1e-20
    epsilonMax = 1e200

    result = -tf.math.log(tf.clip_by_value(result, clip_value_min = epsilon, clip_value_max = epsilonMax))
    result = tf.math.reduce_mean(result)
    
    return result

