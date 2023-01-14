#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:37:20 2021

@author: cxy
"""

from __future__ import division, print_function, absolute_import

import os
import h5py
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
tf.disable_eager_execution()
#tf.disable_v2_behavior()
import tensorflow as tf2
import scipy.io as sio
import gc
import re
from scipy import io

os.environ['CUDA_VISIBLE_DEVICES']='0'
config = tf.ConfigProto(allow_soft_placement = True)
#config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
config.gpu_options.allocator_type = 'BFC'

## Learning rate
starter_learning_rate = 0.0001
epoch = 100

# Network Parameters
Height = 64
Width = 64
batch_size = 64
K = 32 # number of filters 64
s = 8  # filter size
num_stage = 2 # number of stage
B = 8    # number of channel

# training data
#train_data_name = './Generating_training_data/train_new.mat'  
train_data_name = './training_data/Train_all.mat'  

# tf Graph input (only pictures)
P = tf.placeholder(tf.float32, [batch_size, Height, Width, 1])   # Pan image
MS = tf.placeholder(tf.float32, [batch_size, Height, Width, B])  # Upsampled LRMS image
GT = tf.placeholder(tf.float32, [batch_size, Height, Width, B])  # Ground Truth HRMS Image


def eta(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)

def proxU(image):
    "using ResNet to approximate the proximal operator for updating U"  
    inchannels = image.get_shape().as_list()[-1]       
    num_features  =  inchannels//2
    conv = tf.layers.conv2d(image, num_features, 3, padding="same", activation = tf.nn.relu)
        
    for i in range(5):          
        a = tf.layers.conv2d(conv, num_features, 3, padding="same", activation = tf.nn.relu)
        b = tf.layers.conv2d(a, num_features, 3, padding="same", activation = tf.nn.relu)  
        conv = conv + b
    out = tf.layers.conv2d(conv, inchannels, 3, padding = 'SAME')             
    return out

def proxV(image):
    "using ResNet to approximate the proximal operator for updating V"  
    inchannels = image.get_shape().as_list()[-1]       
    num_features  =  inchannels//2
    conv = tf.layers.conv2d(image, num_features, 3, padding="same", activation = tf.nn.relu)
        
    for i in range(5):          
        a = tf.layers.conv2d(conv, num_features, 3, padding="same", activation = tf.nn.relu)
        b = tf.layers.conv2d(a, num_features, 3, padding="same", activation = tf.nn.relu)  
        conv = conv + b
    out = tf.layers.conv2d(conv, inchannels, 3, padding = 'SAME')             
    return out

def proxC(image):
    "using ResNet to approximate the proximal operator for updating C"  
    inchannels = image.get_shape().as_list()[-1]       
    num_features  =  inchannels//2
    conv = tf.layers.conv2d(image, num_features, 3, padding="same", activation = tf.nn.relu)
        
    for i in range(5):          
        a = tf.layers.conv2d(conv, num_features, 3, padding="same", activation = tf.nn.relu)
        b = tf.layers.conv2d(a, num_features, 3, padding="same", activation = tf.nn.relu)  
        conv = conv + b
    out = tf.layers.conv2d(conv, inchannels, 3, padding = 'SAME')             
    return out


def PanCSCNet(P,MS):
    ## Filters of U_Net
    D_ck = tf.get_variable("D_ck", shape=[s, s, K, 1], initializer=tf2.initializers.GlorotUniform())
    D_uk = tf.get_variable("D_uk", shape=[s, s, K, 1], initializer=tf2.initializers.GlorotUniform())
    
    ## Filters of V_Net
    H_ck = tf.get_variable("H_ck", shape=[s, s, K, B], initializer=tf2.initializers.GlorotUniform())
    H_vk = tf.get_variable("H_vk", shape=[s, s, K, B], initializer=tf2.initializers.GlorotUniform())
        
    ## Fused Filters
    G_c = tf.get_variable("G_c", shape=[s, s, K, B], initializer=tf2.initializers.GlorotUniform())
    G_u = tf.get_variable("G_u", shape=[s, s, K, B], initializer=tf2.initializers.GlorotUniform())
    G_v = tf.get_variable("G_v", shape=[s, s, K, B], initializer=tf2.initializers.GlorotUniform())
    
    ## regularization parameters
    eta1 = tf.get_variable('eta1', initializer=tf.constant(0.1), trainable= True) 
    eta2 = tf.get_variable('eta2', initializer=tf.constant(0.1), trainable= True) 
    eta3 = tf.get_variable('eta3', initializer=tf.constant(0.1), trainable= True) 
    
    ## initialize C,U,V
    U = tf.zeros([batch_size, Height, Width, K]) 
    V = tf.zeros([batch_size, Height, Width, K]) 
    C = tf.zeros([batch_size, Height, Width, K])    
    for i in range(num_stage):
        ##  U_Net
        P_c = tf.nn.conv2d(C, D_ck, strides=[1, 1, 1, 1], padding='SAME')  # batch_size x M x N x 1
        P_u = tf.nn.conv2d(U, D_uk, strides=[1, 1, 1, 1], padding='SAME')  # batch_size x M x N x 1
        epsilon_p = tf.subtract(tf.add(P_c,P_u),P)  # batch_size x M x N x 1
        Delta_g = tf.nn.conv2d_transpose(epsilon_p,D_uk,output_shape=[batch_size,Height,Width,K],strides=[1,1,1,1],padding='SAME') 
        # batch_size x M x N x K
        temp = tf.subtract(U,tf.multiply(eta1,Delta_g)) # batch_size x M x N x K
        U = proxU(temp) # batch_size x M x N x K
        
        ## V_Net
        M_c = tf.nn.conv2d(C, H_ck, strides=[1, 1, 1, 1], padding='SAME')  # batch_size x M x N x B
        M_v = tf.nn.conv2d(U, H_vk, strides=[1, 1, 1, 1], padding='SAME')  # batch_size x M x N x B
        epsilon_m = tf.subtract(tf.add(M_c,M_v),MS)  # batch_size x M x N x B
        Delta_h = tf.nn.conv2d_transpose(epsilon_m,H_vk,output_shape=[batch_size,Height,Width,K],strides=[1,1,1,1],padding='SAME')
        # batch_size x M x N x K
        temp = tf.subtract(V,tf.multiply(eta2,Delta_h)) # batch_size x M x N x K
        V = proxV(temp) # batch_size x M x N x K

        ## C_Net
        P_hat = tf.subtract(P, tf.nn.conv2d(U, D_uk, strides=[1, 1, 1, 1], padding='SAME'))  # batch_size x M x N x 1
        M_hat = tf.subtract(MS, tf.nn.conv2d(V, H_vk, strides=[1, 1, 1, 1], padding='SAME')) # batch_size x M x N x B
        N = tf.concat([P_hat,M_hat], 3)  # batch_size x M x N x (B+1)
        L_ck = tf.concat([D_ck,H_ck], 3) # s x s x K x (B+1)
        
        F_c = tf.nn.conv2d(C, L_ck, strides=[1, 1, 1, 1], padding='SAME')  # batch_size x M x N x (B+1)
        epsilon_c = tf.subtract(F_c,N)  # batch_size x M x N x (B+1)
        Delta_l = tf.nn.conv2d_transpose(epsilon_c,L_ck,output_shape=[batch_size,Height,Width,K],strides=[1,1,1,1],padding='SAME')
        # batch_size x M x N x K
        temp = tf.subtract(C,tf.multiply(eta3,Delta_l)) # batch_size x M x N x K
        C = proxC(temp) # batch_size x M x N x K
    
    rec_c = tf.nn.conv2d(C, G_c, strides=[1, 1, 1, 1], padding='SAME')
    rec_v = tf.nn.conv2d(V, G_v, strides=[1, 1, 1, 1], padding='SAME')
    rec_u = tf.nn.conv2d(U, G_u, strides=[1, 1, 1, 1], padding='SAME')
    
    O_rec = rec_c + rec_v + rec_u
    return O_rec
        
f_pred = PanCSCNet(P, MS)
f_true = GT
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.9, staircase=True)
# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(f_true - f_pred, 2))  #1000*
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

all_vars = tf.trainable_variables() 
print("Total parameters' number: %d" %(np.sum([np.prod(v.get_shape().as_list()) for v in all_vars]))) 

saver = tf.train.Saver(max_to_keep=10)
#loading data 
train_data = h5py.File(train_data_name)  # case 2: for large data (for real training v7.3 data in matlab)
#train_data = sio.loadmat(train_data_name)
#### read training data ####
#train_label = train_data['gt_train'][...]  ## ground truth N*H*W*C
#train_data_x = train_data['pan_train'][...]  #### Pan image N*H*W
#train_data_y = train_data['lms_train'][...]  #### MS image interpolation -to Pan scale

train_label = train_data['gt'][...]  ## ground truth N*H*W*C
train_data_x = train_data['pan'][...]  #### Pan image N*H*W
train_data_y = train_data['lms'][...]  #### MS image interpolation -to Pan scale

train_label = np.array(train_label, dtype=np.float32)/2047.
train_data_x = np.array(train_data_x, dtype=np.float32)/2047.
train_data_y = np.array(train_data_y, dtype=np.float32)/2047.
N = train_label.shape[0]

batch_num = len(train_data_x) //batch_size

print(len(train_data_x), batch_num)

#with tf.compat.v1.Session(config=config) as sess:
with tf.Session() as sess:
    sess.run(init)
    if tf.train.get_checkpoint_state('model_test'):   # load previous trained model
     ckpt = tf.train.latest_checkpoint('model_test')
     saver.restore(sess, ckpt)
     ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
     start_point = int(ckpt_num[-1])
        
     print("Load success")
     print('start_point %d' % (start_point))
   
    else:
      print("re-training")
      start_point = 0 
      training_error = []
      
    for j in range(start_point,epoch):
        total_loss = 0
        indices = np.random.permutation(train_data_x.shape[0])
        train_data_x = train_data_x[indices,:,:]
        train_data_y = train_data_y[indices,:,:,:]
        train_label = train_label[indices,:,:,:]

        for idx in range(0, batch_num):
           batch_xs = train_data_x[idx*batch_size : (idx+1)*batch_size,:,:]
           batch_xs = batch_xs[:, :, :, np.newaxis] 
           batch_ys = train_data_y[idx*batch_size : (idx+1)*batch_size,:,:,:]
           batch_label = train_label[ idx * batch_size: (idx + 1) * batch_size,:,:,:]
           _, loss_batch = sess.run([train_step, loss], feed_dict={P: batch_xs, MS: batch_ys, GT: batch_label})
           total_loss += loss_batch
           #print(' ep, idx, loss_batch = %6d:%6d: %6.3f' % (ep,idx, loss_batch))
        print(' epoch %d: total_loss = %6.5f' % (j+1, total_loss))
        training_error.append(total_loss)

        gc.collect()
        checkpoint_path = os.path.join('model_test', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=j+1)
    loss_save_path = os.path.join('model_test', 'training_error.mat')
    io.savemat(loss_save_path, {'training_error': training_error}) 
    print("Optimization Finished!")

