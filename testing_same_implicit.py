from __future__ import division, print_function, absolute_import

import math
import os
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import time
import gc
from metrics import ref_evaluate, no_ref_evaluate

os.environ['CUDA_VISIBLE_DEVICES']='1'
# Parameters
Height=64
Width=64
batch_size = 1
K=32 # number of filters
s=8 # filter size
num_stage=2 # number of layers
B=8
n_res = 5

# tf Graph input (only pictures)
P = tf.placeholder(dtype = tf.float32, shape = [batch_size, Height, Width, 1])   # Pan image
MS1 = tf.placeholder(dtype = tf.float32, shape = [batch_size, Height, Width, B])  # Upsampled LRMS image
GT = tf.placeholder(dtype = tf.float32, shape = [batch_size, Height, Width, B])  # Ground Truth HRMS Image
weights = []


def proxU(image):
    "using ResNet to approximate the proximal operator for updating U"  
    inchannels = image.get_shape().as_list()[-1]       
    num_features  =  inchannels//2
    conv = tf.layers.conv2d(image, num_features, 3, padding="same", activation = tf.nn.relu)
        
    for i in range(n_res):          
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
        
    for i in range(n_res):          
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
        
    for i in range(n_res):          
        a = tf.layers.conv2d(conv, num_features, 3, padding="same", activation = tf.nn.relu)
        b = tf.layers.conv2d(a, num_features, 3, padding="same", activation = tf.nn.relu)  
        conv = conv + b
    out = tf.layers.conv2d(conv, inchannels, 3, padding = 'SAME')             
    return out


def PanCSCNet(P,MS):
    ## Filters of U_Net
    D_ck = tf.get_variable("D_ck", shape=[s, s, K, 1], initializer=tf.contrib.layers.xavier_initializer())
    D_uk = tf.get_variable("D_uk", shape=[s, s, K, 1], initializer=tf.contrib.layers.xavier_initializer())
    
    ## Filters of V_Net
    H_ck = tf.get_variable("H_ck", shape=[s, s, K, B], initializer=tf.contrib.layers.xavier_initializer())
    H_vk = tf.get_variable("H_vk", shape=[s, s, K, B], initializer=tf.contrib.layers.xavier_initializer())
        
    ## Fused Filters
    G_c = tf.get_variable("G_c", shape=[s, s, K, B], initializer=tf.contrib.layers.xavier_initializer())
    G_u = tf.get_variable("G_u", shape=[s, s, K, B], initializer=tf.contrib.layers.xavier_initializer())
    G_v = tf.get_variable("G_v", shape=[s, s, K, B], initializer=tf.contrib.layers.xavier_initializer())
    
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
        

f_pred = PanCSCNet(P, MS1)
#output = tf.clip_by_value(f_pred, 0, 1)

data = h5py.File('./testdata/test.mat')  # case 2: for large data (for real training v7.3 data in matlab)
#### read training data ####
GT = data['gt'][...]  ## ground truth N*H*W*C
GT = np.array(GT,dtype = np.float32) /2047.
PAN = data['pan'][...]  #### Pan image N*H*W
PAN = np.array(PAN,dtype = np.float32) /2047.
LMS = data['lms'][...]  #### MS image interpolation -to Pan scale
LMS = np.array(LMS,dtype = np.float32) /2047.
MS = data['ms'][...]
MS = np.array(MS,dtype = np.float32) /2047.
N = GT.shape[0]


saver = tf.train.Saver()
ref_results = np.zeros((N,6))
no_ref_results = np.zeros((N,3))
Fused_result = np.zeros_like(GT)
with tf.Session() as sess:
     saver.restore(sess, './model_training_implicit/model.ckpt-100')
     for i in range(0,N):
         print('Test sample %d' % (i+1))
         gt = GT[i,:,:,:]
         pan = PAN[i,:,:]
         pan = pan[np.newaxis, :, :, np.newaxis]
         lms = LMS[i,:,:,:]
         lms = lms[np.newaxis, :, :, :]
         ms = MS[i,:,:,:]
         batch_xs=pan[0:batch_size,0:Height, 0:Width, :]
         batch_ys=lms[0:batch_size,0:Height, 0:Width, :]
         Y_p = sess.run(f_pred, feed_dict={P: batch_xs, MS1: batch_ys})
         temp_ref_results = ref_evaluate(np.uint8(Y_p[0,:,:,:]*255), np.uint8(gt*255))
#         temp_no_ref_results = no_ref_evaluate(np.uint8(Y_p[0,:,:,:]*255), np.uint8(pan[0,:,:,:]*255), np.uint8(ms*255))
         ref_results[i,:] = temp_ref_results
         Fused_result[i,:,:,:] = Y_p[0,:,:,:]
#         no_ref_results[i,:] = temp_no_ref_results
     sio.savemat('ref_results.mat', {'ref_results': ref_results})
     sio.savemat('Same_our_model_implicit_layer2_filter32.mat', {'Fused_result': Fused_result,'gt': GT, 'pan': PAN, 'ms': MS, 'lms': LMS})
#     sio.savemat('no_ref_results.mat', {'no_ref_results': no_ref_results}) 
     print('################## reference comparision #######################')
     print('PSNR,   SSIM,   SAM,   ERGAS,    SCC,     Q')
     print([round(i,4) for i in np.mean(ref_results,0)])
     print('################## reference comparision #######################')
           
#     print('################## no reference comparision ####################')
#     print('D_lamda,   D_s,   QNR')
#     print([round(i,4) for i in np.mean(no_ref_results,0)])
#     print('################## no reference comparision ####################')
     
     



