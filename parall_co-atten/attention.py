import numpy as np
import tensorflow as tf

class attention():
    def __init__(self,q_feat,v_feat,dim_d,batch_size):
        self.dim_k = 1024 #hidden dim
        self.q_feat = q_feat
        self.v_feat = v_feat
        self.dim_d = dim_d
        self.batch_size = batch_size    
        
        self.W_affinity = tf.Variable(tf.random_uniform([self.dim_d, self.dim_d], -0.08, 0.08), name='affinity_W')

        ##image embed
        self.W_ques = tf.Variable(tf.random_uniform([self.dim_k, self.dim_d], -0.08, 0.08), name='W_ques')
        self.b_ques = tf.Variable(tf.random_uniform([self.dim_k,1], -0.08, 0.08), name='b_ques')
        self.W_vis = tf.Variable(tf.random_uniform([self.dim_k, self.dim_d], -0.08, 0.08), name='W_vis')
        self.b_vis = tf.Variable(tf.random_uniform([self.dim_k,1], -0.08, 0.08), name='b_vis')
        self.W_hv_t = tf.Variable(tf.random_uniform([self.dim_k,1], -0.08, 0.08), name='W_hv_t')
        self.b_hv_t = tf.Variable(tf.random_uniform([1], -0.08, 0.08), name='b_hv_t')
        #self.b_hv_t = tf.Variable(tf.random_uniform([self.dim_k], -0.08, 0.08), name='b_hv_t')
        self.W_hq_t = tf.Variable(tf.random_uniform([self.dim_k,1], -0.08, 0.08), name='W_hq_t')
        self.b_hq_t = tf.Variable(tf.random_uniform([1], -0.08, 0.08), name='b_hq_t')
        #self.b_hq_t = tf.Variable(tf.random_uniform([self.dim_k], -0.08, 0.08), name='b_hq_t')


    def model(self):

        q_feat = tf.reshape(tf.transpose(self.q_feat,perm=[2,0,1]),[self.dim_d,-1])
        v_feat = tf.transpose(tf.reshape(self.v_feat, [-1,self.dim_d]),perm=[1,0])
        #q_feat [self.dim_d*inst_q]
        #v_feat [self.dim_d*inst_v]
        #self.dim_d = tf.get_shape(q_feat)[0]

        C = tf.tanh(tf.matmul(tf.matmul(tf.transpose(q_feat),self.W_affinity),v_feat))
        h_vis = tf.tanh(tf.matmul(self.W_vis,v_feat)+tf.matmul(tf.matmul(self.W_ques,q_feat)+self.b_ques,C))
        h_ques = tf.tanh(tf.matmul(self.W_ques,q_feat)+tf.matmul(tf.matmul(self.W_vis,v_feat)+self.b_vis,tf.transpose(C)))

        atten_vis = tf.nn.softmax(tf.reshape(tf.matmul(tf.transpose(h_vis),self.W_hv_t)+self.b_hv_t,[self.batch_size,-1]))
        atten_ques = tf.nn.softmax(tf.reshape(tf.matmul(tf.transpose(h_ques),self.W_hq_t)+self.b_hq_t,[self.batch_size,-1]))
        atten_vis = tf.reshape(atten_vis,[self.batch_size,7,7])

        v_feat_ = tf.reshape(v_feat,[self.dim_d,self.batch_size,7,7])
        q_feat_ = tf.reshape(q_feat,[self.dim_d,self.batch_size,-1])

        atten_feat_vis = tf.transpose(v_feat_ * atten_vis, perm=[1,2,3,0]) 
        atten_feat_ques = tf.transpose(q_feat_ * atten_ques, perm=[1,2,0])

        return atten_feat_vis, atten_feat_ques

