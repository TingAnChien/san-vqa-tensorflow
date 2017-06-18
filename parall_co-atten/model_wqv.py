#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os, h5py, sys, argparse
import time
import math
import cv2
import json
import attention
rnn_cell = tf.nn.rnn_cell

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_image, dim_hidden, max_words_q, vocabulary_size, drop_out_rate):

        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.input_embedding_size = input_embedding_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.max_words_q = max_words_q
        self.vocabulary_size = vocabulary_size  
        self.drop_out_rate = drop_out_rate

        # question-embedding
        self.embed_ques_W = tf.Variable(tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_ques_W')

        # encoder: RNN body
        self.lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True)
        self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = 1 - self.drop_out_rate)
        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True)
        self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = 1 - self.drop_out_rate)
        self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])

        # state-embedding
        #self.embed_state_W = tf.Variable(tf.random_uniform([2*rnn_size*rnn_layer, self.dim_hidden], -0.08,0.08),name='embed_state_W')
        #self.embed_state_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_state_b')
        # image-embedding
        self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, self.dim_hidden], -0.08, 0.08), name='embed_image_W')
        self.embed_image_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_image_b')
        # score-embedding
        self.embed_scor_W = tf.Variable(tf.random_uniform([dim_hidden, num_output], -0.08, 0.08), name='embed_scor_W')
        self.embed_scor_b = tf.Variable(tf.random_uniform([num_output], -0.08, 0.08), name='embed_scor_b')
        self.embed_feat_W = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.08, 0.08), name='embed_feat_W')
        self.embed_feat_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_feat_b')
        self.embed_wv_W = tf.Variable(tf.random_uniform([self.input_embedding_size, dim_hidden/2], -0.08, 0.08), name='embed_wv_W')
        self.embed_wv_b = tf.Variable(tf.random_uniform([dim_hidden/2], -0.08, 0.08), name='embed_wv_b')
        self.embed_qv_W = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden/2], -0.08, 0.08), name='embed_qv_W') #modi
        self.embed_qv_b = tf.Variable(tf.random_uniform([dim_hidden/2], -0.08, 0.08), name='embed_qv_b') #modi /2


    def build_model(self):
        image = tf.placeholder(tf.float32, [self.batch_size, 7,7,self.dim_image])
        question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        label = tf.placeholder(tf.int64, [self.batch_size,]) 
        
        state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])
        loss = 0.0
        states_feat = list()
        words_feat = list()
        for i in range(max_words_q):
            if i==0:
                ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
            else:
                tf.get_variable_scope().reuse_variables()
                ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:,i-1])

            ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
            ques_emb = tf.tanh(ques_emb_drop)

            output, state = self.stacked_lstm(ques_emb, state)
            words_feat.append(ques_emb_linear)
            states_feat.append(state) #(tf.tanh(state))  
        #final word embed
        states_feat = tf.pack(states_feat)
        state_emb = tf.transpose(states_feat, perm=[1,0,2])
        words_feat = tf.pack(words_feat)
        words_emb = tf.transpose(words_feat, perm=[1,0,2]) #check d=512
        
        ATT_wv = attention.attention(words_emb, image, 512, self.batch_size)
        atten_vis_emb_0, atten_word_emb_0 = ATT_wv.model()
        atten_word_emb_0 = tf.reduce_sum(atten_word_emb_0,1)
        atten_vis_emb_0 = tf.reduce_sum(tf.reduce_sum(atten_vis_emb_0,1),1)
        h_w = tf.mul(atten_vis_emb_0, atten_word_emb_0) 
        h_w_emb = tf.tanh(tf.nn.xw_plus_b(h_w, self.embed_wv_W,self.embed_wv_b))

        image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
        image_drop = tf.reshape(image_drop, [-1,self.dim_image])
        image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(tf.reshape(image_linear,[self.batch_size, 7,7,self.dim_hidden]))
        ########################################
        #add attention models
        #dummy mapping question from d=26->64
       
        #image [batch,h,w,channels]
        ATT = attention.attention(state_emb, image_emb, self.dim_hidden, self.batch_size)
        atten_vis_emb, atten_ques_emb = ATT.model()
        atten_ques_emb = tf.reduce_sum(atten_ques_emb,1)
        atten_vis_emb = tf.reduce_sum(tf.reduce_sum(atten_vis_emb,1),1)

        #########################################

        s_emb = tf.nn.xw_plus_b(tf.mul(atten_vis_emb, atten_ques_emb),self.embed_qv_W,self.embed_qv_b) 
        
        scores = tf.tanh(tf.nn.xw_plus_b(tf.concat(1,[s_emb,h_w_emb]),self.embed_feat_W,self.embed_feat_b))
        scores_emb = tf.nn.xw_plus_b(scores,self.embed_scor_W,self.embed_scor_b)
        

        # Calculate cross entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(scores_emb, label)

        # Calculate loss
        loss = tf.reduce_mean(cross_entropy)
        return loss, image, question, label
    
    def build_generator(self):
        image = tf.placeholder(tf.float32, [self.batch_size, 7,7,self.dim_image])
        question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        
        #image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        #question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])
        loss = 0.0
        states_feat = list()
        words_feat = list()
        for i in range(max_words_q):
            if i==0:
                ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
            else:
                tf.get_variable_scope().reuse_variables()
                ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:,i-1])

            ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
            ques_emb = tf.tanh(ques_emb_drop)

            output, state = self.stacked_lstm(ques_emb, state)
            words_feat.append(ques_emb_linear)
            states_feat.append(state) #(tf.tanh(state))  
        #final word embed
        states_feat = tf.pack(states_feat)
        state_emb = tf.transpose(states_feat, perm=[1,0,2])
        words_feat = tf.pack(words_feat)
        words_emb = tf.transpose(words_feat, perm=[1,0,2]) #check d=512
        
        ATT_wv = attention.attention(words_emb, image, 512, self.batch_size)
        atten_vis_emb_0, atten_word_emb_0 = ATT_wv.model()
        atten_word_emb_0 = tf.reduce_sum(atten_word_emb_0,1)
        atten_vis_emb_0 = tf.reduce_sum(tf.reduce_sum(atten_vis_emb_0,1),1)
        h_w = tf.mul(atten_vis_emb_0, atten_word_emb_0) 
        h_w_emb = tf.tanh(tf.nn.xw_plus_b(h_w, self.embed_wv_W,self.embed_wv_b))

        image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
        image_drop = tf.reshape(image_drop, [-1,self.dim_image])
        image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(tf.reshape(image_linear,[self.batch_size, 7,7,self.dim_hidden]))
        ########################################
        #add attention models
        #dummy mapping question from d=26->64
       
        #image [batch,h,w,channels]
        ATT = attention.attention(state_emb, image_emb, self.dim_hidden, self.batch_size)
        atten_vis_emb, atten_ques_emb = ATT.model()
        atten_ques_emb = tf.reduce_sum(atten_ques_emb,1)
        atten_vis_emb = tf.reduce_sum(tf.reduce_sum(atten_vis_emb,1),1)

        #########################################

        s_emb = tf.nn.xw_plus_b(tf.mul(atten_vis_emb, atten_ques_emb),self.embed_qv_W,self.embed_qv_b) 
        #scores_emb = tf.nn.xw_plus_b(scores,self.embed_scor_W,self.embed_scor_b)
        
        scores = tf.tanh(tf.nn.xw_plus_b(tf.concat(1,[s_emb,h_w_emb]),self.embed_feat_W,self.embed_feat_b))
        
        # FINAL ANSWER
        generated_ANS = tf.nn.xw_plus_b(scores, self.embed_scor_W, self.embed_scor_b)
        generated_ANS = tf.nn.softmax(generated_ANS)

        return generated_ANS, image, question
    
#####################################################
#                 Global Parameters                 #  
#####################################################
print('Loading parameters ...')
# Data input setting
input_img_h5 = '../data_img.h5'
input_ques_h5 = '../data_prepro.h5'
input_json = '../data_prepro.json'

# Train Parameters setting
learning_rate = 0.00008                  # learning rate for rmsprop
#starter_learning_rate = 3e-4
learning_rate_decay_start = -1          # at what iteration to start decaying learning rate? (-1 = dont)
batch_size = 200                       # batch_size for each iterations
input_embedding_size = 512 #200              # the encoding size of each token in the vocabulary
rnn_size = 512                          # size of the rnn in number of hidden nodes in each layer
rnn_layer = 2                           # number of the rnn layer
dim_image = 512
dim_hidden = 2048 #1024                 # size of the common embedding vector
num_output = 1000                       # number of output answers
img_norm = 1                            # normalize the image feature. 1 = normalize, 0 = not normalize
decay_factor = 0.99997592083
#decay_factor = 0.99999999 #follow the paper setting

# Check point
checkpoint_path = './logs/v2/'

# misc
gpu_id = 0
max_itr = 150000
n_epochs = 300
max_words_q = 26
num_answer = 1000
#####################################################

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v

def get_data():

    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_train')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_train')
        # convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('question aligning')
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])

    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        
        img_feature = np.transpose(img_feature,(0,2,3,1))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,[512,1,1,1]),(1,2,3,0))+1e-7)
    
    return dataset, img_feature, train_data

def get_data_test():
    dataset = {}
    test_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        tem = hf.get('images_test')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_test')
        # convert into 0~82459
        test_data['img_list'] = np.array(tem)-1
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
        # MC_answer_test
        #tem = hf.get('MC_ans_test')
        #test_data['MC_ans_test'] = np.array(tem)
        # answer is 1~1000
        tem = hf.get('answers')
        test_data['answers'] = np.array(tem)-1


    print('question aligning')
    test_data['question'] = right_align(test_data['question'], test_data['length_q'])

    print('Normalizing image feature')
    if img_norm:
        #tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        
        #img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))
        img_feature = np.transpose(img_feature,(0,2,3,1))
        #values = np.transpose(np.tile(tem,[512,1,1,1]),(1,2,3,0)) 
        #mask = (values != 0).astype(np.float32)
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,[512,1,1,1]),(1,2,3,0))+1e-7)
        #img_feature = mask * np.divide(img_feature, values+1e-7)

    return dataset, img_feature, test_data

def train():
    print 'loading dataset...'
    dataset, img_feature, train_data = get_data()
    num_train = train_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print 'vocabulary_size : ' + str(vocabulary_size)

    print 'constructing  model...'
    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = batch_size,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            max_words_q = max_words_q,  
            vocabulary_size = vocabulary_size,
            drop_out_rate = 0.5)

    tf_loss, tf_image, tf_question, tf_label = model.build_model()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(max_to_keep=10)#, var_list=[var for var in tvars if 'embed' in var.name or 'b_' in var.name or 'W' in var.name])
        #saver.restore(sess, 'logs/model-18000')
    tvars = tf.trainable_variables()
    lr = tf.Variable(learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    
    #opt = tf.train.RMSPropOptimizer(learning_rate=lr,momentum=0.99)
    # gradient clipping
    gvs = opt.compute_gradients(tf_loss,tvars)
    with tf.device('/cpu:0'):    
        clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs if grad is not None]
    train_op = opt.apply_gradients(clipped_gvs)


    sess.run(tf.initialize_all_variables())
    print 'start training...'
    for itr in range(max_itr):
        tStart = time.time()
        # shuffle the training data
        index = np.random.random_integers(0, num_train-1, batch_size)

        current_question = train_data['question'][index,:]
        current_length_q = train_data['length_q'][index]
        current_answers = train_data['answers'][index]
        current_img_list = train_data['img_list'][index]
        current_img = img_feature[current_img_list,:]

        # do the training process!!!
        _, loss = sess.run(
                [train_op, tf_loss],
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question,
                    tf_label: current_answers
                    })

        current_learning_rate = lr*decay_factor
        lr.assign(current_learning_rate).eval(session=sess)

        tStop = time.time()
        if np.mod(itr, 20) == 0:
            print "Iteration: ", itr, " Loss: ", loss, " Learning Rate: ", lr.eval(session=sess)
            print ("Time Cost:", round(tStop - tStart,2), "s")
        if np.mod(itr, 3000) == 0 and itr != 0 : #15000
            print "Iteration ", itr, " is done. Saving the model ..."
            saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)

    print "Finally, saving the model ..."
    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"


def test(model_path='./logs/model-36000'):
    print 'loading dataset...'
    dataset, img_feature, test_data = get_data_test()
    num_test = test_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print 'vocabulary_size : ' + str(vocabulary_size)

    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = batch_size,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            max_words_q = max_words_q,
            vocabulary_size = vocabulary_size,
            drop_out_rate = 0)

    tf_answer, tf_image, tf_question, = model.build_generator()

    #sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    with tf.device('/cpu:0'):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

    tStart_total = time.time()
    result = []
    for current_batch_start_idx in xrange(0,num_test-1,batch_size):
    #for current_batch_start_idx in xrange(0,3,batch_size):
        tStart = time.time()
        # set data into current*
        if current_batch_start_idx + batch_size < num_test:
            current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
        else:
            current_batch_file_idx = range(current_batch_start_idx,num_test)

        current_question = test_data['question'][current_batch_file_idx,:]
        current_length_q = test_data['length_q'][current_batch_file_idx]
        current_img_list = test_data['img_list'][current_batch_file_idx]
        current_ques_id  = test_data['ques_id'][current_batch_file_idx]
        current_img = img_feature[current_img_list,:] # (batch_size, dim_image)

        # deal with the last batch
        if(len(current_img)<batch_size):
                pad_img = np.zeros((batch_size-len(current_img),7,7,dim_image),dtype=np.int)
                pad_q = np.zeros((batch_size-len(current_img),max_words_q),dtype=np.int)
                pad_q_len = np.zeros(batch_size-len(current_length_q),dtype=np.int)
                pad_q_id = np.zeros(batch_size-len(current_length_q),dtype=np.int)
                pad_ques_id = np.zeros(batch_size-len(current_length_q),dtype=np.int)
                pad_img_list = np.zeros(batch_size-len(current_length_q),dtype=np.int)
                current_img = np.concatenate((current_img, pad_img))
                current_question = np.concatenate((current_question, pad_q))
                current_length_q = np.concatenate((current_length_q, pad_q_len))
                current_ques_id = np.concatenate((current_ques_id, pad_q_id))
                current_img_list = np.concatenate((current_img_list, pad_img_list))


        generated_ans = sess.run(
                tf_answer,
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question
                    })

        top_ans = np.argmax(generated_ans, axis=1)


        # initialize json list
        for i in xrange(0,batch_size):
            ans = dataset['ix_to_ans'][str(top_ans[i]+1)]
            if(current_ques_id[i] == 0):
                continue
            result.append({u'answer': ans, u'question_id': str(current_ques_id[i])})

        tStop = time.time()
        print ("Testing batch: ", current_batch_file_idx[0])
        print ("Time Cost:", round(tStop - tStart,2), "s")
    print ("Testing done.")
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
    # Save to JSON
    print 'Saving result...'
    my_list = list(result)
    dd = json.dump(my_list,open('testing/data.json','w'))

if __name__ == '__main__':
    
    with tf.device('/gpu:'+str(0)):
        train()
    """
    with tf.device('/gpu: 0'):
        test()
    
    """
