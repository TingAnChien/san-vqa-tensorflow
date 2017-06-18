from nltk.tokenize import word_tokenize
from scipy.ndimage.filters import gaussian_filter
import scipy.misc as misc
import numpy as np
import gensim
import cv2
import caffe
import tensorflow as tf
import json
import os
import time
import pdb

rnn_cell = tf.nn.rnn_cell

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_image, dim_hidden, dim_attention, max_words_q, drop_out_rate, sess):

        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.input_embedding_size = input_embedding_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.dim_att = dim_attention
        self.max_words_q = max_words_q
        self.drop_out_rate = drop_out_rate
        self._sess = sess

        # encoder: RNN body
        self.lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True)
        self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = 1 - self.drop_out_rate)
        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True)
        self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = 1 - self.drop_out_rate)
        self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])

        # image-embedding
        self.embed_image_W = tf.Variable(tf.random_uniform([self.dim_image[2], self.dim_hidden], -0.08, 0.08), name='embed_image_W')
        self.embed_image_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_image_b')
        # score-embedding
        self.embed_scor_W = tf.Variable(tf.random_uniform([self.dim_hidden, num_output], -0.08, 0.08), name='embed_scor_W')
        self.embed_scor_b = tf.Variable(tf.random_uniform([num_output], -0.08, 0.08), name='embed_scor_b')

    def build_generator(self):
        self.image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image[0], self.dim_image[1], self.dim_image[2]])
        self.question = tf.placeholder(tf.float32, [self.batch_size, self.max_words_q, self.input_embedding_size])

        state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])
        loss = 0.0
        with tf.variable_scope("embed"):
            for i in range(max_words_q):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = self.stacked_lstm(self.question[:, i], state)

        # multimodal (fusing question & image)
        question_emb = state

        image_emb = tf.reshape(self.image, [-1, self.dim_image[2]]) # (b x m) x d
        image_emb = tf.nn.xw_plus_b(image_emb, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(image_emb)

        #attention models
        with tf.variable_scope("att1"):
            self.prob_att1, comb_emb = self.attention(question_emb, image_emb)
        with tf.variable_scope("att2"):
            self.prob_att2, comb_emb = self.attention(comb_emb, image_emb)
        comb_emb = tf.nn.dropout(comb_emb, 1 - self.drop_out_rate)
        scores_emb = tf.nn.xw_plus_b(comb_emb, self.embed_scor_W, self.embed_scor_b) 

        # FINAL ANSWER
        self.generated_ANS = tf.nn.softmax(scores_emb)

    def attention(self, question_emb, image_emb):
        # Attention weight
        # question-attention
        ques_att_W = tf.get_variable('ques_att_W', [self.dim_hidden, self.dim_att], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        ques_att_b = tf.get_variable('ques_att_b', [self.dim_att], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        # image-attention
        image_att_W = tf.get_variable('image_att_W', [self.dim_hidden, self.dim_att], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        image_att_b = tf.get_variable('image_att_b', [self.dim_att], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        # probability-attention
        prob_att_W = tf.get_variable('prob_att_W', [self.dim_att, 1], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        prob_att_b = tf.get_variable('prob_att_b', [1], 
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))

        question_att = tf.expand_dims(question_emb, 1) # b x 1 x d
        question_att = tf.tile(question_att, tf.constant([1, self.dim_image[0] * self.dim_image[1], 1])) # b x m x d
        question_att = tf.reshape(question_att, [-1, self.dim_hidden]) # (b x m) x d
        question_att = tf.tanh(tf.nn.xw_plus_b(question_att, ques_att_W, ques_att_b)) # (b x m) x k
        
        image_att = tf.tanh(tf.nn.xw_plus_b(image_emb, image_att_W, image_att_b)) # (b x m) x k

        output_att = tf.tanh(image_att + question_att) # (b x m) x k
        output_att = tf.nn.dropout(output_att, 1 - self.drop_out_rate)
        prob_att = tf.nn.xw_plus_b(output_att, prob_att_W, prob_att_b) # (b x m) x 1
        prob_att = tf.reshape(prob_att, [self.batch_size, self.dim_image[0] * self.dim_image[1]]) # b x m
        prob_att = tf.nn.softmax(prob_att)

        image_att = []
        image_emb = tf.reshape(image_emb, [self.batch_size, self.dim_image[0] * self.dim_image[1], self.dim_hidden]) # b x m x d
        for b in range(self.batch_size):
            image_att.append(tf.matmul(tf.expand_dims(prob_att[b,:],0), image_emb[b,:,:]))

        image_att = tf.pack(image_att)
        image_att = tf.reduce_sum(image_att, 1)

        comb_emb = tf.add(image_att, question_emb)

        return prob_att, comb_emb

    def test(self, image, question):

        generated_ans, prob_att1, prob_att2 = self._sess.run(
            [self.generated_ANS, self.prob_att1, self.prob_att2],
            feed_dict={
                self.image: image,
                self.question: question
            })
        return generated_ans, prob_att1, prob_att2

########################################################## 
web_path = '/path/to/your/website'
vqa_data_json = os.path.join(web_path, 'vqa_data.json')
answer_txt = os.path.join(web_path, 'media/vqa_ans.txt')
att_image = os.path.join(web_path, 'media/att.jpg')
#ques = 'What is this photo taken looking through?'
#imname = 'test.jpg'

itoa_json = 'ix_to_ans.json'            # mapping idx to answer
cnn_proto = 'vgg19/deploy.prototxt'
cnn_model = 'vgg19/VGG_ILSVRC_19_layers.caffemodel'
vqa_model = '/path/to/your/vqa_model'
w2v_model = 'w2v/GoogleNews-vectors-negative300.bin'
model_w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_model, binary=True)

input_embedding_size = 300              # the encoding size of each token in the vocabulary
rnn_size = 256                          # size of the rnn in number of hidden nodes in each layer
rnn_layer = 2                           # number of the rnn layer
dim_image = (7, 7, 512)
dim_hidden = 1024                 # size of the common embedding vector
dim_attention = 512                   # size of attention embedding
num_output = 1000                       # number of output answers
max_words_q = 26
[IMG_HEIGHT, IMG_WIDTH] = [224, 224]
itoa = json.load(open(itoa_json, 'r'))
########################################################## 

def extract_feat(doc):
    feat = []
    for word in doc:
        try:
            feat.append(model_w2v[word])
        except:
            pass
    return feat

def prepro_text(ques):

    # tokenize question
    txt = word_tokenize(ques.lower())
    # encode question (left_align)
    label_arrays = np.zeros((1, max_words_q, input_embedding_size), dtype='float32')
    feat = np.array(extract_feat(txt))
    label_length = min(max_words_q, len(feat)) # record the length of this sequence
    label_arrays[0, :label_length, :] = feat

    return label_arrays

def extract_imfeat(img):

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(cnn_proto, cnn_model, caffe.TEST)
    mean = np.array((103.939, 116.779, 123.680), dtype=np.float32)

    I = misc.imresize(img, (IMG_HEIGHT, IMG_WIDTH))-mean
    I = np.transpose(I, (2, 0, 1))
    net.blobs['data'].data[:] = np.array([I])
    net.forward()
    imfeat = net.blobs['pool5'].data[...].copy()
    mag = np.sqrt(np.sum(np.multiply(imfeat, imfeat), axis=1))
    imfeat = np.transpose(imfeat,(0,2,3,1))
    imfeat = np.divide(imfeat, np.transpose(np.tile(mag,[512,1,1,1]),(1,2,3,0)) + 1e-8)
    return imfeat

def vis_attention(img, att_map):
    
    att_map = np.reshape(att_map, [7,7])
    att_map = att_map.repeat(32, axis=0).repeat(32, axis=1)
    att_map = np.tile(np.expand_dims(att_map, 2),[1,1,3])
    att_map[:,:,1:] = 0
    # apply gaussian
    att_map = gaussian_filter(att_map, sigma=7)
    att_map = (att_map-att_map.min()) / att_map.max()
    att_map = cv2.resize(att_map, (img.shape[1], img.shape[0]))
    new_img = att_map*255*0.8 + img*0.2

    return new_img

def test(model_path='model/model_lstm_att2/model-70000'):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = 1,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            dim_attention = dim_attention,
            max_words_q = max_words_q,
            drop_out_rate = 0,
            sess = sess)

    model.build_generator()

    sess.run(tf.initialize_all_variables())
    t_vars = tf.trainable_variables()

    with tf.device('/cpu: 0'):
        saver = tf.train.Saver(t_vars)
        saver.restore(sess, model_path)
    
    while 1:
        if os.path.isfile(vqa_data_json):
            # get data
            vqa_data = json.load(open(vqa_data_json, 'r'))
            ques = vqa_data['question']
            imname = os.path.join(web_path, vqa_data['image'].encode('utf-8')[1:])
            img = cv2.imread(imname).astype(np.float32)
            os.remove(vqa_data_json)

            question = prepro_text(ques)
            imfeat = extract_imfeat(img)

            generated_ans, prob_att1, prob_att2 = model.test(imfeat, question)
            generated_ans = np.argmax(generated_ans, axis=1)[0]
            att1 = vis_attention(img, prob_att1)
            att2 = vis_attention(img, prob_att2)
            cv2.imwrite(att_image, att2.astype(np.uint8))

            ans = itoa[str(generated_ans+1)]
            with open(answer_txt, 'w') as f:
                f.write(ans)
            print('Q: {}'.format(ques))
            print('A: {}'.format(ans))
        time.sleep(1)

if __name__ == '__main__':
    with tf.device('/gpu:0'):
        test(vqa_model)

