from nltk.tokenize import word_tokenize
from scipy.misc import imresize
import numpy as np
import caffe
import cv2
import tensorflow as tf
import json
import os
import time
import pdb

rnn_cell = tf.nn.rnn_cell

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_image, dim_hidden, max_words_q, vocabulary_size, drop_out_rate, sess):

        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.input_embedding_size = input_embedding_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.max_words_q = max_words_q
        self.vocabulary_size = vocabulary_size
        self.drop_out_rate = drop_out_rate
        self._sess = sess

        self.image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        self.question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])

        # question-embedding
        self.embed_ques_W = tf.Variable(tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_ques_W')

        # encoder: RNN body
        self.lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True)
        self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = 1 - self.drop_out_rate)
        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True)
        self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = 1 - self.drop_out_rate)
        self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])

        # state-embedding
        self.embed_state_W = tf.Variable(tf.random_uniform([2*rnn_size*rnn_layer, self.dim_hidden], -0.08,0.08),name='embed_state_W')
        self.embed_state_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_state_b')
        # image-embedding
        self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, self.dim_hidden], -0.08, 0.08), name='embed_image_W')
        self.embed_image_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_image_b')
        # score-embedding
        self.embed_scor_W = tf.Variable(tf.random_uniform([dim_hidden, num_output], -0.08, 0.08), name='embed_scor_W')
        self.embed_scor_b = tf.Variable(tf.random_uniform([num_output], -0.08, 0.08), name='embed_scor_b')
   
    def build_generator(self):
        
        state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])
        loss = 0.0
        for i in range(max_words_q):
            if i==0:
                ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
            else:
                tf.get_variable_scope().reuse_variables()
                ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, self.question[:,i-1])
            ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
            ques_emb = tf.tanh(ques_emb_drop)

            output, state = self.stacked_lstm(ques_emb, state)
        
        # multimodal (fusing question & image)
        state_drop = tf.nn.dropout(state, 1-self.drop_out_rate)
        state_linear = tf.nn.xw_plus_b(state_drop, self.embed_state_W, self.embed_state_b)
        state_emb = tf.tanh(state_linear)

        image_drop = tf.nn.dropout(self.image, 1-self.drop_out_rate)
        image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(image_linear)

        scores = tf.mul(state_emb, image_emb)
        scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)

        # FINAL ANSWER
        self.generated_ANS = tf.nn.softmax(tf.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b))

    def test(self, image, question):

        generated_ans = self._sess.run(
            self.generated_ANS,
            feed_dict={
                self.image: image,
                self.question: question
            })
        return generated_ans

########################################################## 
web_path = '/path/to/your/website'
vqa_data_json = os.path.join(web_path, 'vqa_data.json')
answer_txt = os.path.join(web_path, 'media/vqa_ans.txt')
#ques = 'What is this photo taken looking through?'
#imname = 'test.jpg'

# need to be saved from prepro.py
itoa_json = 'ix_to_ans.json'            # mapping idx to answer
wtoi_json = 'word_to_ix.json'           # mapping word to idx
cnn_proto = 'vgg19/deploy.prototxt'
cnn_model = 'vgg19/VGG_ILSVRC_19_layers.caffemodel'
vqa_model = '/path/to/your/vqa_model'

input_embedding_size = 200              # the encoding size of each token in the vocabulary
rnn_size = 512                          # size of the rnn in number of hidden nodes in each layer
rnn_layer = 2                           # number of the rnn layer
dim_image = 4096
dim_hidden = 1024                       # size of the common embedding vector
num_output = 1000                       # number of output answers
max_words_q = 26
[IMG_HEIGHT, IMG_WIDTH] = [224, 224]
itoa = json.load(open(itoa_json, 'r'))
wtoi = json.load(open(wtoi_json, 'r'))
vocabulary_size = len(wtoi.keys())
########################################################## 

def prepro_text(ques):

    # tokenize question
    txt = word_tokenize(ques.lower())
    question = [w if wtoi.get(w, len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
    print(question)

    # encode question (right_align)
    label_arrays = np.zeros((1, max_words_q), dtype='uint32')
    label_length = min(max_words_q, len(question)) # record the length of this sequence
    for i in range(2, label_length+1):
        label_arrays[0, -i] = wtoi[question[-i]]-1
    return label_arrays

def extract_imfeat(imname):

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(cnn_proto, cnn_model, caffe.TEST)
    mean = np.array((103.939, 116.779, 123.680), dtype=np.float32)

    I = imresize(cv2.imread(imname), (IMG_HEIGHT, IMG_WIDTH))-mean
    I = np.transpose(I, (2, 0, 1))
    net.blobs['data'].data[:] = np.array([I])
    net.forward()
    imfeat = net.blobs['fc7'].data[...].copy()
    mag = np.sqrt(np.sum(np.multiply(imfeat, imfeat), axis=1))
    imfeat = np.divide(imfeat, np.transpose(np.tile(mag,(4096,1))))
    return imfeat

def test(model_path='../model_save/model-15000'):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = 1,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            max_words_q = max_words_q,
            vocabulary_size = vocabulary_size,
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
            os.remove(vqa_data_json)

            question = prepro_text(ques)
            imfeat = extract_imfeat(imname)

            generated_ans = model.test(imfeat, question)
            generated_ans = np.argmax(generated_ans, axis=1)[0]

            ans = itoa[str(generated_ans+1)]
            with open(answer_txt, 'w') as f:
                f.write(ans)
            print('Q: {}'.format(ques))
            print('A: {}'.format(ans))
        time.sleep(1)

if __name__ == '__main__':
    with tf.device('/gpu:0'):
        test(vqa_model)

