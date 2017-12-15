import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
import os
from keras.backend.tensorflow_backend import set_session
import time
import random
import pickle

EMBEDDING_DIM = 200
WINDOW_SIZE = 4
mini_batch=100
alpha=0.9
n_iters = 1
lr=0.0000001

def configure(use_cpu=False, gpu_memory_fraction=0.25, silence_warnings=True):
    if silence_warnings:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if use_cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.25
        set_session(tf.Session(config=config))

configure(use_cpu=True)        
#Read the essays and tokenize each word to a number
essays=pd.read_csv('Data/AES/training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')[['essay_id','essay_set','essay','domain1_score']]
tokenizer=Tokenizer()
tokenizer.fit_on_texts(essays['essay'])
essay_sequences=tokenizer.texts_to_sequences(essays['essay'])

#Normalize the scores
normalised_scores = []
minimum_scores = [-1, 2, 1, 0, 0, 0, 0, 0, 0]
maximum_scores = [-1, 12, 6, 3, 3, 4, 4, 30, 60]
for index, row in essays.iterrows():
    score = row['domain1_score']
    essay_set = row['essay_set']
    normalised_score = (score - minimum_scores[essay_set]) / (maximum_scores[essay_set] - minimum_scores[essay_set])
    normalised_scores.append(normalised_score)
essay_scores=normalised_scores

vocab_size=len(tokenizer.word_index.items())

#load embedding matrix
with open('Glove/glove.6B.200d.txt') as file:
        embeddings = {}
        for line in file:
            values = line.split()
            embeddings[values[0]] = np.array(values[1:], 'float64')
embedding_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#make training data
data = []
for sentence,score in zip(essay_sequences,essay_scores):
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1]: 
            if nb_word != word:
                data.append([word, nb_word,score])

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

n_ex=len(data)

# making placeholders for x_train, y_train and yscore_label
x = tf.placeholder(tf.float64, shape=(None, vocab_size + 1))
y_label = tf.placeholder(tf.float64, shape=(None, vocab_size + 1))
yscore_label=tf.placeholder(tf.float64, shape=(None, 1))

W1 = tf.Variable(embedding_matrix)
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM],dtype=tf.float64)) #bias
hidden_representation = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size + 1],dtype=tf.float64))
b2 = tf.Variable(tf.random_normal([vocab_size + 1],dtype=tf.float64))

W3=tf.Variable(tf.random_normal([EMBEDDING_DIM,1],dtype=tf.float64))
b3 = tf.Variable(tf.random_normal([1],dtype=tf.float64))

prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))
prediction_score= tf.add(tf.matmul(hidden_representation,W3), b3)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
score_loss=tf.reduce_sum(tf.pow(prediction_score-yscore_label,2))/(2*mini_batch)
loss=(alpha*score_loss)+((1-alpha)*cross_entropy_loss)

#define one training step
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# train for n_iter iterations
random.seed(42)
random.shuffle(data)
for j in range(n_iters):
	print("Epoch:{}".format(j))
	for i in range(0,n_ex,mini_batch):
		x_train = [] # input word
		y_train = [] # output word
		yscore_train=[] #output score
		for data_word in data[i:i+mini_batch]:
			x_train.append(to_one_hot(data_word[0],vocab_size+1))
			y_train.append(to_one_hot(data_word[1],vocab_size+1))
			yscore_train.append([data_word[2]])
		sess.run(train_step, feed_dict={x: np.asarray(x_train), y_label: np.asarray(y_train), yscore_label: np.asarray(yscore_train)})
		print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: np.asarray(x_train), y_label: np.asarray(y_train), yscore_label: np.asarray(yscore_train)}))

#trained embeddings
embedding_vectors = sess.run(W1 + b1)

with open('SSWE','wb') as fp:
	pickle.dump(embedding_vectors.tolist(),fp)

