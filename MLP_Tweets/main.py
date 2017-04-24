#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:35:52 2017
@Module Description: This module trains and tests a MLP for sentiment analysis
@See: https://pythonprogramming.net/data-size-example-tensorflow-deep-learning-tutorial/
"""


import os
path = ‘your_own_path’
os.chdir(path)


import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 1500
n_nodes_hl2 = 1000
n_nodes_hl3 = 500

n_classes = 3

batch_size = 128
total_batches = int(1600000/batch_size)
    
# claim x and y and their type
x = tf.placeholder('float', [None, 3335])
y = tf.placeholder('float')

# specify the MLP topology
#data_width = int(data.shape[1])
hidden_1_layer = {'f_fum':n_nodes_hl1,
              'weight':tf.Variable(tf.random_normal([3335, n_nodes_hl1])),
              'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
              'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
              'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
              'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
              'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
            'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
            'bias':tf.Variable(tf.random_normal([n_classes])),}  

"""
@Function Description: This function specify a MLP topology, and will be called in the NN training function
@Author: Haosheng Luo
@Param: The functon takes in folowing input parameters:
    Hidden layers and nodes
    Nparrays of independent variables
    # of sentiment types
@Return: The function returns the values in output layer in each batch of training 
@Business logic: NA 
"""

def neural_network_model(data):

    # compute values in each layer in a feed-forward way
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)
    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']
    return output

# create a saver object to save and restore checkpoint files
save = tf.train.Saver()

"""
@Function Description: This function implements the following tasks:
    Train a MLP through minimizing the cross entropy 
    Parse the text inline so as to accommodate big data
    Save the checkpoint files of training 
@Author: Haosheng Luo    
@Param: The functon takes in folowing input parameters:
    Working directory
    Cleaned text and the associated sentiment
    Lexicon
    # of epoch  
@Return: The function overwrites the checkpoint files of model parameters in each epoch, to avoid unexpected interrupt of training 
@Business logic: NA 
"""

def train_neural_network(train_set, lexicon, saver, hm_epochs = 15):
    
    # create a folder to store checkpoint file of model training
    meta_saving_path = 'training_meta/'
    full_path = path + meta_saving_path
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    # specify the model, the loss fucntion and optimizer
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # create a log file for referencing epochs
    tf_log = full_path+'tf.log'

    # start the tf session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # read the epoch # in the log
        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:', epoch)
        except:
            epoch = 1
            
        # load lexicon
        with open(lexicon,'rb') as f:
                lexicon = pickle.load(f)
         
        # start training   
        print('Start training | Current time:',datetime.now().strftime('%H:%M:%S'),)
        while epoch <= hm_epochs:
            if epoch != 1:
                
                # restore the previous checkpoint file
                save_path = './'+meta_saving_path+'model'
                saver.restore(sess, save_path)

            epoch_loss = 0
            
            # parse the text and convert it into vectors of predictors inline
            with open(train_set, buffering=2500000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = eval(line.split(':::')[0])
                    batch_y.append(label)
                    
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = [0]*len(lexicon)

                    for word in current_words:
                        if word in lexicon:
                            index_value = lexicon.index(word)
                            features[index_value] += 1
                    
                    batch_x.append(features)
                    
                    # train MLP / tweak params after a predefined batch of data parsed
                    if len(batch_x) >= batch_size:
                        batch_x = np.array(batch_x)
                        batch_y = np.array(batch_y)
                        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        if (batches_run/2000).is_integer():
                            print('Batch run:',batches_run,'/',total_batches,'| Epoch:',epoch,'| Batch Loss:',c,
                                  '| Current time:',datetime.now().strftime('%H:%M:%S'),)

            # overwrite the previous checkpoint file, display the progress and write training logs
            saver.save(sess, "./"+meta_saving_path+"model")
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            print('Model saved to', full_path)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n') 
            epoch += 1

train_neural_network(train_set = 'train_set_shuffled.csv', lexicon ='lexicon.pickle', saver = save, hm_epochs = 10)

"""
@Function Description: This function implements the following tasks:
    Test a MLP 
    Parse the text inline so as to accommodate big data
@Author: Haosheng Luo    
@Param: The functon takes in folowing input parameters:
    Working directory
    Cleaned text for test and the associated sentiment
    Lexicon
    # of epoch  
@Return: The function returns the accuracy of sentiment prediction
@Business logic: NA 
"""

test_set = 'test_set_processed.pickle'

def test_neural_network(test_set, saver):
    
    meta_saving_path = 'training_meta/'
    
    x = tf.placeholder('float')
    y = tf.placeholder('float')
    
    prediction = neural_network_model(x)
    #saver = tf.train.Saver()
    
    with open(test_set,'rb') as f:
        test_set_processed = pickle.load(f)
    f.close()
          
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #save_path = './'+meta_saving_path+'model.meta'
        #new_saver = tf.train.import_meta_graph(save_path)
        save_path = './'+meta_saving_path+'model'
        #new_saver.restore(sess, save_path)
        saver.restore(sess, save_path)
        
        test_x = np.array([l[0] for l in test_set_processed])
        test_y = np.array([l[1] for l in test_set_processed])
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Tested',len(test_set_processed),'samples.')
        print('Accuracy:',accuracy.eval(feed_dict={x: test_x, y: test_y}))


test_neural_network(test_set = 'test_set_processed.pickle', saver = save)







