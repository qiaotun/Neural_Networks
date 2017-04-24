#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:12:16 2017

@author: luohaosheng
@function: identify MNIST handwriting numbers 
@model: MLP
@data: MNIST
@pre-requisite: MNIST installed
"""

import os
path = ‘your_own_path’
os.chdir(path)

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(path, one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 #feed the net with 100 images at a time (batch) to adjust the weight

# claim x and y w/ height * width (optional, but tf will not throw an error if the input shape is 
# different from expectation) and their type
x = tf.placeholder('float',[None, 784]) #input
y = tf.placeholder('float') #output

# build parameter dictionary
hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes]))}

def neural_network_model(data):   

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return output
    
save = tf.train.Saver()

#def train_neural_network(x = tf.placeholder('float',[None, 784]), y = tf.placeholder('float')):
def train_neural_network(saver, n_epoch=10):
    
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y)) #y: real label
    
    # default learning rate = 0.001 
    optimizer = tf.train.AdamOptimizer().minimize(cost)    

    # create a folder to store checkpoint file of model training
    meta_saving_path = 'training_meta/'
    full_path = path + meta_saving_path
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 1
        
        while epoch <= n_epoch:
            if epoch != 1:
                # restore the previous checkpoint
                save_path = './'+meta_saving_path+'model'
                saver.restore(sess, save_path)
                
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # in other dataset, this pre-built function doesn't work
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) 
                #                             it's like rename epoch_x and epoch_y into x and y
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y}) 
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', n_epoch, '; loss:', epoch_loss)
            
            # overwrite the previous checkpoint
            saver.save(sess, "./"+meta_saving_path+"model")
            #saver.save(sess, meta_saving_path+"model", global_step = epoch)
            epoch += 1
         
train_neural_network(saver = save)        


def test_neural_network(saver):
    
    meta_saving_path = 'training_meta/'
    
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float')
    
    prediction = neural_network_model(x)
    #saver = tf.train.Saver()
          
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #save_path = './'+meta_saving_path+'model.meta'
        #new_saver = tf.train.import_meta_graph(save_path)
        save_path = './'+meta_saving_path+'model'
        #new_saver.restore(sess, save_path)
        saver.restore(sess, save_path)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval(feed_dict={x: mnist.test.images, y:mnist.test.labels}))


test_neural_network(save)








