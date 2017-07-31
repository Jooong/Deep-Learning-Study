#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:01:41 2017

@author: root
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import trange

poi_array = np.fromfile("/home/ubuntu/notebooks/imgs/poisonous_fungi_array.npy")
edi_array = np.fromfile("/home/ubuntu/notebooks/imgs/edible_fungi_array.npy")

idx_5000 = 5000 * (224*224*3)
poi_array = poi_array[:idx_5000]
edi_array = edi_array[:idx_5000]

poi_array = poi_array.reshape((5000,224,224,3)).astype("float32")
edi_array = edi_array.reshape((5000,224,224,3)).astype("float32")

X_array = np.concatenate((poi_array,edi_array))
Y_array = np.concatenate((np.array([[1,0] for _ in range(5000)]),np.array([[0,1] for _ in range(5000)]))).astype("float32")
X_train, X_test, Y_train, Y_test = train_test_split(X_array,Y_array, test_size=0.1, random_state=42)


X = tf.placeholder(tf.float32,[None,224,224,3])

# layer 1
W1 = tf.Variable(tf.random_normal([3,3,3,32],stddev=0.01))    ; print(W1)  # 3*3 filter가 32개
W1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1],padding="SAME")    ; print(conv1)#  224*224 size의 그림 * 32개
W1 = tf.nn.relu(W1)   ; print(relu1)
W1 = tf.nn.max_pool(W1, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") ; print(pool1) # 112*112의 그림 * 32개

# layer 2
W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))    ; print(W2)  # 3*3 filter가 64개
W2 = tf.nn.conv2d(W1, W2, strides=[1,1,1,1],padding="SAME")    ; print(conv2)#  112*112 size의 그림 * 64개
W2 = tf.nn.relu(W2)   ; print(relu2)
W2 = tf.nn.max_pool(W2, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") ; print(pool2) # 56*56의 그림 * 64개

# layer 3
W3 = tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))    ; print(W3)  # 3*3 filter가 128개
W3 = tf.nn.conv2d(W2, W3, strides=[1,1,1,1],padding="SAME")    ; print(conv3)#  56*56 size의 그림 * 128개
W3 = tf.nn.relu(W3)   ; print(relu3)
W3 = tf.nn.max_pool(W3, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") ; print(pool3) # 28*28의 그림 * 128개

# layer 4
W4 = tf.Variable(tf.random_normal([3,3,128,256],stddev=0.01))    ; print(W4)  # 3*3 filter가 256개
W4 = tf.nn.conv2d(W3, W4, strides=[1,1,1,1],padding="SAME")    ; print(conv4)#  28*28 size의 그림 * 256개
W4 = tf.nn.relu(W4)   ; print(relu4)
W4 = tf.nn.max_pool(W4, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") ; print(pool4) # 14*14의 그림 * 256개

# layer 5
W5 = tf.Variable(tf.random_normal([3,3,256,512],stddev=0.01))    ; print(W5)  # 3*3 filter가 512개
W5 = tf.nn.conv2d(W4, W5, strides=[1,1,1,1],padding="SAME")    ; print(conv5)#  14*14 size의 그림 * 512개
W5 = tf.nn.relu(W5)   ; print(relu5)
W5 = tf.nn.max_pool(W5, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") ; print(pool5) # 7*7의 그림 * 512개
flattened = tf.reshape(W5, [-1, 7*7*512])

# fully connected
D1 = tf.layers.dense(flattened,4096,activation=tf.nn.relu)
D1 = tf.nn.dropout(D1,0.5)

D2 = tf.layers.dense(D1,4096,activation=tf.nn.relu)
D2 = tf.nn.dropout(D2,0.5)

logit = tf.layers.dense(D2,2,activation=tf.nn.softmax)


sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
Y = tf.placeholder(tf.float32,shape=[None,2])


lr = 0.001
training_epochs = 10
batch_size = 100
idx = 0
i = 0

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(cost)

sess.run(tf.global_variables_initializer())


print('Learning started. It takes sometime.')

for epoch in trange(training_epochs):

    avg_cost = 0
    
    total_batch = int(X_train.shape[0]/ batch_size)

    for i in trange(total_batch):        
        batch_xs, batch_ys = X_train[idx:idx+batch_size], Y_train[idx:idx+batch_size]
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        Y_ = tf.stack(batch_ys)
        correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        print('batch:', '%05d' % (i + 1), 'accuracy =', '{:.9f}'.format(sess.run(accuracy,feed_dict={X: batch_xs, 
                                                                        Y: batch_ys})))
        i += 1
        idx += batch_size

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))


print('Learning Finished!')
