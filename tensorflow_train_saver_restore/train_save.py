#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:23:54 2018

@author: Joe
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

g = tf.Graph()

## define the computation graph
with g.as_default():
    ## placeholders
    
    tf.set_random_seed(123)
    tf_x = tf.placeholder(shape=(None), 
                          dtype=tf.float32, 
                          name='tf_x')
    tf_y = tf.placeholder(shape=(None), 
                          dtype=tf.float32,
                          name='tf_y')
    
    ## define the variable (model parameters)
    weight = tf.Variable(
        tf.random_normal(
            shape=(1, 1), 
            stddev=0.25),
        name = 'weight')
    bias = tf.Variable(0.0, name='bias')
    ## build the model
    y_hat = tf.add(weight * tf_x, bias, 
                   name='y_hat')
    print(y_hat)
    
    ## compute the cost
    cost = tf.reduce_mean(tf.square(tf_y - y_hat), 
                          name='cost')
    print(cost)
    ## train
    optim = tf.train.GradientDescentOptimizer(
        learning_rate=0.001)
    train_op = optim.minimize(cost, name='train_op')
    print('train_op:',train_op)

## create a random toy dataset for regression


np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, 
                             scale=(0.5 + t*t/3), 
                             size=None)
        y.append(r)
    return  x, 1.726*x -0.84 + np.array(y)


x, y = make_random_data() 

plt.plot(x, y, 'o')
# plt.savefig('images/14_03.png', dpi=300)
plt.show()
## launch the graph

x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]


## training the model
n_epochs = 500
training_costs = []

with tf.Session(graph=g) as sess:
    ## first, run the variables initializer
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ## train the model for n_epochs
    for e in range(n_epochs):
        c, _ = sess.run([cost, train_op], 
                        feed_dict={tf_x: x_train,
                                   tf_y: y_train})
        training_costs.append(c)
        if not e % 50:
            print('Epoch %4d: %.4f' % (e, c))
    saver.save(sess, './trained-model')


