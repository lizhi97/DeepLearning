#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:29:20 2018

@author: Joe
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]
g2 = tf.Graph()
with tf.Session(graph=g2) as sess:
    new_saver = tf.train.import_meta_graph(
        './trained-model.meta')
    new_saver.restore(sess, './trained-model')
    
    y_pred = sess.run('y_hat:0', 
                      feed_dict={'tf_x:0' : x_test})

print('SSE: %.4f' % (np.sum(np.square(y_pred - y_test))))

x_arr = np.arange(-2, 4, 0.1)

g2 = tf.Graph()
with tf.Session(graph=g2) as sess:
    new_saver = tf.train.import_meta_graph(
        './trained-model.meta')
    new_saver.restore(sess, './trained-model')
    
    y_arr = sess.run('y_hat:0', 
                      feed_dict={'tf_x:0' : x_arr})

plt.figure()
plt.plot(x_train, y_train, 'bo')
plt.plot(x_test, y_test, '^', alpha=0.3)
plt.plot(x_arr, y_arr.T[:, 0], '-r', lw=3)
# plt.savefig('images/14_05.png', dpi=400)
plt.show()