#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:15:26 2018

@author: Joe
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP

mnist = np.load('mnist_scaled.npz')
mnist.files




X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train', 
                                    'X_test', 'y_test']]


n_epochs = 200

## @Readers: PLEASE IGNORE IF-STATEMENT BELOW
##
## This cell is meant to run fewer epochs when
## the notebook is run on the Travis Continuous Integration
## platform to test the code on a smaller dataset
## to prevent timeout errors; it just serves a debugging tool

if 'TRAVIS' in os.environ:
    n_epochs = 20




nn = NeuralNetMLP(n_hidden=100, 
                  l2=0.01, 
                  epochs=n_epochs, 
                  eta=0.0005,
                  minibatch_size=100, 
                  shuffle=True,
                  seed=1)

nn.fit(X_train=X_train[:55000], 
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])


# ---
# **Note**
# 
# In the fit method of the MLP example above,
# 
# ```python
# 
# for idx in mini:
# ...
#     # compute gradient via backpropagation
#     grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
#                                       a3=a3, z2=z2,
#                                       y_enc=y_enc[:, idx],
#                                       w1=self.w1,
#                                       w2=self.w2)
# 
#     delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
#     self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
#     self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
#     delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
# ```
# 
# `delta_w1_prev` (same applies to `delta_w2_prev`) is a memory view on `delta_w1` via  
# 
# ```python
# delta_w1_prev = delta_w1
# ```
# on the last line. This could be problematic, since updating `delta_w1 = self.eta * grad1` would change `delta_w1_prev` as well when we iterate over the for loop. Note that this is not the case here, because we assign a new array to `delta_w1` in each iteration -- the gradient array times the learning rate:
# 
# ```python
# delta_w1 = self.eta * grad1
# ```
# 
# The assignment shown above leaves the `delta_w1_prev` pointing to the "old" `delta_w1` array. To illustrates this with a simple snippet, consider the following example:
# 
# 




a = np.arange(5)
b = a
print('a & b', np.may_share_memory(a, b))


a = np.arange(5)
print('a & b', np.may_share_memory(a, b))


# (End of note.)
# 
# ---




plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
#plt.savefig('images/12_07.png', dpi=300)
plt.show()




plt.plot(range(nn.epochs), nn.eval_['train_acc'], 
         label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], 
         label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
#plt.savefig('images/12_08.png', dpi=300)
plt.show()




y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred)
       .astype(np.float) / X_test.shape[0])

print('Test accuracy: %.2f%%' % (acc * 100))




miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.savefig('images/12_09.png', dpi=300)
plt.show()
