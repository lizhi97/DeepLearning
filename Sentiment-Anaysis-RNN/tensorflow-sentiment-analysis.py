#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:21:53 2018

@author: Joe
"""
import gzip
import numpy as np
import pickle
import pandas as pd
from string import punctuation
from collections import Counter
from sentimentrnn import SentimentRNN


with gzip.open('movie_data.csv.gz') as f_in, open('movie_data.csv', 'wb') as f_out:
    f_out.writelines(f_in)
    
df = pd.read_csv('movie_data.csv', encoding='utf-8')

counts = Counter()
for i,review in enumerate(df['review']):
    text = ''.join([c if c not in punctuation else ' '+c+' ' for c in review]).lower()
    df.loc[i,'review'] = text
    counts.update(text.split())
word_counts = sorted(counts, key=counts.get, reverse=True)

word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}
mapped_reviews = []
for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])
 
mapped_reviews = load_object('./mapped_reviews')

sequence_length = 200  ## sequence length (or T in our formulas)
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)
for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    sequences[i, -len(row):] = review_arr[-sequence_length:]
X_train = sequences[:25000, :]
y_train = df.loc[:25000, 'sentiment'].values
X_test = sequences[25000:, :]
y_test = df.loc[25000:, 'sentiment'].values

np.random.seed(123)

n_words = max(list(word_to_int.values())) + 1

rnn = SentimentRNN(n_words=n_words, 
                   seq_len=sequence_length,
                   embed_size=256, 
                   lstm_size=128, 
                   num_layers=1, 
                   batch_size=100, 
                   learning_rate=0.001)

rnn.train(X_train, y_train, num_epochs=40)

## Test: 
preds = rnn.predict(X_test)
y_true = y_test[:len(preds)]
print('Test Acc.: %.3f' % (
      np.sum(preds == y_true) / len(y_true)))




## Get probabilities:
proba = rnn.predict(X_test, return_proba=True)
