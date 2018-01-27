#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:57:06 2018

@author: Joe
"""
import pickle
class ObjectTemp:
#    def __init__(self, filename):
        #self.f = filename
    def restore(f):
        pickle_in = open(f,"rb")
        object_new = pickle.load(pickle_in)
        return object_new
    def save(object_out,f):        
        pickle_out = open(f,"wb")
        pickle.dump(object_out, pickle_out)
        pickle_out.close()
#a = 'test'
#ot = ObjectTemp();
#ot.save(a,"test")
#b = ot.restore(a)
#print(b)
#you can use  from charrnn import CharRNN