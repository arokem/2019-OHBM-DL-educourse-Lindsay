#import gzip
import pickle
import numpy as np
#import tensorflow as tf
#import matplotlib.pyplot as plt
#import sklearn.linear_model as skl
#import scipy.ndimage as sp


class DataObject:

    def __init__(self, b_size):

       f = open('trunc_data.pkl', 'rb')
       dataset = pickle.load(f,encoding='latin1') #has a list of images and a list of class labels
       f.close()

       images = dataset[0];
       images = (images - np.mean(images[:]))/np.std(images[:])
       labels = dataset[1];
       self.saved_images=images[0:200,:]
       self.saved_labels=labels[0:200]

       train_num = int(.8*len(labels)) #16k
       valtest_num = int(.1*len(labels))
       self.train_images = images[0:train_num,:]
       self.train_labels = labels[0:train_num]
       self.val_images = images[train_num:train_num+valtest_num,:]
       self.val_labels = labels[train_num:train_num+valtest_num]
       self.test_images = images[train_num+valtest_num:,:]
       self.test_labels = labels[train_num+valtest_num:]

       self.b_size = b_size
       self.tot_batch = train_num // self.b_size
       self.epoch = 0
       self.cur_batch = 0

    def get_trainbatch(self):
        train_num = len(self.train_labels)
        self.cur_batch += 1
        if self.cur_batch == self.tot_batch: #not using some data if batches dont fit in
            self.epoch += 1
            self.cur_batch = 0
            print('epoch '+ str(self.epoch))
            np.random.seed(self.epoch)
            rand_inds = np.random.choice(train_num,train_num, replace=False) #shuffle images with each epoch
            self.train_labels = self.train_labels[rand_inds] 
            self.train_images = self.train_images[rand_inds,:]

        batch_images = self.train_images[self.cur_batch*self.b_size:(self.cur_batch+1)*self.b_size,:]
        batch_labels = self.train_labels[self.cur_batch*self.b_size:(self.cur_batch+1)*self.b_size]    
        return batch_images, batch_labels


