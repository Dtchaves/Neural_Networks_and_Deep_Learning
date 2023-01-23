import numpy as np
import matplotlib.pyplot as plt
import h5py

    
def load_dataset():
  train_dataset = h5py.File('/home/diogo/Documentos/IC/Neural_Networks_and_Deep_Learning_exercices/Cat_identifier_logistic_regression/dataset/train_catvnoncat.h5', "r")
  train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
  train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

  test_dataset = h5py.File('/home/diogo/Documentos/IC/Neural_Networks_and_Deep_Learning_exercices/Cat_identifier_logistic_regression/dataset/test_catvnoncat.h5', "r")
  test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
  test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

  classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
  train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
  test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
  return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
  
def initialize_with_zeros(dim):
  w = np.zeros((dim,1))
  b = 0
  assert(w.shape == (dim, 1))
  assert(isinstance(b, float) or isinstance(b, int))
  return w, b

def normalize_data(x_orig):
  x = x_orig.reshape(x_orig.shape[0], -1).T
  return x

def sigmoid(x):
  sig = 1/(1+np.exp(-x))
  return sig

