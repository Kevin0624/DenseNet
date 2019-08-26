import tensorflow as tf
import cifar10_data_load as cdl
import matplotlib.pyplot as plt
import layers as layers
import numpy as np

# input training data

train_images = cdl.GetTrainDataByLabel("data")
train_labels = cdl.GetTrainDataByLabel("labels")

test_images = cdl.GetTestDataByLabel("data")
test_labels = cdl.GetTestDataByLabel("labels")

input_images = tf.placeholder(tf.float32, shape=(None, 32, 32, 3)) # bfloat16
input_labels = tf.placeholder(tf.float32, shape=(None, 10)) # one-hot encoding


# print(train_images.shape)
# print(np.array(train_labels).shape)

# print("==============================")

# print(test_images.shape)
# print(np.array(test_labels).shape)