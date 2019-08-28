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


# ResNet 是 add  , DenseNet 是 concatenate

# section 1
W_conv1 = layers.weight_variable([7, 7, 3, 16])
b_conv1 = layers.bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(input_images, W_conv1) + b_conv1)
h_pool1 = layers.max_pool(h_conv1, 3, 2)

# Dense Block 1


# Transition Layer 1


# Dense Block 2


# Trainsition Layer 2


# Dense Block 3


# Trainsition Layer 3


# Dense Block 4


# Classification Layer
# 7*7 global average pool
# 1000 fullyconected , softmax


