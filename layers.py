import tensorflow as tf

# 權重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# 傳統的2D conv
def conv2d(x, W, stride_num):
    return tf.nn.conv2d(x, W, strides=[1, stride_num, stride_num, 1], padding='SAME')

# maxpooling
def max_pool(x, ksize_num,  stride_num):
    return tf.nn.max_pool(x, ksize=[1, ksize_num, ksize_num, 1], strides=[1, stride_num, stride_num, 1], padding='SAME')

# averagepooling
def average_pool(x, ksize_num, stride_num):
    return tf.nn.avg_pool(x, ksize=[1, ksize_num, ksize_num, 1], strides=[1, stride_num, stride_num, 1], padding="SAME")

# DenseBlock
def DenseBlock():
    
# batchnormalization