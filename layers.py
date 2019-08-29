import tensorflow as tf


class DenseNet():

    # 代表宣告時會自動執行的函式，是宣告類別的起手式
    def __init__(self,       
                num_classes,  # number of classes
                growth_rate, 
                bc_mode, 
                block_config, 
                reduction, 
                dropout_rate, 
                weight_decay, 
                nesterov_momentum):
        
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.bc_mode = bc_mode
        self.block_config = block_config
        self.reduction = reduction
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
    
    # 權重初始化
    def weight_variable(self, 
                        shape):

        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    # bias 初始化
    def bias_variable(self, 
                        shape):

        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(inital)

    # 傳統的 2D conv
    def conv2d(self, 
                x, 
                W, 
                stride_num):

        return tf.nn.conv2d(x, W, strides=[1, stride_num, stride_num, 1], padding = 'SAME')

    # Maxpooling 
    def max_pool(self, 
                x, 
                ksize_num, 
                stride_num):

        return tf.nn.max_pool(x, ksize = [1, ksize_num, ksize_num, 1], stides = [1, stride_num, stride_num, 1], padding = 'SAME')

    # Average Pooling
    def average_pool(self, 
                        x, 
                        ksize_num, 
                        stride_num):

        return tf.nn.avg_pool(x, ksize = [1, ksize_num, ksize_num, 1], strides = [1, stride_num, stride])

    # Batch Normalization
    def batch_norm(self, 
                    x, 
                    training):

        return tf.keras.layers.BatchNormalization(x, training = training)

    # Dropout
    def dropout(self, 
                x,  
                training):

        return tf.layers.dropout(x, rate = self.dropout_rate, training = training)
        
    # Composite function
    def composite_function(self, x, W, stride_num, training ):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        # batch normalization 
        output = self.batch_norm(x, training)

        # Relu
        output = tf.nn.relu(output)

        # Conv2d
        output = self.conv2d(output, W, stride_num)

        # dropout
        output = self.dropout(output, training)

        return output
    
    # bottleneck
    def bottleneck(self, x, W, stride_num, training ):

        #batch normalization
        output = self.batch_norm(x, training)

        #Relu
        output = tf.nn.relu(output)

        #Conv2d
        output = self.conv2d(output, W, stride_num)

        #dropout
        output = self.dropout(output, training)

        return output

    #



