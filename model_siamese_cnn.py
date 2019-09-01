import tensorflow as tf
import tensorflow.contrib.slim as slim

class Siamese_CNN:
    embedding_size = 300
    drop_keep_prob = 1.0
    filter_sizes = [2,3,4,5]
    filter_nums = 30

class Siamese_CNN:
    def __init__(self, config, vocab_len, word_vector, max_sequence_len):
        self.input_x1 = tf.placeholder(tf.int32, [None, None], name = 'input_x1')
        self.input_x2 = tf.placeholder(tf.int32, [None, None], name = 'input_x2')
        self.input_y = tf.placeholder(tf.int32, [None], name = 'intput_y')

        with tf.name_scope('embedding'):
            if word_vector:
                self.W = tf.Variable(tf.cast(word_vector, tf.float32, name ='embedding' ), name = 'embedding')
            else:
                self.W = tf.Variable(tf.random_normal([vocab_len, word_vector], mean=0.0, stddev= 0.1), name = 'embedding')

            self.embed_word_1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embed_word_1 = tf.nn.dropout(self.embed_word_1, keep_prob= config.drop_keep_prob)
            self.embed_word_2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            self.embed_word_2 = tf.nn.dropout(self.embed_word_2, keep_prob= config.drop_keep_prob)

        with tf.variable_scope("conv"):
            pooled_output_1 = []
            for i, filter_size in enumerate(config.filter_sizes):
                conv = slim.conv2d(self.embed_word_1, num_outputs = config.filter_nums, kernel_size=[filter_size, config.embedding], padding='VALID',
                                   stride = 1, activation_fn=tf.nn.leaky_relu, scope='conv{}'.format(filter_size)
                                   )
                pooled = slim.max_pool2d(conv, kernel_size = [max_sequence_len - filter_size + 1, 1])
                pooled_output_1.append(pooled)

            num_filters_total = config.filter_nums * len(config.filter_nums)
            self.pool = tf.concat(pooled_output_1, 3)
            self.pool_1 = tf.reshape(self.pool, [-1, num_filters_total])

        with tf.variable_scope("conv", reuse=True):
            pooled_output_2 = []
            for i, filter_size in enumerate(config.filter_sizes):
                conv = slim.conv2d(self.embed_word_2, num_outputs = config.filter_nums, kernel_size=[filter_size, config.embedding], padding='VALID',
                                   stride = 1, activation_fn=tf.nn.leaky_relu, scope='conv{}'.format(filter_size)
                                   )
                pooled = slim.max_pool2d(conv, kernel_size = [max_sequence_len - filter_size + 1])
                pooled_output_2.append(pooled)

            num_filters_total = config.filter_nums * len(config.filter_nums)
            self.pool = tf.concat(pooled_output_2, 3)
            self.pool_2 = tf.reshape(self.pool, [-1, num_filters_total])

        with tf.name_scope('Manhattan_distance'):
            self.manhattan_distance =  tf.reduce_sum(tf.abs(self.pool_1 - self.pool_2), axis=-1)
            self.manhattan_distance_exp = tf.exp(-self.manhattan_distance)


        with tf.name_scope('loss'):
            self.pred_y = self.manhattan_distance_exp
            self.loss = tf.reduce_mean(tf.sqrt(self.pred_y - self.input_y))




