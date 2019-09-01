import tensorflow as tf

class Siamese_LSTM_Condif:
    file_path = 'task3_train.txt'
    vocab_path = 'vocab.json'
    learning_rate = 0.01
    embedding_size = 300
    drop_keep_prob = 1.0
    hidden_sizes = [100,150]
    Bidirection = False
    batch_size = 32
    epoch = 50
    checkpoint_every = 1000

class Siamese_LSTM:
    def __init__(self, config, vocab_len, word_vector):
        self.input_x1 = tf.placeholder(tf.int32, [None, None], name = "input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, None], name = "input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name = "input_y")
        self.sequence_1_len = tf.placeholder(tf.int32,[None], name = "sequence_1_len")
        self.sequence_2_len = tf.placeholder(tf.int32,[None], name = "sequence_2_len")
        self.config = config

        with tf.name_scope('embedding'):
            if word_vector is not None:
                self.W = tf.Variable(tf.cast(word_vector, dtype=tf.float32, name="word_vector"), name= 'embedding')
                print("<<<use word2vev>>>")
            else:
                self.W = tf.Variable(tf.truncated_normal([vocab_len, config.embedding_size],mean=0.0, stddev= 1.0), name='embedding')
                #tf.summary.histogram('word embedding', self.W)
                print("<<<use random std_normal distribution word vector>>>")

            self.embed_word_1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embed_word_1 = tf.nn.dropout(self.embed_word_1, keep_prob=config.drop_keep_prob)
            self.embed_word_2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            self.embed_word_2 = tf.nn.dropout(self.embed_word_2, keep_prob=config.drop_keep_prob)

        with tf.name_scope('LSTM'):  ## 这里共享权值，使用了reuse。 另一种实现方法（还未测试）：不使用两个rnn，直接把sentence_1 和 sentence_2 输入到同一个rnn里面
            if config.Bidirection:
                for idx, hidden_size in enumerate(config.hidden_sizes):
                    with tf.variable_scope('Bi_LSTM' + str(idx)) as scope:
                        lstm_fw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units = hidden_size, reuse = True),
                                                                     state_keep_prob = config.drop_keep_prob)
                        lstm_bw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units= hidden_size, reuse = True),
                                                                     state_keep_prob = config.drop_keep_prob)

                        output_1, last_step_output_1 = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, self.embed_word_1,
                                                                                       sequence_length = self.sequence_1_len,
                                                                                       dtype= tf.float32)

                        lstm_fw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units = hidden_size, reuse = True),
                                                                       state_keep_prob = config.drop_keep_prob)
                        lstm_bw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units = hidden_size, reuse = True),
                                                                       state_keep_prob = config.drop_keep_prob)

                        output_2, last_step_output_2 = tf.nn.bidirectional_dynamic_rnn(lstm_bw_cell_2, lstm_bw_cell_2, self.embed_word_2,
                                                                                       sequence_length = self.sequence_1_len,
                                                                                       dtype = tf.float32)


            else:
                for idx, hidden_size in enumerate(config.hidden_sizes):
                    with tf.variable_scope('LSTM' + str(idx)) as scope:

                        lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units= hidden_size),
                                                                    state_keep_prob = config.drop_keep_prob)

                        output_1, self.state_1 = tf.nn.dynamic_rnn(lstm_cell_1, self.embed_word_1,
                                                                         sequence_length = self.sequence_1_len,
                                                                         dtype = tf.float32)

                        # lstm_cell_2 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units= hidden_size, reuse = True),
                        #                                             state_keep_prob = config.drop_keep_prob)

                        output_2, self.state_2 = tf.nn.dynamic_rnn(lstm_cell_1, self.embed_word_2,
                                                                         sequence_length = self.sequence_2_len,
                                                                         dtype = tf.float32)

                        self.embed_word_1 = output_1
                        self.embed_word_2 = output_2


                # [batch_size, time_steps, hidden_size]  [-1, batch_size, hidden_size]
                col_1 = self.sequence_1_len - 1
                row = tf.range(config.batch_size)
                col_2 = self.sequence_2_len - 1
                index_1 = tf.unstack(tf.stack([row, col_1], axis=0), axis=1)
                index_2 = tf.unstack(tf.stack([row, col_2], axis=0), axis=1)


                self.last_layer_last_step_1 = tf.gather_nd(self.embed_word_1, index_1)
                self.last_layer_last_step_2 = tf.gather_nd(self.embed_word_2, index_2)
                # self.last_layer_last_step_1 = self.embed_word_1[:, -1, :]
                # self.last_layer_last_step_2 = self.embed_word_2[:, -1, :]

        with tf.name_scope('Manhattan_distance'):
            self.manhattan_distance =  tf.reduce_mean(tf.pow(self.last_layer_last_step_1 - self.last_layer_last_step_2,2) ,axis=-1)
            self.manhattan_distance_exp = tf.exp(-self.manhattan_distance)


        with tf.name_scope('loss'):
            self.pred_y = self.manhattan_distance_exp
            binary_cross_entry = self.input_y * tf.log1p(self.pred_y) + (1 - self.input_y) * tf.log1p(1 - self.pred_y)
            self.loss = tf.reduce_mean(binary_cross_entry, name='loss')
            #tf.summary.scalar('loss', self.loss)




