import tensorflow as tf

class DSSM_LSTM_config:
    file_path = 'task3_train.txt'  #"dssm_data.txt"
    vocab_path = 'vocab.json'
    embedding = 100
    drop_keep_prob = 1.0
    hidden_sizes = [128]
    batch_size = 32
    epoch = 10
    learning_rate = 0.005
    embedding_size = 300
    Bidirection = False
    evaluate_every = 1000
    checkpoint_every = 1000
    neg = 10


class DSSM_LSTM:
    def __init__(self, config,vocab_len, word_vector, train = True):
        if train:
            self.query = tf.placeholder(tf.int32, [None, None], name='input_x1')
            self.query_len = tf.placeholder(tf.int32,[None], name='query_len')
            self.doc = tf.placeholder(tf.int32, [config.neg+1, None, None], name='input_x2')
            self.doc_len = tf.placeholder(tf.int32, [config.neg+1,None], name='doc_1_len')
            # query: [batch_size, seq_len]
            # doc: [neg+pos, batch_size, seq_len]

            with tf.name_scope('embedding'):
                if word_vector is not None:
                    self.W = tf.Variable(tf.cast(word_vector, dtype=tf.float32, name='word_vector'), name= "embedding_vector")
                    print("use word2vec")
                else:
                    self.W = tf.Variable(tf.random_normal([vocab_len, config.embedding],mean=0., stddev=1.), name='embedding_w')
                    print("use random embedding word vector")

                # query: [batch_size, seq_len, embed_size]
                self.embed_query = tf.nn.embedding_lookup(self.W, self.query)
                # doc: [ [batch_size, seq_len, embed_size] for item in list]
                self.doc_unstack = [doc for doc in tf.unstack(self.doc)]
                self.embed_doc = [tf.nn.embedding_lookup(self.W, doc) for doc in self.doc_unstack]


            with tf.name_scope('LSTM'):
                for idx, hidden_size in enumerate(config.hidden_sizes):
                    if config.Bidirection:
                        with tf.variable_scope('Bi-LSTM' + str(idx)) as scope:
                            #前向
                            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                                                                 state_is_tuple=True),
                                                                         output_keep_prob=config.drop_keep_prob)
                            #后向
                            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                                                                 state_is_tuple=True),
                                                                         output_keep_prob=config.drop_keep_prob)
                            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                            # current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                            outputs_1, current_state_1 = tf.nn.bidirectional_dynamic_rnn(lstm_bw_cell, lstm_fw_cell, self.embed_word_1,
                                                                                       sequence_length = self.sequence_1_len,
                                                                                       dtype = tf.float32, scope = 'bi-lstm' + str(idx))
                            scope.reuse_variables()
                            #前向
                            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                                                                 state_is_tuple=True),
                                                                         output_keep_prob=config.drop_keep_prob)
                            #后向
                            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                                                                 state_is_tuple=True),
                                                                         output_keep_prob=config.drop_keep_prob)
                            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                            # current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                            outputs_2, current_state_2 = tf.nn.bidirectional_dynamic_rnn(lstm_bw_cell, lstm_fw_cell, self.embed_word_2,
                                                                                       sequence_length = self.sequence_1_len,
                                                                                       dtype = tf.float32, scope = 'bi-lstm' + str(idx))



                    else:
                        with tf.variable_scope('LSTM' + str(idx)) as scope:
                            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, name='lstm0'),
                                                                        output_keep_prob=config.drop_keep_prob)
                            outputs_query, _ = tf.nn.dynamic_rnn(lstm_cell, self.embed_query, sequence_length = self.query_len,
                                                                        dtype= tf.float32, scope='lstm' + str(idx))
                            outputs_doc_list = [ tf.nn.dynamic_rnn(lstm_cell, self.embed_doc[i], sequence_length = self.doc_len[i], dtype= tf.float32, scope='lstm'+str(idx))[1]
                              for i in range(config.neg+1) ]

                            # outputs_query: [batch_size, seq_len, lstm_hidden_size]
                            # outputs_doc_list: [ [batch_size, seq_len, lstm_hidden_size] for item in list]
                            self.embed_query = outputs_query
                            self.embed_doc = outputs_doc_list


                row = tf.range(config.batch_size)
                col_1 = self.query_len - 1
                col_doc = [doc_len-1 for doc_len in self.doc_len]

                index_1 = tf.unstack(tf.stack([row, col_1], axis=0), axis=1)
                index_doc = [tf.unstack(tf.stack([row, col], axis=0), axis=1) for col in col_doc]
                #[batch, -1, lstm_hidden_size]  取每个序列最后一个字 lstm输出的h

                self.last_layer_last_step_query = tf.gather_nd(self.embed_query, index_1)
                self.last_layer_last_step_doc_list = tf.gather_nd(self.embed_word_2, index_2)

            with tf.name_scope("cosine_similarity"):

                def cosine(batch_query, batch_doc):
                    query_l2 = tf.sqrt(tf.reduce_sum(batch_query*batch_query, axis=-1))
                    doc_l2 = tf.sqrt(tf.reduce_sum(batch_doc*batch_doc, axis=-1))
                    q_d_mul = tf.reduce_sum(batch_query*batch_doc, axis=-1)
                    score = tf.div(q_d_mul, query_l2*doc_l2 + 1e-10, name='batch_cosine_similarity')
                    return score


                self.query_and_pos_sim = tf.exp(cosine(self.last_layer_last_step_1, self.last_layer_last_step_2))
                self.query_and_neg_sim = tf.exp(cosine(self.last_layer_last_step_1, self.last_layer_last_step_3))

                self.pos_and_neg_score_add = self.query_and_pos_sim + self.query_and_neg_sim
                self.query_and_pos_prob = self.query_and_pos_sim/self.pos_and_neg_score_add
                self.query_and_neg_prob = self.query_and_neg_sim/self.pos_and_neg_score_add


            with tf.name_scope("loss"):
                self.loss = -tf.reduce_mean(tf.log(self.query_and_pos_prob))


            with tf.name_scope("accuracy"):
                self.accuracy = tf.less_equal(self.query_and_neg_prob,self.query_and_pos_prob)
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, dtype=tf.float32))


                        # #共享参数
                        # scope.reuse_variables()
                        #
                        # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        #                                           output_keep_prob=config.drop_keep_prob)
                        # outputs_2, current_state_2 = tf.nn.dynamic_rnn(lstm_cell, self.embed_word_1, sequence_length = self.sequence_1_len,
                        #                                              dtype=tf.float32, scope='lstm' + str(idx))

        if not train :
            self.input_query = tf.placeholder(tf.int32, [None, None], name='input_query')
            self.input_doc = tf.placeholder(tf.int32, [None, None], name="input_doc")
            self.query_len = tf.placeholder(tf.int32, [None], name="sequence_query_len")
            self.doc_len = tf.placeholder(tf.int32, [None], name="sequence_doc_len")
            self.input_y = tf.placeholder(tf.float32, [None], name='input_y')

            with tf.name_scope('embedding'):
                if word_vector is not None:
                    self.W = tf.Variable(tf.cast(word_vector, dtype=tf.float32, name='word_vector'), name= "embedding_vector")
                    print("use word2vec")
                else:
                    self.W = tf.Variable(tf.random_normal([vocab_len, config.embedding],mean=0., stddev=1.), name='embedding_w')
                    print("use random embedding word vector")
                #[batch_size, seq_len]
                self.embed_query = tf.nn.embedding_lookup(self.W, self.input_query)
                self.embed_doc = tf.nn.embedding_lookup(self.W, self.input_doc)

                #[batch_size, seq_len, embedding_size]
                self.embed_query = tf.nn.dropout(self.embed_query, keep_prob=config.drop_keep_prob)
                self.embed_doc = tf.nn.dropout(self.embed_doc, keep_prob=config.drop_keep_prob)



            with tf.name_scope('LSTM'):
                for idx, hidden_size in enumerate(config.hidden_sizes):
                    if config.Bidirection:
                        with tf.variable_scope('Bi-LSTM' + str(idx)) as scope:
                            #前向
                            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                                                                 state_is_tuple=True),
                                                                         output_keep_prob=config.drop_keep_prob)
                            #后向
                            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                                                                 state_is_tuple=True),
                                                                         output_keep_prob=config.drop_keep_prob)
                            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                            # current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                            outputs_1, current_state_1 = tf.nn.bidirectional_dynamic_rnn(lstm_bw_cell, lstm_fw_cell, self.embed_word_1,
                                                                                       sequence_length = self.sequence_1_len,
                                                                                       dtype = tf.float32, scope = 'bi-lstm' + str(idx))
                            scope.reuse_variables()
                            #前向
                            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                                                                 state_is_tuple=True),
                                                                         output_keep_prob=config.drop_keep_prob)
                            #后向
                            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                                                                 state_is_tuple=True),
                                                                         output_keep_prob=config.drop_keep_prob)
                            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                            # current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                            outputs_2, current_state_2 = tf.nn.bidirectional_dynamic_rnn(lstm_bw_cell, lstm_fw_cell, self.embed_word_2,
                                                                                       sequence_length = self.sequence_1_len,
                                                                                       dtype = tf.float32, scope = 'bi-lstm' + str(idx))



                    else:
                        with tf.variable_scope('LSTM' + str(idx)) as scope:
                            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size),
                                                                        output_keep_prob=config.drop_keep_prob)
                            outputs_1, last_word_1 = tf.nn.dynamic_rnn(lstm_cell, self.embed_query, sequence_length = self.query_len,
                                                                        dtype= tf.float32, scope='lstm' + str(idx))
                            outputs_2, last_word_2 = tf.nn.dynamic_rnn(lstm_cell, self.embed_doc, sequence_length = self.doc_len,
                                                                        dtype= tf.float32, scope='lstm' + str(idx))
                            #[batch_size, seq_len, lstm_hidden_size]
                            self.embed_word_1 = outputs_1
                            self.embed_word_2 = outputs_2

                row = tf.range(config.batch_size)
                col_1 = self.query_len - 1
                col_2 = self.doc_len - 1


                index_1 = tf.unstack(tf.stack([row, col_1], axis=0), axis=1)
                index_2 = tf.unstack(tf.stack([row, col_2], axis=0), axis=1)
                #[batch, -1, lstm_hidden_size]  取每个序列最后一个字 lstm输出的h
                self.last_layer_last_step_1 = tf.gather_nd(self.embed_query, index_1)
                self.last_layer_last_step_2 = tf.gather_nd(self.embed_query, index_2)


            with tf.name_scope("cosine_similarity"):

                def cosine(batch_query, batch_doc):
                    query_l2 = tf.sqrt(tf.reduce_sum(batch_query*batch_query, axis=-1))
                    doc_l2 = tf.sqrt(tf.reduce_sum(batch_doc*batch_doc, axis=-1))
                    q_d_mul = tf.reduce_sum(batch_query*batch_doc, axis=-1)
                    score = tf.div(q_d_mul, query_l2*doc_l2 + 1e-10, name='batch_cosine_similarity')
                    return  0.5 + 0.5 * score


                self.query_and_doc_sim = cosine(self.last_layer_last_step_1, self.last_layer_last_step_2)

            with tf.name_scope("loss"):
                self.pred_y = self.query_and_doc_sim
                binary_cross_entry = self.input_y * tf.log1p(self.pred_y) + (1 - self.input_y) * tf.log1p(
                    1 - self.pred_y)
                self.loss = tf.reduce_mean(binary_cross_entry, name="loss")


            with tf.name_scope("accuracy"):
                self.accuracy = tf.less_equal(self.pred_y, 0.5)
                self.accuracy = tf.cast(self.accuracy, dtype=tf.float32)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.input_y, self.accuracy), dtype=tf.float32))






