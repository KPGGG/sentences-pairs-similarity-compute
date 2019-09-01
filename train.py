import tensorflow as tf

from data_helper import gen_data, next_batch
from data_help_dssm import gen_data_dssm, next_batch_dssm
from modell_siamese_lstm import Siamese_LSTM, Siamese_LSTM_Condif
from model_dssm_lstm import DSSM_LSTM, DSSM_LSTM_config







def Siamese_LSTM_train():
    config = Siamese_LSTM_Condif
    train_1, train_2, train_labels, eval_1, eval_2, eval_labels, vocab_len = gen_data(config.file_path,
                                                                                      config.vocab_path)
    with tf.Graph().as_default():
        session_config =tf.ConfigProto(allow_soft_placement= True, log_device_placement = False)
        session_config.gpu_options.allow_growth = True

        sess = tf.Session(config= session_config)


        with sess.as_default():
            siamese_lstm = Siamese_LSTM(config=config, vocab_len=vocab_len, word_vector = None)
            global_step = tf.Variable(0, name='global_step', trainable=True)
            optimizer = tf.train.AdamOptimizer(config.learning_rate, )
            grads_and_vars = optimizer.compute_gradients(siamese_lstm.loss)
            #tf.summary.histogram('grad',grads_and_vars)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            #merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('logs/', sess.graph)

            sess.run(tf.global_variables_initializer())

            def train_step(batch_x1, batch_x2, batch_y, sequence_len_1, sequence_len_2, max_len=None):
                feed_dict = {
                    siamese_lstm.input_x1:batch_x1,
                    siamese_lstm.input_x2:batch_x2,
                    siamese_lstm.input_y:batch_y,
                    siamese_lstm.sequence_1_len:sequence_len_1,
                    siamese_lstm.sequence_2_len:sequence_len_2,
                }
                #print("input_x1", batch_x1)
                #print("input_x2", batch_x2)
                _, step, dis_exp, output_1, output_2, distance, loss, pred_y, input_y = sess.run(
                    [train_op, global_step, siamese_lstm.manhattan_distance_exp, siamese_lstm.last_layer_last_step_1,
                     siamese_lstm.last_layer_last_step_2, siamese_lstm.manhattan_distance, siamese_lstm.loss, siamese_lstm.pred_y,
                     siamese_lstm.input_y],
                    feed_dict=feed_dict
                )
                # print("output_1", output_1)
                # print("output_2", output_2)
                # print("distance", distance)
                # print("dis_exp", dis_exp)

                # print("distance", distance)
                #print("pred_y", pred_y)
                #print("input_y", input_y)
                #x = tf.print([(grad, var) for grad, var in grads_and_vars])
                #print(sess.run(x))
                print("step:{}, loss:{}".format(step, loss))

            for i in range(config.epoch):

                print("start training")
                for batch_train in next_batch(train_1, train_2, train_labels, config.batch_size):
                    train_step(batch_train['input_sent_1'], batch_train['input_sent_2'], batch_train['input_y'],
                               batch_train['sentences_1_len'], batch_train['sentences_2_len'])


                    current_step = tf.train.global_step(sess, global_step)
                    #rs = sess.run(merged)
                    #writer.add_summary(rs, current_step)

                    if  current_step % config.checkpoint_every == 0:
                        path =saver.save(sess, "models/siamese_lstm/siames_model.ckpt-{}".format(current_step))


def DSSM_LSTM_train():
    config = DSSM_LSTM_config
    query, doc_1, doc_2, train_labels_1, train_labels_2, eval_query, eval_doc1, eval_doc2, eval_labels_1, eval_labels_2, \
    vocab_len = gen_data_dssm(config.file_path, config.vocab_path)

    with tf.Graph().as_default():
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True

        sess = tf.Session(config=session_config)

        with sess.as_default():
            dssm_lstm = DSSM_LSTM(config=DSSM_LSTM_config, vocab_len=vocab_len, word_vector=None, train = True)
            global_step = tf.Variable(0, name='global_step', trainable=True)
            #learning_rate = tf.train.exponential_decay(config.learning_rate, global_step=global_step, decay_steps=10, decay_rate=2)

            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grad_and_var = optimizer.compute_gradients(dssm_lstm.loss)
            train_op = optimizer.apply_gradients(grad_and_var, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            print(tf.global_variables())
            sess.run(tf.global_variables_initializer())

            def train_step(batch_q, batch_d1, batch_d2, batch_y1, batch_y2, query_len, doc_len_1, doc_len_2, max_len=None):
                feed_dict = {
                    dssm_lstm.input_x1:batch_q,
                    dssm_lstm.input_x2:batch_d1,
                    dssm_lstm.input_x3:batch_d2,
                    dssm_lstm.input_y1:batch_y1,
                    dssm_lstm.input_y2:batch_y2,
                    dssm_lstm.query_len:query_len,
                    dssm_lstm.doc_1_len:doc_len_1,
                    dssm_lstm.doc_2_len:doc_len_2
                }


                _, step, loss, prob, accuracy = sess.run([train_op, global_step, dssm_lstm.loss, dssm_lstm.query_and_pos_prob, dssm_lstm.accuracy], feed_dict)

                #print("prob:{}".format(prob))
                print("step:{}, loss:{}, accuracy:{}".format(step, loss, accuracy))




            def eval_step(batch_q, batch_d1, batch_d2, batch_y1, batch_y2, query_len, doc_len_1, doc_len_2, max_len=None):
                feed_dict = {
                    dssm_lstm.input_x1: batch_q,
                    dssm_lstm.input_x2: batch_d1,
                    dssm_lstm.input_x3: batch_d2,
                    dssm_lstm.input_y1: batch_y1,
                    dssm_lstm.input_y2: batch_y2,
                    dssm_lstm.query_len: query_len,
                    dssm_lstm.doc_1_len: doc_len_1,
                    dssm_lstm.doc_2_len: doc_len_2
                }

                step, loss, accuracy = sess.run([global_step, dssm_lstm.loss, dssm_lstm.accuracy], feed_dict= feed_dict)
                print(">>>>>>>>>>>>>>> eval <<<<<<<<<<<<<<  step:{}, loss:{}, accuracy:{}".format(step, loss, accuracy))



            for i in range(config.epoch):

                print("start training")
                for batch_train in next_batch_dssm(query, doc_1, doc_2, train_labels_1, train_labels_2, config.batch_size):
                    train_step(batch_train['query_pad'], batch_train['doc_1_pad'], batch_train['doc_2_pad'],
                               batch_train['labels_1'], batch_train['labels_2'],batch_train['query_len'],
                               batch_train['doc_1_len'],batch_train['doc_2_len'])


                    current_step = tf.train.global_step(sess, global_step)
                    #rs = sess.run(merged)
                    #writer.add_summary(rs, current_step)
                    if current_step % config.evaluate_every == 0:
                        for batch_eval in next_batch_dssm(eval_query, eval_doc1, eval_doc2, eval_labels_1, eval_labels_2, config.batch_size):
                            eval_step(batch_eval['query_pad'], batch_eval['doc_1_pad'], batch_eval['doc_2_pad'],
                               batch_eval['labels_1'], batch_eval['labels_2'],batch_eval['query_len'],
                               batch_eval['doc_1_len'],batch_eval['doc_2_len'])


                    if  current_step % config.checkpoint_every == 0:
                        path =saver.save(sess, "models/dssm_lstm/dssm_model.ckpt-{}".format(current_step))
                        print("Saved model checkpoint to {}\n".format(path))


DSSM_LSTM_train()