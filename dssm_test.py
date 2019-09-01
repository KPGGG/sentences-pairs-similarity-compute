import tensorflow as tf
import pprint
from data_helper import gen_data, next_batch
from model_dssm_lstm import DSSM_LSTM, DSSM_LSTM_config


config = DSSM_LSTM_config
train_1, train_2, train_labels, eval_1, eval_2, eval_labels, vocab_len = gen_data(config.file_path, config.vocab_path)
with tf.Graph().as_default():
    session_config =tf.ConfigProto(allow_soft_placement= True, log_device_placement = False)
    session_config.gpu_options.allow_growth = True

    sess = tf.Session(config= session_config)

    with sess.as_default():
        dssm_lstm = DSSM_LSTM(config=config, vocab_len=vocab_len, word_vector = None, train=False)
        global_step = tf.Variable(0, name='global_step', trainable=True)

        variable = [var for var in tf.global_variables() if
                    var.op.name == 'embedding/embedding_w:0' or 'LSTM0/lstm0/lstm_cell/kernel:0'
                    or 'LSTM0/lstm0/lstm_cell/bias:0']

        checkpoint_path = 'models/dssm_lstm/dssm_model.ckpt-12000'
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        pprint.pprint(reader.get_variable_to_shape_map())
        sess.run(tf.assign(variable[0], reader.get_tensor('embedding/embedding_w')))
        sess.run(tf.assign(variable[1], reader.get_tensor('LSTM0/lstm0/lstm_cell/kernel')))
        sess.run(tf.assign(variable[2], reader.get_tensor('LSTM0/lstm0/lstm_cell/bias')))



        def eval_step(batch_x1, batch_x2, batch_y, sequence_len_1, sequence_len_2, max_len=None):
            feed_dict = {
                dssm_lstm.input_query:batch_x1,
                dssm_lstm.input_doc:batch_x2,
                dssm_lstm.input_y:batch_y,
                dssm_lstm.query_len:sequence_len_1,
                dssm_lstm.doc_len:sequence_len_2,
            }
            loss, accuracy = sess.run([dssm_lstm.loss, dssm_lstm.accuracy], feed_dict=feed_dict )



            print("loss:{}, acc:{}".format(loss, accuracy))
            return loss, accuracy

        for i in range(config.epoch):
            total_loss, total_acc = [], []
            for batch_train in next_batch(eval_1, eval_2, eval_labels, config.batch_size):
                loss, acc = eval_step(batch_train['input_sent_1'], batch_train['input_sent_2'], batch_train['input_y'],
                            batch_train['sentences_1_len'], batch_train['sentences_2_len'])
                total_loss.append(loss)
                total_acc.append(acc)


            def mean(item):
                return sum(item) / len(item)

            print("<<<>>>> mean_loss:{}, mean_acc:{}".format(mean(total_loss),mean(total_acc)))

                #current_step = tf.train.global_step(sess, global_step)








