import random
import jieba
import json

def load_vocab(input_file):
    """
    输入 字典路径
    输出 字典 字典长度
    :param input_file:
    :return:
    """
    with open(input_file, 'r', encoding='utf-8') as reader:
        vocab = json.load(reader)

    return vocab, len(vocab)

def data_load_dssm(input_file):
    query = []
    doc1 = []
    doc2 = []

    with open(input_file,'rt', encoding='utf-8', errors='ignore') as reader:
        for line in reader.readlines():
            line = line.split('\t')
            query.append([x for x in jieba.cut(line[0])])
            doc1.append([x for x in jieba.cut(line[1])])
            doc2.append([x for x in jieba.cut(line[2])])
    labels_1 = [1]*len(doc1)
    labels_2 = [0]*len(doc2)

    return query, doc1, doc2, labels_1, labels_2

def data2idx(sentences, vocab):
    """
    输入 句子
    输出 句子中词再词典里的index
    :param sentences:
    :param vocab:
    :return:
    """
    return [[vocab.get(word, vocab["<UNK>"]) for word in sentence] for sentence in sentences]


def split_train_eval(query, doc_1, doc_2, labels_1, labels_2, train_num_percent = 0.8):
    """
    输入 两句
    输出 训练集 测试集分割
    :param sentences_1:
    :param sentences_2:
    :param labels:
    :param train_num_percent:
    :return:
    """
    train_num = int(len(query)*train_num_percent)

    train_query = query[:train_num]
    train_doc_1 = doc_1[:train_num]
    train_doc_2 = doc_2[:train_num]

    train_labels_1 = labels_1[:train_num]
    train_labels_2 = labels_2[:train_num]

    eval_query = query[train_num:]
    eval_doc_1 = doc_1[train_num:]
    eval_doc_2 = doc_2[train_num:]

    eval_labels_1 = labels_1[train_num:]
    eval_labels_2 = labels_2[train_num:]
    return train_query, train_doc_1, train_doc_2, train_labels_1, train_labels_2, \
           eval_query, eval_doc_1, eval_doc_2, eval_labels_1, eval_labels_2




def gen_data_dssm(input_file, vocab_file):
    vocab, vocab_len = load_vocab(vocab_file)
    query, doc1, doc2, labels_1, labels_2 = data_load_dssm(input_file)
    query = data2idx(query, vocab)
    doc1 = data2idx(doc1, vocab)
    doc2 = data2idx(doc2, vocab)
    query, doc_1, doc_2, train_labels_1, train_labels_2, \
    eval_query, eval_doc1, eval_doc2, eval_labels_1, eval_labels_2 = split_train_eval(query, doc1, doc2, labels_1, labels_2)

    return query, doc_1, doc_2, train_labels_1, train_labels_2, eval_query, eval_doc1, eval_doc2, eval_labels_1, eval_labels_2, vocab_len




def padding(query, doc_1, doc_2, labels_1, labels_2):
    query_len = [len(sentence) for sentence in query]
    doc_1_len = [len(sentence) for sentence in doc_1]
    doc_2_len = [len(sentence) for sentence in doc_2]



    # max_len_1 = max(sentences_1_len)
    # max_len_2 = max(sentences_2_len)
    max_len = max(query_len + doc_2_len + doc_1_len)
    query_pad = [sentence + (max_len - len(sentence))*[5739] for sentence in query]
    doc_1_pad = [sentence + (max_len - len(sentence))*[5739] for sentence in doc_1]
    doc_2_pad = [sentence + (max_len - len(sentence))*[5739] for sentence in doc_2]


    #sentences_len = [len(sentence) for sentence in sentences_1_pad] + [len(sentence) for sentence in sentences_2_pad]

    # if len(set(sentences_len)) > 1:
    #     print("padding fail, length of sentences no consist")

    return dict(query_pad = query_pad, doc_1_pad = doc_1_pad, doc_2_pad = doc_2_pad, labels_1 = labels_1,
                query_len = query_len, doc_1_len = doc_1_len, doc_2_len = doc_2_len, labels_2 = labels_2,
                max_len = max_len)


def next_batch_dssm(train_query, train_doc_1, train_doc_2, labels_1, labels_2, batch_size):
    train_data = zip(train_query, train_doc_1, train_doc_2)
    train_label = zip(labels_1, labels_2)
    mid_val = list(zip(train_data, train_label))
    random.shuffle(mid_val)
    x, y = zip(*mid_val)
    query, doc_1, doc_2 = zip(*x)
    labels_1, labels_2 = zip(*y)


    query, doc_1, doc_2 = list(query), list(doc_1), list(doc_2)
    labels_1, labels_1 = list(labels_1), list(labels_2)

    num_batch = len(query) // batch_size

    for i in range(num_batch):
        start = i*batch_size
        end = start + batch_size
        batch_query = query[start:end]
        batch_doc_1 = doc_1[start:end]
        batch_doc_2 = doc_2[start:end]

        labels_1 = labels_1[start:end]
        labels_2 = labels_2[start:end]

        yield padding(batch_query, batch_doc_1, batch_doc_2, labels_1, labels_2)