import json
import jieba
import random

def data_load(input_file):
    """
    输入 数据路径，txt 文件每行包含 句子1 句子2 label
    输出 每行第一句分词后list， 第二句分词后list 和 labels list
    :param input_file:
    :return:
    """
    sentences_1 = []
    sentences_2 = []
    labels = []

    with open(input_file,'rt', encoding='utf-8', errors='ignore') as reader:
        for sentence in reader.readlines():
            sentence = sentence.split('	')
            label = sentence[-1]
            sentences_1.append([x for x in jieba.cut(sentence[0])])
            sentences_2.append([x for x in jieba.cut(sentence[1])])
            labels.append(label)

    print(sentences_1[0])
    print(sentences_2[0])
    print(labels[0])
    return sentences_1, sentences_2, labels






def data_load_tfidf(input_file):
    """
    由于sklearn 计算 tfidf时需要内置tokenizer 所以此处不分词，而是把jieba.cut 加入TfidfVectorizer
    :param input_file:
    :return:
    """
    sentences_1 = []
    sentences_2 = []
    labels = []

    with open(input_file,'rt', encoding='utf-8', errors='ignore') as reader:
        for sentence in reader.readlines():
            sentence = sentence.split('	')
            label = sentence[-1]
            sentences_1.append(sentence[0])
            sentences_2.append(sentence[1])
            labels.append(int(label))

    print(sentences_1[0])
    print(sentences_2[0])
    print(labels[0])
    return sentences_1, sentences_2, labels


def stop_word_load(input_file):
    stop_word_list = []
    with open(input_file, 'rt', encoding='utf-8', errors='ignore') as reader:
        for word in reader.readlines():
            stop_word_list.append(word)

    return stop_word_list




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

def split_train_eval(sentences_1, sentences_2, labels, train_num_percent = 0.8):
    """
    输入 两句
    输出 训练集 测试集分割
    :param sentences_1:
    :param sentences_2:
    :param labels:
    :param train_num_percent:
    :return:
    """
    train_num = int(len(sentences_1)*train_num_percent)

    train_1 = sentences_1[:train_num]
    train_2 = sentences_2[:train_num]
    train_labels = labels[:train_num]

    eval_1 = sentences_1[train_num:]
    eval_2 = sentences_2[train_num:]
    eval_lables = labels[train_num:]

    return train_1, train_2, train_labels, eval_1, eval_2, eval_lables

def data2idx(sentences, vocab):
    """
    输入 句子
    输出 句子中词再词典里的index
    :param sentences:
    :param vocab:
    :return:
    """
    return [[vocab.get(word, vocab["<UNK>"]) for word in sentence] for sentence in sentences]


def gen_data(input_file, vocab_file):
    vocab, vocab_len = load_vocab(vocab_file)
    sentences_1, sentences_2, labels = data_load(input_file)
    sentences_1 = data2idx(sentences_1, vocab)
    sentences_2 = data2idx(sentences_2, vocab)
    train_1, train_2, train_labels, eval_1, eval_2, eval_labels = split_train_eval(sentences_1, sentences_2, labels)

    return train_1, train_2, train_labels, eval_1, eval_2, eval_labels, vocab_len

def padding(sentences_1, sentences_2, labels):
    sentences_1_len = [len(sentence) for sentence in sentences_1]
    sentences_2_len = [len(sentence) for sentence in sentences_2]
    # max_len_1 = max(sentences_1_len)
    # max_len_2 = max(sentences_2_len)
    max_len = max(sentences_1_len + sentences_2_len)
    sentences_1_pad = [sentence + (max_len - len(sentence))*[5739] for sentence in sentences_1]
    sentences_2_pad = [sentence + (max_len - len(sentence))*[5739] for sentence in sentences_2]

    #sentences_len = [len(sentence) for sentence in sentences_1_pad] + [len(sentence) for sentence in sentences_2_pad]

    # if len(set(sentences_len)) > 1:
    #     print("padding fail, length of sentences no consist")

    return dict(input_sent_1 = sentences_1_pad, input_sent_2 = sentences_2_pad, input_y = labels,
                sentences_1_len = sentences_1_len, sentences_2_len = sentences_2_len, max_len = max_len)


def next_batch(train_data_1, train_data_2, labels, batch_size):
    train_data = zip(train_data_1, train_data_2)
    mid_val = list(zip(train_data, labels))
    random.shuffle(mid_val)
    x, y = zip(*mid_val)
    x1, x2 = zip(*x)

    x1, x2, y = list(x1), list(x2), list(y)

    num_batch = len(x1) // batch_size

    for i in range(num_batch):
        start = i*batch_size
        end = start + batch_size
        batch_x1 = x1[start:end]
        batch_x2 = x2[start:end]
        batch_y = y[start:end]

        yield padding(batch_x1, batch_x2, batch_y)





