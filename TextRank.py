import jieba
import jieba.analyse
import numpy as np

from tf_idf import cosine
from word2vec import train_w2v, load_w2v
from data_helper import data_load_tfidf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def compute_similarity(sentence_1, sentence_2):
    """
    使用textrank提取topk个关键词，两个句子的关键词取交集再 除于 两句关键词个数取对数 之和
    :param sentence_1:
    :param sentence_2:
    :return:
    """
    sent_1_keyword = jieba.analyse.textrank(sentence_1, topK=3)
    sent_2_keyword = jieba.analyse.textrank(sentence_2, topK=3)


    #sent_1_top_word = [item[0] for item in sent_1_keyword]
    #sent_2_top_word = [item[0] for item in sent_1_keyword]

    sent_1_len = len(sent_1_keyword)
    sent_2_len = len(sent_2_keyword)

    co_occurrence = [item for item in sent_1_keyword  if item in sent_2_keyword]

    similarity = len(co_occurrence)/(np.abs(np.log(sent_1_len)) + np.abs(np.log(sent_2_len)))

    return similarity



def compute_similarity_w2v(sentence_1, sentence_2, model, hidden_size=100):
    """
    :param sentence_1:
    :param sentence_2:
    :return:
    """
    sent_1_keyword = jieba.analyse.textrank(sentence_1, topK=3)
    sent_2_keyword = jieba.analyse.textrank(sentence_2, topK=3)


    sent_1_w2v = []
    sent_2_w2v = []

    for keyword in sent_1_keyword:
        if keyword in model:
            sent_1_w2v.append(model[keyword])

    for keyword in sent_2_keyword:
        if keyword in model:
            sent_2_w2v.append(model[keyword])

    if len(sent_1_w2v) == 0:
        sent_1_w2v.append([0]*hidden_size)

    if len(sent_2_w2v) == 0:
        sent_2_w2v.append([0]*hidden_size)

    sent_1_w2v = np.mat(sent_1_w2v)
    sent_2_w2v = np.mat(sent_2_w2v)


    sent_1_w2v_mean = sent_1_w2v.mean(axis = 0)
    sent_2_w2v_mean = sent_2_w2v.mean(axis = 0)

    return cosine(sent_1_w2v_mean, sent_2_w2v_mean)




def comput_acc(input_file, use_w2v=False, train_word2vec=False, word2vec_path='word2vec.model',stop_word_path=None):
    """
    设定阈值，以cos相似度分类
    :param input_file:
    :param stop_word_path:
    :return:
    """
    sentences_1, sentences_2, labels = data_load_tfidf(input_file)
    assert len(sentences_1) == len(sentences_2)
    pred_labels = []

    if use_w2v == False:
        for i, _ in enumerate(sentences_1):
            sim = compute_similarity(sentences_1[i], sentences_2[i])
            if sim > 0.4:
                pred_labels.append(1.0)
            else:
                pred_labels.append(0.0)

        acc = accuracy_score(labels, pred_labels)
        prec = precision_score(labels, pred_labels)
        recal = recall_score(labels, pred_labels)
        f1 = f1_score(labels, pred_labels)
        print("acc:{}\npre:{}\nrecall:{}\nf1:{}\n".format(acc,prec,recal,f1))

    else:
        if train_word2vec:
            train_w2v(input_file)

        model = load_w2v(word2vec_path)
        for i, _ in enumerate(sentences_1):
            cos = compute_similarity_w2v(sentences_1[i], sentences_2[i], model, hidden_size=100)
            if cos> 0.65:
                pred_labels.append(1.0)
            else:
                pred_labels.append(0.0)
        acc = accuracy_score(labels, pred_labels)
        prec = precision_score(labels, pred_labels)
        recal = recall_score(labels, pred_labels)
        f1 = f1_score(labels, pred_labels)
        print("acc:{}\npre:{}\nrecall:{}\nf1:{}\n".format(acc,prec,recal,f1))


if __name__ == "__main__":
    comput_acc('task3_train.txt', use_w2v=True, train_word2vec=False)