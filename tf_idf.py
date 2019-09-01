import jieba
import numpy as np
from data_helper import data_load_tfidf, stop_word_load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def gen_tfidf(input_file, stop_word_path=None):
    """
    输入 语料库; stop word
    输出 句子1 tfidf; 句子2 tfidf; labels
    :param input_file:
    :param stop_word_path:
    :return:
    """
    sentence_1, sentence_2, labels = data_load_tfidf(input_file)
    all_sentence = sentence_1 + sentence_2
    stop_word_list = stop_word_load(stop_word_path)


    tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba.cut, stop_words=stop_word_list)

    all_sentence_to_tfidf = tfidf_vectorizer.fit_transform(all_sentence)
    all_sentence_to_tfidf = all_sentence_to_tfidf.toarray()
    sentence_1_tfidf, sentence_2_tfidf = all_sentence_to_tfidf[:len(sentence_1)], all_sentence_to_tfidf[len(sentence_1):]

    return sentence_1_tfidf, sentence_2_tfidf, labels

def cosine(vector_1, vector_2):
    vector_1_mat = np.mat(vector_1, dtype='float_')
    vector_2_mat = np.mat(vector_2, dtype='float_')

    inner_product = float(vector_1_mat*vector_2_mat.T)


    l2_norm_product = np.linalg.norm(vector_1_mat) * np.linalg.norm(vector_2_mat)
    if l2_norm_product == 0.0:
        cos = 1.0
    else:
        cos = inner_product/l2_norm_product

    return cos


def comput_acc(input_file, stop_word_path=None):
    """
    设定阈值，以cos相似度分类
    :param input_file:
    :param stop_word_path:
    :return:
    """
    sentence_1_tfidf, sentence_2_tfidf, labels = gen_tfidf(input_file, stop_word_path)
    assert len(sentence_1_tfidf) == len(sentence_2_tfidf)
    pred_labels = []
    for i, _ in enumerate(sentence_1_tfidf):
        cos = cosine(sentence_1_tfidf[i], sentence_2_tfidf[i])
        if cos > 0.12:
            pred_labels.append(1.0)
        else:
            pred_labels.append(0.0)

    acc = accuracy_score(labels, pred_labels)
    prec = precision_score(labels, pred_labels)
    recal = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    print("acc:{}\npre:{}\nrecall:{}\nf1:{}\n".format(acc,prec,recal,f1))

if __name__ == '__main__':
    path_file = "task3_train.txt"
    path_stop_word = "stop_words.txt"
    comput_acc(path_file, path_stop_word)