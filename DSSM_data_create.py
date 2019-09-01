import jieba
import random
from tqdm import tqdm

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
            sentences_1.append(sentence[0])
            sentences_2.append(sentence[1])
            labels.append(label)

    print(sentences_1[0])
    print(sentences_2[0])
    print(labels[0])
    return sentences_1, sentences_2, labels


def creat_dssm_data(input_file, dssm_data_output_file, dssm_labels_output_file,neg_nums = 1):
    query = []
    doc = []
    dssm_labels = []
    sentences_1, sentences_2, labels = data_load(input_file)
    sentences_1_pop_i = sentences_1.copy()

    for i, label in enumerate(labels):
        sentences_1_pop_i.pop(i)  ##去除query 以防采样到query
        all_sentences = sentences_1_pop_i + sentences_2
        if int(label) == 1:
            temp_query = sentences_1[i]
            temp_neg = []
            temp_neg.append(sentences_2[i])   ## temp_neg

            neg_samples_equal_query = True

            while neg_samples_equal_query==True:
                neg_samples = random.sample(all_sentences, neg_nums)

                for neg_sample in neg_samples:
                    if neg_sample == temp_query:
                        print("采样到query了。")
                        neg_samples = None

                if neg_samples:
                    break

            # for neg_sample in neg_samples:
            #     if neg_sample == temp_query: ## 目的：保证负采样中没有query ; 操作：判断采样的负例是否与query相等，若相等跳出for循环
            #         print("采样到query了。")
            #         break
            #
            #     else:
            #         temp_neg.append(neg_sample)
            #
            # if len(temp_neg) != neg_nums + 1: ## 目的：保证有 1个正例 和 1个负例 ; 操作：若不满足则跳出 本次循环
            #     continue
            temp_neg.extend(neg_samples)
            query.append(temp_query)
            doc.append(temp_neg)
            dssm_labels.append([1,0])
        sentences_1_pop_i = sentences_1.copy()  ## 复原 方便下次采样
        print(len(sentences_1_pop_i))

    print(len(query))
    print(len(doc))
    print(len(dssm_labels))

    print('query:', query[0])
    print('doc:', doc[0])
    print('dssm_labels', dssm_labels[0])

    with open(dssm_data_output_file, 'w', encoding='utf-8', errors='ignore') as writer:
        for i, _ in enumerate(query):
            writer.write(query[i] +'\t'+ doc[i][0] +'\t'+ doc[i][1] +'\n')

    with open(dssm_labels_output_file, 'w', encoding='utf-8', errors='ignore') as writer:
        for i, _ in enumerate(dssm_labels):
            writer.write(str(dssm_labels[i][0]) + '\t' + str(dssm_labels[i][1]) + '\n')


if __name__ == "__main__":
    input_file = 'task3_train.txt'
    creat_dssm_data(input_file, 'dssm_data.txt', 'dssm_labels.txt')