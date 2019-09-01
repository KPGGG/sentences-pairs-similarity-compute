import tensorflow as tf
import jieba
from  tensorflow.contrib import learn
import json

def read_corpus(input_file):
    """

    :param input_file: str
    :return:  all_sentneces:list[str]  labels:list[int]   max_sentence_length: int
    """
    all_sentences = []
    labels = []

    with open(input_file, 'rt', encoding='utf-8', errors='ignore') as reader:
        for sentence in reader.readlines():
            sentence = sentence.split('	')
            label = sentence[-1]
            all_sentences.append(sentence[0])
            all_sentences.append(sentence[1])
            labels.append(label)

    print("corpus read")
    max_sentence_length = max([len(sentence) for sentence in all_sentences])

    return all_sentences, labels, max_sentence_length

def chinese_tokenizer(docs):
    """
    jieba 分词包装
    :param docs:
    :return:
    """
    for doc in docs:
        yield jieba.cut(doc)


def creat_corpus(input_file, output_file):
    all_sentences, labels, max_sentence_length = read_corpus(input_file)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_sentence_length, min_frequency=1, tokenizer_fn= chinese_tokenizer)
    vocab_processor.fit(all_sentences)

    with open('vocab.json', 'w', encoding='utf_8') as f:
        json.dump(vocab_processor.vocabulary_._mapping, f, ensure_ascii=False)

    print("vocab create")

if __name__ == "__main__":
    input_path = 'task3_train.txt'
    creat_corpus(input_path,None)