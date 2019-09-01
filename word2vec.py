import gensim
from data_helper import data_load

def train_w2v(input_file, hidden_size=100):
    sentences_1, sentences_2, labels = data_load(input_file)
    all_sentences = sentences_1 + sentences_2
    model = gensim.models.word2vec.Word2Vec(all_sentences, size = hidden_size, window=5,min_count=1)
    model.vocabulary
    model.save("word2vec.model")
    print("train word2vec successfully")


def load_w2v(embedding_path='word2vec.model'):
    model = gensim.models.Word2Vec.load(embedding_path)
    return model

