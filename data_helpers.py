import numpy as np
import pandas as pd
import nltk
import re
from sklearn.utils import shuffle
from collections import Counter
import itertools

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_from_disk(lang):
    pd_df = pd.read_csv('./data/en.csv')

    pd_df= pd_df[['community', 'title', 'textBody']]
    pd_df.dropna()
    pd_df['body'] = pd_df['textBody'].fillna(value='')
    pd_df['title'] = pd_df['title'].fillna(value='')

    pd_df['message'] = pd_df.title.str.cat(pd_df.body)
    pd_df['tags'] = pd_df['community'].fillna(value='')

    pd_df.drop(['body', 'title'], axis=1, inplace=True)
    pd_df = shuffle(pd_df)
    pd_df['tags'].dropna()
    pd_df['message'] = pd_df['message'].str[:256]
    #pd_df = pd_df.groupby("tags").filter(lambda x: len(x) >= 1000) # remove all the tags which has less than 5K posts

    # Map the actual labels to one hot labels
    labels = sorted(list(set(pd_df['tags'].tolist())))
    print(labels)
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_raw = pd_df['message'].apply(lambda x: clean_str(x)).tolist()
    y_raw = pd_df['tags'].apply(lambda y: label_dict[y]).tolist()
    return x_raw, y_raw,labels


def pad_sentences(sentences, padding_word="<PAD/>", maxlen=256):
    """
    Pads all the sentences to the same length. The length is defined by the longest sentence.
     Returns padded sentences.
    """
    print('padding sentences ...')
    if maxlen > 0:
        sequence_length = maxlen
    else:
        sequence_length = max(len(s) for s in sentences)

    print('max sentence length is ', sequence_length)
    print('number of sentences ',len(sentences))
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence = (sentence[:sequence_length]) if len(sentence) > sequence_length else sentence

        num_padding = sequence_length - len(sentence)

        replaced_newline_sentence = []
        for char in list(sentence):
            if char == "\n":
                replaced_newline_sentence.append("<NEWLINE/>")
            elif char == " ":
                replaced_newline_sentence.append("<SPACE/>")
            else:
                replaced_newline_sentence.append(char)

        new_sentence = replaced_newline_sentence + [padding_word] * num_padding

        # new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Map from index to word
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Map from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary
    """
    x = np.array([[vocabulary[word] if word in vocabulary else 0 for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def sentence_to_index(sentence, vocabulary, maxlen):
    sentence = clean_str(sentence)
    raw_input = [list(sentence)]
    sentences_padded = pad_sentences(raw_input, maxlen=maxlen)
    raw_x, dummy_y = build_input_data(sentences_padded, [0], vocabulary)
    return raw_x


def load_data():
    x_raw, y_raw,labels = load_data_from_disk('en') #load_data_from_disk()

    sentences_padded = pad_sentences(x_raw)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, y_raw, vocabulary)
    print('data loaded ......')
    return [x, y, vocabulary, vocabulary_inv,labels]


if __name__ == "__main__":
    load_data_from_disk('en')
