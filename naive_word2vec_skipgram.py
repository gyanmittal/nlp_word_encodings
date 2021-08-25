'''
Author: Gyan Mittal
Corresponding Document: https://gyan-mittal.com/nlp-ai-ml/nlp-word2vec-skipgram-neural-network-iteration-based-methods-word-embeddings
Brief about word2vec: A team of Google researchers lead by Tomas Mikolov developed, patented, and published Word2vec in two publications in 2013.
For learning word embeddings from raw text, Word2Vec is a computationally efficient predictive model.
Word2Vec methodology is used to calculate Word Embedding based on Neural Network/ iterative.
Word2Vec methodology have two model architectures: the Continuous Bag-of-Words (CBOW) model and the Skip-Gram model.
About Code: This code demonstrates the basic concept of calculating the word embeddings
using word2vec methodology using Skip-Gram model.
'''

from collections import Counter
import itertools
import numpy as np
import re
import matplotlib.pyplot as plt

# Clean the text after converting it to lower case
def naive_clean_text(text):
    text = text.strip().lower() #Convert to lower case
    text = re.sub(r"[^A-Za-z0-9]", " ", text) #replace all the characters with space except mentioned here
    return text

def naive_softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

def prepare_training_data(corpus_sentences):
    window_size = 1
    split_corpus_sentences_words = [naive_clean_text(sentence).split() for sentence in corpus_sentences]

    X_center_word_train = []
    y_context_words_train = []

    word_counts = Counter(itertools.chain(*split_corpus_sentences_words))
    #print("word_counts", "\n", word_counts)

    vocab_word_index = {x: i for i, x in enumerate(word_counts)}
    reverse_vocab_word_index = {value: key for key, value in vocab_word_index.items()}
    vocab_size = len(vocab_word_index)

    for sentence in split_corpus_sentences_words:
        for i in range(len(sentence)):
            center_word = [0 for x in range(vocab_size)]
            center_word[vocab_word_index[sentence[i]]] = 1
            context = [0 for x in range(vocab_size)]

            for j in range(i - window_size, i + window_size+1):
                #print(i, "\t", j)
                if i != j and j >= 0 and j < len(sentence):
                    context[vocab_word_index[sentence[j]]] = 1
            X_center_word_train.append(center_word)
            y_context_words_train.append(context)

    return X_center_word_train, y_context_words_train, vocab_word_index


def naive_softmax_loss_and_gradient(h, context_word_idx, W1):

    u = np.dot(W1.T, h)
    #print("u:\t", u)
    yhat = naive_softmax(u)
    #print("yhat:\t", yhat)

    loss_context_word = -np.log(yhat[context_word_idx])
    #print("loss_context_word:\t", loss_context_word)
    #prediction error
    e = yhat.copy()
    e[context_word_idx] -= 1
    #print("e:\t", e)

    dW_context_word = np.dot(W1, e)
    dW1_context_word = np.dot(h[:, np.newaxis], e[np.newaxis, :])

    #print("dW_context_word:\t", dW_context_word)
    #print("dW1_context_word:\t", dW1_context_word)

    return loss_context_word, dW_context_word, dW1_context_word


def train(x_train, y_train, vocab_word_index, embedding_dim=2, epochs=1000, learning_rate_alpha=1e-03):
    vocab_size = len(vocab_word_index)

    np.random.seed(0)
    W = np.random.normal(0, .1, (vocab_size, embedding_dim))
    W1 = np.random.normal(0, .1, (embedding_dim, vocab_size))
    #print("W1:\t", W1, W1.shape)
    #print("W:\t", W, W.shape)

    for epoch_number in range(0, epochs):
        loss = 0
        for i in range(len(x_train)):
            #print("x_train[i]\t", x_train[i])
            #print("y_train[i]\t", y_train[i])
            dW = np.zeros(W.shape)
            dW1 = np.zeros(W1.shape)
            h = np.dot(W.T, x_train[i])
            #print("h:\t", h)
            #exit(0)
            for context_word_idx in range(vocab_size):
                if (y_train[i][context_word_idx]):
                    #print("context_word_idx:\t", context_word_idx)
                    loss_context_word, dW_context_word, dW1_context_word = naive_softmax_loss_and_gradient(h, context_word_idx, W1)
                    loss += loss_context_word
                    current_center_word_idx = -1
                    for c_i in range(len(x_train[i])):
                        if x_train[i][c_i] == 1:
                            current_center_word_idx = c_i
                            break
                    dW[current_center_word_idx] += dW_context_word
                    dW1 += dW1_context_word
                    #print("dW:\t", dW)
                    #print("dW1:\t", dW1)
            #exit(0)

            W -= learning_rate_alpha * dW
            W1 -= learning_rate_alpha * dW1
            #print("W:\t", W)
            #print("W1:\t", W1)
            #exit(0)
        loss /= len(x_train)
        if(epoch_number%(epochs/10) == 0 or epoch_number == (epochs - 1) or epoch_number == 0):
            print("epoch ", epoch_number, " loss = ", loss, " learning_rate_alpha:\t", learning_rate_alpha)
    return W, W1

def plot_embeddings(W, vocab_word_index):
    for word, i in vocab_word_index.items():
        x_coord = W[i][0]
        y_coord = W[i][1]
        plt.scatter(x_coord, y_coord)
        plt.annotate(word, (x_coord, y_coord))
        print(word, ":\t[", x_coord, ",", y_coord, "]")
    plt.show()

embedding_dim = 2
epochs = 1000
learning_rate_alpha = 1e-03
corpus_sentences = ["I love playing Football", "I love playing Cricket", "I love playing sports"]
#corpus_sentences = ["I love playing Football"]

x_train, y_train, vocab_word_index = prepare_training_data(corpus_sentences)
#print(vocab_word_index)
#print("x_train:\t", x_train)
#print("y_train:\t", y_train)

W, W1 = train(x_train, y_train, vocab_word_index, embedding_dim, epochs, learning_rate_alpha)

print("W1:\t", W1, W1.shape)
print("W:\t", W, W.shape)
plot_embeddings(W, vocab_word_index)
#plot_embeddings(W1.T, vocab_word_index)