'''
Author: Gyan Mittal
Corresponding Document:
Brief about word2vec:
About Code: This code demonstrates the concept of calculating the word embeddings using word2vec methodology
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

    center_word_train = []
    context_words_train = []

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

            for j in range(i - window_size, i + window_size):
                if i != j and j >= 0 and j < len(sentence):
                    context[vocab_word_index[sentence[j]]] += 1
            center_word_train.append(center_word)
            context_words_train.append(context)

    return center_word_train, context_words_train, vocab_word_index


def naive_softmax_loss_and_gradient(center_word_embed_weight_vec, context_word_idx, U_context_words_weights):

    yhat = naive_softmax(np.dot(U_context_words_weights.T, center_word_embed_weight_vec))
    loss = -np.log(yhat[context_word_idx])
    yhatCopyMinusOneAtIndex = yhat.copy()
    yhatCopyMinusOneAtIndex[context_word_idx] -= 1

    grad_center_vec = np.dot(U_context_words_weights, yhatCopyMinusOneAtIndex)
    grad_outside_vecs = np.dot(yhatCopyMinusOneAtIndex[:, np.newaxis], center_word_embed_weight_vec[np.newaxis, :]).T

    return loss, grad_center_vec, grad_outside_vecs


def train(center_word_train, context_words_train, vocab_word_index, embedding_dim=2, epochs=1000, learning_rate_alpha=1e-03):
    vocab_size = len(vocab_word_index)

    np.random.seed(0)
    V_center_word_weights = np.random.normal(0, .1, (vocab_size, embedding_dim))
    U_context_words_weights = np.random.normal(0, .1, (embedding_dim, vocab_size))

    for epoch_number in range(0, epochs):
        loss = 0
        for i in range(len(center_word_train)):

            grad_center_vectors = np.zeros(V_center_word_weights.shape)
            grad_context_vectors = np.zeros(U_context_words_weights.shape)
            for context_word_idx in range(vocab_size):
                if (context_words_train[i][context_word_idx]):
                    center_word_embed_weight_vec = np.dot(V_center_word_weights.T, center_word_train[i])
                    l, grad_center, grad_outside = naive_softmax_loss_and_gradient(center_word_embed_weight_vec, context_word_idx, U_context_words_weights)
                    loss += l
                    current_center_word_idx = -1
                    for c_i in range(len(center_word_train[i])):
                        if center_word_train[i][c_i] == 1:
                            current_center_word_idx = c_i
                            break
                    grad_center_vectors[current_center_word_idx] += grad_center
                    grad_context_vectors += grad_outside

            U_context_words_weights -= learning_rate_alpha * grad_context_vectors
            V_center_word_weights -= learning_rate_alpha * grad_center_vectors
        loss /= len(center_word_train)
        if(epoch_number%(epochs/10) == 0 or epoch_number == (epochs - 1) or epoch_number == 0):
            print("epoch ", epoch_number, " loss = ", loss, " learning_rate_alpha:\t", learning_rate_alpha)
    return V_center_word_weights, U_context_words_weights

def plot_embeddings(V_center_word_weights, vocab_word_index):
    for word, i in vocab_word_index.items():
        x = V_center_word_weights[i][0]
        y = V_center_word_weights[i][1]
        plt.scatter(x, y)
        plt.annotate(word, (x, y))
    plt.show()

embedding_dim = 2
epochs = 10000
learning_rate_alpha = 1e-03
corpus_sentences = ["I love playing football", "I love playing cricket", "I love playing sports"]

center_word_train, context_words_train, vocab_word_index = prepare_training_data(corpus_sentences)
print(vocab_word_index)
V_center_word_weights, U_context_words_weights = train(center_word_train, context_words_train, vocab_word_index, embedding_dim, epochs, learning_rate_alpha)

print("U_context_words_weights:\t", U_context_words_weights, U_context_words_weights.shape)
print("V_center_word_weights:\t", V_center_word_weights, V_center_word_weights.shape)
plot_embeddings(V_center_word_weights, vocab_word_index)
#plot_embeddings(U_context_words_weights.T, vocab_word_index)