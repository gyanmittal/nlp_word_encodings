'''
Author: Gyan Mittal
Corresponding Document: https://gyan-mittal.com/nlp-ai-ml/nlp-word-embedding-svd-based-methods/#DocTermMatrix
Brief about Document Term Matrix based SVD:
In this, we count the occurrence of each word in each document.
Doing this creates a large matrix of the size of vocab_word_index into the number of documents.
After applying SVD, we can reduce the dimensionality of the matrix into reduced dimensions, which becomes the embdding of every word.
About Code: This code demonstrates the concept of Document Term Matrix based SVD with simple example corpus
'''

from collections import Counter
import itertools
from util import naive_clean_text
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

#Reduce the dimensionality of a matrix
def reduce_to_k_dim(M, k=2):

    svd = TruncatedSVD(n_components = k, n_iter = 100, random_state = 456, tol = 0.0)
    reduce_matrix_x = svd.fit_transform(M)
    #print(W)
    return reduce_matrix_x

def naive_doc_term_matrix(corpus):

    split_docs_words= [naive_clean_text(doc).split() for doc in corpus]
    print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    print("word_counts", "\n", word_counts)

    vocab_word_index = {x: i for i, x in enumerate(word_counts)}
    reverse_vocab_word_index = {i: x for i, x in enumerate(word_counts)}
    print("vocab_word_index\n", vocab_word_index)
    print("reverse_vocabulary\n", reverse_vocab_word_index)

    vocab_size = len(vocab_word_index)
    N = len(corpus)
    matrix_x = np.zeros((vocab_size, N))
    print(matrix_x)

    for i, doc in enumerate(split_docs_words):
        for word in doc:
            matrix_x[vocab_word_index[word]][i] += 1

    print("N:\t", N)
    print("matrix_x:\n", matrix_x)

    matrix_x_dict = {}
    for row in enumerate(matrix_x):
        row_index, word_doc_vector = row
        matrix_x_dict[reverse_vocab_word_index[row_index]] = word_doc_vector

    print("Doc Term Matrix:")

    print('{:10s},{:10s}, {}'.format("word index", "word", "Doc Co Occurrence"))
    for key, value in matrix_x_dict.items():
        print('{:<10d}, {:10s}, {}'.format(vocab_word_index[key], key, value))
    
    return  matrix_x, vocab_word_index, reverse_vocab_word_index

def plot_embeddings(reduce_matrix_x, vocabulary):
    
    for word, i in vocabulary.items():
        x = reduce_matrix_x[i][0]
        y = reduce_matrix_x[i][1]
        #print(word, ":\t", x_train, ":\t", y_train)
        plt.scatter(x, y)
        plt.annotate(word, (x, y))
    plt.show()

#corpus = ["I love playing cricket", "you love playing football", "We love playing cricket", "All love playing football"] #, "You love all sports", "We love all sports"]
corpus = ["I love playing both sports, Cricket and Football", "Indians play both sports, Cricket and Football", "Football is more popular sport than Cricket"]

#matrix_x, vocab_word_index, reverse_vocabulary = naive_window_co_occurrence_matrix(corpus, window_size=1)
matrix_x, vocabulary, reverse_vocabulary = naive_doc_term_matrix(corpus)

#Reduce the matrix to 2 columns
reduce_matrix_x = reduce_to_k_dim(matrix_x, k=2)

reduce_matrix_x_dict = {}
for row in enumerate(reduce_matrix_x):
    row_index, word_doc_vector = row
    reduce_matrix_x_dict[reverse_vocabulary[row_index]] = word_doc_vector

print ("\n\nReduced Matrix [Embeddings Matrix]:")
print('{:10s}, {:10s}, {}'.format("Word Index", "Word", "Embedding"))
for key, value in reduce_matrix_x_dict.items():
    print('{:<10d}, {:10s}, {}'.format(vocabulary[key], key, value))

plot_embeddings(reduce_matrix_x, vocabulary)
