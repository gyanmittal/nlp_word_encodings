'''
Author: Gyan Mittal
Corresponding Document: https://gyan-mittal.com/nlp-ai-ml/nlp-word-embedding-svd-based-methods/#WindowBasedCoOccurrenceMatrix
Brief about Window-based Co-Occurrence Matrix based SVD:
In this, we count the associated reverse_vocab_word_index mapping between a predefined window.
Idea is that similar documents keep the company of similar reverse_vocab_word_index. After applying this to our corpus, we get the matrix of the size of vocab_word_index * size of the vocab_word_index.
After applying SVD, we can reduce the dimensionality of the matrix into reduced dimensions, which becomes the embdding of every word.
About Code: This code demonstrates the concept of Window-based Co-Occurrence Matrix based SVD with simple example corpus
'''
from collections import Counter
import itertools
from util import naive_clean_text
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def reduce_to_k_dim(M, k=2):
    
    svd = TruncatedSVD(n_components = k, n_iter = 100, random_state = 456, tol = 0.0)
    reduce_matrix_x = svd.fit_transform(M)
    #print(V_center_word_weights)
    return reduce_matrix_x

def naive_window_co_occurrence_matrix(corpus, window_size=4):

    split_docs_words= [naive_clean_text(doc).split() for doc in corpus]
    #print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    #print("word_counts", "\n", word_counts)
    #reverse_vocab_word_index = [x for i, x in enumerate(word_counts)]
    vocab_word_index = {x: i for i, x in enumerate(word_counts)}
    reverse_vocab_word_index = {i: x for i, x in enumerate(word_counts)}
    #print("vocab_word_index\n", vocab_word_index)
    #print("reverse_vocab_word_index\n", reverse_vocab_word_index)

    vocab_size = len(vocab_word_index)
    #print("vocab_size:\t", vocab_size)
    matrix_x = np.zeros((vocab_size, vocab_size))
    for line in split_docs_words:
        for i in range(len(line)):
            #print(i, line[i])
            target = line[i]
            target_index = vocab_word_index[target]
            left = max(i - window_size, 0)
            right = min(i + window_size, len(line) - 1)
            for j in range(left, right + 1):
                window_word = line[j]
                if(i != j and target_index != vocab_word_index[window_word]):
                    matrix_x[target_index][vocab_word_index[window_word]] += 1

    matrix_x_dict = {}
    for row in enumerate(matrix_x):
        row_index, word_doc_vector = row
        matrix_x_dict[reverse_vocab_word_index[row_index]] = word_doc_vector

    print("Window Co Occurrence Matrix:")

    print('{:10s}, {:10s}, {}'.format("word index", "word", "Co Occurrence value"))
    for key, value in matrix_x_dict.items():
        print('{:<10d}, {:10s}, {}'.format(vocab_word_index[key], key, value))
    
    return  matrix_x, vocab_word_index, reverse_vocab_word_index

def plot_embeddings(reduce_matrix_x, vocabulary):
    
    for word, i in vocabulary.items():
        x = reduce_matrix_x[i][0]
        y = reduce_matrix_x[i][1]
        #print(word, ":\t", x, ":\t", y)
        plt.scatter(x, y)
        plt.annotate(word, (x, y))
    plt.show()

corpus = ["I love playing cricket", "you love playing football", "We love playing cricket", "I love all sports", "You love all sports", "We love all sports"]
#corpus = ["I love playing football", "I love playing cricket", "I love playing sports"]
matrix_x, vocabulary, reverse_vocabulary = naive_window_co_occurrence_matrix(corpus, window_size=1)

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
