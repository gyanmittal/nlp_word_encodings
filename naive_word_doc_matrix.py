from collections import Counter
import itertools
from util import naive_clean_text
import numpy as np
from sklearn.decomposition import TruncatedSVD


def naive_word_doc_matrix(docs):

    split_docs_words= [naive_clean_text(doc).split() for doc in docs]
    print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    print("word_counts", "\n", word_counts)

    vocabulary = {x: i for i, x in enumerate(word_counts)}
    print("vocabulary\n", vocabulary)

    vocab_size = len(vocabulary)
    N = len(docs)
    word_doc_matrix = np.zeros((vocab_size, N))
    #word_doc_matrix = np.zeros((N, vocab_size))

    print(word_doc_matrix)

    for i, doc in enumerate(split_docs_words):
        for word in doc:
            word_doc_matrix[vocabulary[word]][i] += 1
    print("N:\t", N)
    print("word_doc_matrix:\n", word_doc_matrix)
    
    print("Running Truncated SVD over %i words..." % (word_doc_matrix.shape[0]))

    k = 2
    n_iters = 10

    svd = TruncatedSVD(n_components = k, n_iter = n_iters, random_state = 123, tol = 0.0)
    word_doc_matrix_reduced = svd.fit_transform(word_doc_matrix)
    print(word_doc_matrix_reduced.shape)
    print(vocabulary)
    print(word_doc_matrix_reduced)
    
    '''
    vocabulary_hot_vector = {word: [1 if i == vocabulary[word] else 0 for i in range(len(vocabulary))] for i, word in enumerate(vocabulary)}
    print("vocabulary_hot_vector\n", vocabulary_hot_vector)
    
    doc_sequence =[[vocabulary[word] for word in sen] for sen in split_docs_words]
    print("doc_sequence:\n", doc_sequence)
    
    one_hot_vector =[[[1 if i == vocabulary[word] else 0 for i in range(len(vocabulary))] for word in sen] for sen in split_docs_words]
    print("one_hot_vector:\n", one_hot_vector)
    '''

#Example 1
docs = ["I love playing football.", "Indians play cricket", "I love playing cricket"]
#Example 2
#docs =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game."]
naive_word_doc_matrix(docs)

