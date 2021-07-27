from collections import Counter
import itertools
from util import naive_clean_text
import numpy as np
from sklearn.decomposition import TruncatedSVD

def naive_doc_term_matrix(docs):
    
    #split_docs_words= [naive_clean_text('<start/> ' + doc + ' <end/>').split() for doc in corpus]
    split_docs_words= [naive_clean_text(doc).split() for doc in docs]
    print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    print("word_counts", "\n", word_counts)

    vocabulary = {x: i for i, x in enumerate(word_counts)}
    reverse_vocabulary = {i: x for i, x in enumerate(word_counts)}
    print("vocabulary\n", vocabulary)
    print("reverse_vocabulary\n", reverse_vocabulary)

    vocab_size = len(vocabulary)
    N = len(docs)
    doc_term_matrix = np.zeros((vocab_size, N))
    print(doc_term_matrix)

    for i, doc in enumerate(split_docs_words):
        for word in doc:
            doc_term_matrix[vocabulary[word]][i] += 1

    print("N:\t", N)
    print("doc_term_matrix:\n", doc_term_matrix)
    
    doc_term_matrix_dict = {}
    for row in enumerate(doc_term_matrix):
        row_index, doc_term_vector = row
        doc_term_matrix_dict[reverse_vocabulary[row_index]] = doc_term_vector

    print("doc_term_matrix_dict:")

    print('{:10s} {:10s} {}'.format("word index", "word", "value"))
    for key, value in doc_term_matrix_dict.items():
        print('{:<10d} {:10s} {}'.format(vocabulary[key], key, value))
    
    '''
    print("Running Truncated SVD over %i words..." % (doc_term_matrix.shape[0]))

    k = 2
    n_iters = 100

    svd = TruncatedSVD(n_components = k, n_iter = n_iters, random_state = 123, tol = 0.0)
    doc_term_matrix_reduced = svd.fit_transform(doc_term_matrix)
    print(doc_term_matrix_reduced.shape)
    print(vocabulary)
    print(doc_term_matrix_reduced)
    doc_term_matrix_reduced_dict = {}
    
    for row in enumerate(doc_term_matrix_reduced):
        row_index, doc_term_vector = row
        doc_term_matrix_reduced_dict[reverse_vocabulary[row_index]] = doc_term_vector

    print("doc_term_matrix_reduced_dict:\t")

    print('{:10s} {:10s} {}'.format("word index", "word", "value"))
    for key, value in doc_term_matrix_reduced_dict.items():
        print('{:<10d} {:10s} {}'.format(vocabulary[key], key, value))

    '''
    '''
    vocabulary_hot_vector = {word: [1 if i == vocabulary[word] else 0 for i in range(len(vocabulary))] for i, word in enumerate(vocabulary)}
    print("vocabulary_hot_vector\n", vocabulary_hot_vector)
    
    doc_sequence =[[vocabulary[word] for word in sen] for sen in split_docs_words]
    print("doc_sequence:\n", doc_sequence)
    
    one_hot_vector =[[[1 if i == vocabulary[word] else 0 for i in range(len(vocabulary))] for word in sen] for sen in split_docs_words]
    print("one_hot_vector:\n", one_hot_vector)
    '''

#Example 1
#corpus = ["I love playing football.", "Indians love to play cricket", "I love playing cricket"]
#Example 2
#corpus =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game.", "I love playing football.", "Indians love to play cricket", "I love playing cricket"]
docs = ["I love cricket", "I love football", "I love sports", "you love cricket"]
naive_doc_term_matrix(docs)

