from nltk.corpus import brown
import pprint
from collections import Counter
import itertools
from util import naive_clean_text
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

START_TOKEN = '<START>'
END_TOKEN = '<END>'

def read_corpus(category="news"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = brown.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(brown.words(f))] + [END_TOKEN] for f in files]

def reduce_to_k_dim(M, k=2):
    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    #svd = TruncatedSVD(n_components = k, n_iter = n_iters, random_state = 456, tol = 0.0)
    svd = TruncatedSVD(n_components = k, n_iter = n_iters, tol = 0.0)
    M_reduced = svd.fit_transform(M)
    print(M_reduced.shape)

    print("Done.")
    return M_reduced

def naive_window_co_occurrence_matrix(split_docs_words, window_size=4):

    #split_docs_words= [naive_clean_text(doc).split() for doc in docs]
    #print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    print("word_counts", "\n", word_counts)
    words = [x for i, x in enumerate(word_counts)]
    vocabulary = {x: i for i, x in enumerate(word_counts)}
    reverse_vocabulary = {i: x for i, x in enumerate(word_counts)}
    #print("vocabulary\n", vocabulary)
    #print("reverse_vocabulary\n", reverse_vocabulary)

    vocab_size = len(vocabulary)
    print("vocab_size:\t", vocab_size)
    matrix_x = np.zeros((vocab_size, vocab_size))
    for line in split_docs_words:
        for i in range(len(line)):
            #print(i, line[i])
            target = line[i]
            target_index = vocabulary[target]
            left = max(i - window_size, 0)
            right = min(i + window_size, len(line) - 1)
            for j in range(left, right + 1):
                window_word = line[j]
                if(i != j and target_index != vocabulary[window_word]):
                    matrix_x[target_index][vocabulary[window_word]] += 1

    matrix_x_dict = {}
    for row in enumerate(matrix_x):
        row_index, word_doc_vector = row
        matrix_x_dict[reverse_vocabulary[row_index]] = word_doc_vector

    #print("word_doc_matrix_dict:")

    #print('{:10s} {:10s} {}'.format("word index", "word", "value"))
    #for key, value in matrix_x_dict.items():
        #print('{:<10d} {:10s} {}'.format(vocabulary[key], key, value))

    return words, matrix_x, vocabulary

def plot_embeddings(M_reduced, word2Ind, words):
    '''
    plt.figure(figsize=(10, 10))
    for word in list(unique_word_dict.keys()):
    coord = embedding_dict.get(word)
    plt.scatter(coord[0], coord[1])
    plt.annotate(word, (coord[0], coord[1]))
    '''
    #plt.figure(figsize=(10, 10))
    words_index = [word2Ind[word] for word in words]
    x_coords = [M_reduced[word_index][0] for word_index in words_index]
    y_coords = [M_reduced[word_index][1] for word_index in words_index]
    for i, word in enumerate(words):
        x = x_coords[i]
        y = y_coords[i]
        print(word, "\t",  x, "\t", y)
        plt.scatter(x, y)
        plt.annotate(word, (x, y))
    plt.show()


    '''
    words_index = [word2Ind[word] for word in words]
    print(words_index)
    x_coords = [M_reduced[word_index][0] for word_index in words_index]
    y_coords = [M_reduced[word_index][1] for word_index in words_index]

    for i, word in enumerate(words):
        x = x_coords[i]
        y = y_coords[i]
        print(word, "\t",  x, "\t", y)
        plt.scatter(x, y, marker = 'x', color = 'red')
        plt.text(x + 0.0003, y + 0.0003, word, fontsize = 9)
    plt.show()
    '''


'''
reuters_corpus = read_corpus()
pprint.pprint(reuters_corpus[:3], compact=True, width=100)
words, matrix_x, vocabulary = naive_window_co_occurrence_matrix(reuters_corpus, window_size=4)
pprint.pprint(matrix_x[:3], compact=True, width=100)
'''
#docs = ["Cricket is a very popular game in India", "I like playing cricket among other games", "In USA very few people play cricket", "There are eleven players in a cricket team", "Cricket is played in various formats", "cricket is mostly played by colonial countries"]
docs = ["I love playing cricket", "you love playing football", "Amit love playing cricket", "you love all sports", "Gyan love all sports"]# "Cricket and Football are great sports", "Football is most playing sport", "cricket is not most playing sport", "Not every sport is as popular as football"]

#docs = ["I love playing football.", "you like playing cricket", "many show interest in  playing cricket"]
#docs = [ "I ate dinner.", "We had a three-course meal.", "Brad came to dinner with us.", "He loves fish tacos.", "In the end, we all felt like we ate too much.", "We all agreed; it was a magnificent evening." ]
#docs = ["The future king is the prince", "Daughter is the princess", "Son is the prince", "Only a man can be a king", "Only a woman can be a queen", "The princess will be a queen", "Queen and king rule the realm", "The prince is a strong man", "The princess is a beautiful woman", "The royal family is the king and queen and their children", "Prince is only a boy now", "A boy will be a man"]


split_docs_words = [naive_clean_text(doc).split() for doc in docs]
words, matrix_x, vocabulary = naive_window_co_occurrence_matrix(split_docs_words, window_size=1)

reduce_matrix_x = reduce_to_k_dim(matrix_x)
pprint.pprint(reduce_matrix_x[:3], compact=True, width=100)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(reduce_matrix_x, axis=1)
#M_normalized = reduce_matrix_x / M_lengths[:, np.newaxis] # broadcasting
M_normalized = reduce_matrix_x

print("vocabulary\n", vocabulary)
#words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
#words = ["county", "election", "india", "washington",  "president", "kennedy", "elected", "friendly", "friend", "president-elect", "assemblies", "voters", "governments", "administration", "money", "taxpayers", "citizens", "wife", "divorce", "sports", "football", "sport", "sportsman", "december", "november"]
#words = ["sports", "football", "sport", "sportsman", "december", "november", "king", "queen", "man", "woman", "men", "women", "boy", "girl", "county", "election"]
#words = ["i", "love", "playing", "football", "indians", "cricket"]
#words = ["love", "playing", "football", "cricket"]
words = list(vocabulary.keys())
plot_embeddings(M_normalized, vocabulary, words)
