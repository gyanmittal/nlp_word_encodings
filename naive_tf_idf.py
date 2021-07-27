'''
Author: Gyan Mittal
Corresponding Document:
Brief about TF-IDF (Term Frequency- Inverse Document Frequency):
To overcome the limitation of common words in the Bag of Word (BOW) methodology, a technique called TF-IDF (Term Frequency- Inverse Document Frequency) has been developed.
It gives the importance of any term in a document if it is occurring multiple times in that document,
and also penalize the importance of a term in a document in case it is occurring very frequently in various documents in the corpus.
About Code: This code demonstrates the TF-IDF (Term Frequency- Inverse Document Frequency) with two simple example corpus
'''
from collections import Counter
import itertools
import numpy as np
import math
from util import naive_clean_text

def naive_tf_idf(corpus):
    split_docs_words= [naive_clean_text(doc).split() for doc in corpus]
    word_counts = Counter(itertools.chain(*split_docs_words))
    #print("word_counts", "\n", word_counts)

    vocab_word_index = {x: i for i, x in enumerate(word_counts)}
    reverse_vocab_word_index = {i: x for i, x in enumerate(word_counts)}
    print("vocab_word_index\n", vocab_word_index)
    print("reverse_vocab_word_index\n", reverse_vocab_word_index)
    
    vocab_size = len(vocab_word_index)
    N = len(corpus)
    tf = np.zeros((N, vocab_size))
    tf_dict = {}
    
    for i, doc in enumerate(split_docs_words):
        doc_dict = { word : 0 for word in vocab_word_index }
        for word in doc:
            tf[i][vocab_word_index[word]] += 1
            doc_dict[word] = doc_dict[word] + 1
        tf_dict[i] = doc_dict

    print("N:\t", N)
    print("tf:\t", tf)
    print("tf_dict:\t", tf_dict)

    df = {}
    for word in vocab_word_index:
        for i, doc in enumerate(split_docs_words):
            if word in doc:
                df[word] = df[word] + 1 if word in df else 1

    print("df\t", df)
    
    #idf = np.zeros(vocab_size)
    idf = {}
    for i, word in enumerate(vocab_word_index):
        idf[word] = round(math.log(N/df[word]), 3)

    print("idf:\t", idf)

    tf_idf = np.zeros((N, vocab_size))
    tf_idf_dict = {}
    
    for i, doc in enumerate(split_docs_words):
        doc_dict = { word : 0 for word in vocab_word_index }
        for word in doc:
            tf_idf[i][vocab_word_index[word]] = tf[i][vocab_word_index[word]] * idf[word] # tf * idf
            doc_dict[word] =  tf[i][vocab_word_index[word]] * idf[word] # tf * idf
        tf_idf_dict[i] = doc_dict
    print("tf_idf:\t", np.around(tf_idf, decimals = 3))
    print("tf_idf_dict:\t", tf_idf_dict)

#corpus =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game."]
corpus = ["I love playing football.", "Indians love playing Cricket."]
naive_tf_idf(corpus)

