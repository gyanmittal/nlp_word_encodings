from collections import Counter
import itertools
import numpy as np
import math
from util import naive_clean_text

def naive_tf_idf(docs):
    print(docs)
    split_docs_words= [naive_clean_text(doc).split() for doc in docs]
    word_counts = Counter(itertools.chain(*split_docs_words))
    #print("word_counts", "\n", word_counts)

    vocabulary = {x: i for i, x in enumerate(word_counts)}
    reverse_vocabulary = {i: x for i, x in enumerate(word_counts)}
    print("vocabulary\n", vocabulary)
    print("reverse_vocabulary\n", reverse_vocabulary)
    
    vocab_size = len(vocabulary)
    N = len(docs)
    tf = np.zeros((N, vocab_size))
    tf_dict = {}
    
    for i, doc in enumerate(split_docs_words):
        doc_dict = { word : 0 for word in vocabulary }
        for word in doc:
            tf[i][vocabulary[word]] += 1
            doc_dict[word] = doc_dict[word] + 1
        tf_dict[i] = doc_dict

    print("N:\t", N)
    print("tf:\t", tf)
    print("tf_dict:\t", tf_dict)

    df = {}
    for word in vocabulary:
        for i, doc in enumerate(split_docs_words):
            if word in doc:
                df[word] = df[word] + 1 if word in df else 1

    print("df\t", df)
    
    #idf = np.zeros(vocab_size)
    idf = {}
    for i, word in enumerate(vocabulary):
        idf[word] = round(math.log(N/df[word]), 3)

    print("idf:\t", idf)

    tf_idf = np.zeros((N, vocab_size))
    tf_idf_dict = {}
    
    for i, doc in enumerate(split_docs_words):
        doc_dict = { word : 0 for word in vocabulary }
        for word in doc:
            tf_idf[i][vocabulary[word]] = tf[i][vocabulary[word]] * idf[word] # tf * idf 
            doc_dict[word] =  tf[i][vocabulary[word]] * idf[word] # tf * idf
        tf_idf_dict[i] = doc_dict
    print("tf_idf:\t", np.around(tf_idf, decimals = 3))
    print("tf_idf_dict:\t", tf_idf_dict)

docs =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game."]
#docs = ["I love playing football.", "Indians love playing Cricket."]
naive_tf_idf(docs)

