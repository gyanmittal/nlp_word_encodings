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
    print("vocabulary\n", vocabulary)
    
    vocab_size = len(vocabulary)
    N = len(docs)
    tf = np.zeros((N, vocab_size))
    
    for i, doc in enumerate(split_docs_words):
        for word in doc:
            tf[i][vocabulary[word]] += 1
    print("N:\t", N)
    print("tf:\t", tf)

    df = {}
    for word in vocabulary:
        for i, doc in enumerate(split_docs_words):
            if word in doc:
                df[word] = df[word] + 1 if word in df else 1

    print("df\t", df)
    
    #idf = np.zeros(vocab_size)
    idf = {}
    for i, word in enumerate(vocabulary):
        idf[word] = math.log(N/df[word])

    print("idf:\t", idf)

    tf_idf = np.zeros((N, vocab_size))


    for i, doc in enumerate(split_docs_words):
        for word in doc:
            tf_idf[i][vocabulary[word]] = tf[i][vocabulary[word]] * idf[word] # tf * idf 

    print("tf_idf:\t", tf_idf)

#docs =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game."]
docs = ["I love playing football.", "Indians love playing Cricket."]
naive_tf_idf(docs)

