'''
Author: Gyan Mittal
Corresponding Document: https://gyan-mittal.com/nlp-ai-ml/nlp-bag-of-words-bow-model/
Brief about BOW:
In NLP, the technique Bag-of-Words (BOW) model counts the occurrence of each word within a document.
The count can be considered as the weightage of a word in a document.
This algorithm uses Label/ Integer word encoding.
Bag pf words algorithm is useful in Search, Recommendation, Classification, etc. use cases.
About Code: This code demonstrates the Bag-of-Words (BOW) with two simple example corpus
'''

from collections import Counter
import itertools
import numpy as np
from util import naive_clean_text

# naive algorithm impementation of Bag-of-Words (BOW)
def naive_bow(corpus):
    split_docs_words= [naive_clean_text(doc).split() for doc in corpus]
    print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    print("word_counts", "\n", word_counts)

    vocab_word_index = {x: i for i, x in enumerate(word_counts)}
    print("vocabulary\n", vocab_word_index)

    vocab_size = len(vocab_word_index)
    no_docs = len(corpus)
    print(vocab_size, no_docs)
    bow = np.zeros((no_docs, vocab_size))
    for i, doc in enumerate(split_docs_words):
        for word in doc:
            bow[i][vocab_word_index[word]] += 1
    print(bow)

#Sample Corpus
#corpus =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game."]
corpus = ["I love playing football.", "Indians love playing Cricket."]
naive_bow(corpus)


