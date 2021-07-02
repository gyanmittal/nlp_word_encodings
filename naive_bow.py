from collections import Counter
import itertools
import numpy as np
from util import naive_clean_text

def naive_bow(docs):
    split_docs_words= [naive_clean_text(doc).split() for doc in docs]
    print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    print("word_counts", "\n", word_counts)

    vocabulary = {x: i for i, x in enumerate(word_counts)}
    print("vocabulary\n", vocabulary)
   
    vocab_size = len(vocabulary)
    no_docs = len(docs)
    print(vocab_size, no_docs)
    bow = np.zeros((no_docs, vocab_size))
    for i, doc in enumerate(split_docs_words):
        for word in doc:
            bow[i][vocabulary[word]] += 1

    print(bow)


#docs =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game."]
docs = ["I love playing football.", "Indians love playing Cricket."]
naive_bow(docs)


