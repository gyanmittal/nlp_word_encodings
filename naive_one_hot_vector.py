from collections import Counter
import itertools
from util import naive_clean_text

def naive_one_hot_vector(docs):
    split_docs_words= [naive_clean_text(doc).split() for doc in docs]
    print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    print("word_counts", "\n", word_counts)

    vocabulary = {x: i for i, x in enumerate(word_counts)}
    print("vocabulary\n", vocabulary)
    
    vocabulary_hot_vector = {word: [1 if i == vocabulary[word] else 0 for i in range(len(vocabulary))] for i, word in enumerate(vocabulary)}
    print("vocabulary_hot_vector\n", vocabulary_hot_vector)
    
    doc_sequence =[[vocabulary[word] for word in sen] for sen in split_docs_words]
    print("doc_sequence:\n", doc_sequence)
    
    one_hot_vector =[[[1 if i == vocabulary[word] else 0 for i in range(len(vocabulary))] for word in sen] for sen in split_docs_words]
    print("one_hot_vector:\n", one_hot_vector)

#Example 1
#docs = ["I love playing football.", "Indians love playing Cricket."]
#Example 2
docs =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game."]
naive_one_hot_vector(docs)

