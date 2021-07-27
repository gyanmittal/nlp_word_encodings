'''
Author: Gyan Mittal
Corresponding Document: https://gyan-mittal.com/nlp-ai-ml/nlp-word-encoding-by-one-hot-encoding/
Brief about One–Hot–Encoding:
One of the simplest forms of word encoding to represent the word in NLP is One–Hot–Vector–Encoding.
It requires very little computing power to convert text data into one-hot encoding data, and it’s easy to implement.
One–Hot–Encoding has the advantage over Label/ Integer encoding, that the result is binary rather than ordinal, it does not suffer from undesirable bias.
About Code: This code demonstrates the concept of One–Hot–Encoding with two simple example corpus
'''
from collections import Counter
import itertools
from util import naive_clean_text

def naive_one_hot_vector(corpus):
    split_docs_words= [naive_clean_text(doc).split() for doc in corpus]
    print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    print("word_counts", "\n", word_counts)

    vocab_word_index = {x: i for i, x in enumerate(word_counts)}
    print("vocabulary\n", vocab_word_index)

    # =One hot vector of each word in the vocabulary
    vocabulary_hot_vector = {word: [1 if i == vocab_word_index[word] else 0 for i in range(len(vocab_word_index))] for i, word in enumerate(vocab_word_index)}
    print("vocabulary_hot_vector\n", vocabulary_hot_vector)

    # Each doc in corpus can be represented as secquence of word id's instead of words
    doc_sequence_id =[[vocab_word_index[word] for word in sen] for sen in split_docs_words]
    print("doc_sequence_id:\n", doc_sequence_id)

    # Each doc in corpus can be represented as secquence of one hot vectors instead of words and word id's
    one_hot_vector =[[[1 if i == vocab_word_index[word] else 0 for i in range(len(vocab_word_index))] for word in sen] for sen in split_docs_words]
    print("one_hot_vector:\n", one_hot_vector)

#Example 1
corpus = ["I love playing football.", "Indians love playing Cricket."]
#Example 2
#corpus =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game."]
naive_one_hot_vector(corpus)

