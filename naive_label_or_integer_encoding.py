'''
Author: Gyan Mittal
Corresponding Document: https://gyan-mittal.com/nlp-ai-ml/nlp-label-integer-encoding-of-words/
Brief about Label or Integer Encoding:
In the field of NLP, generally, AI/ ML Algorithms don’t work on text data. We have to find some way to represent text data into numerical data.
Any data consists of reverse_vocab_word_index. So in case, we find some way to convert (Encode) reverse_vocab_word_index into numerical data, then our whole data could be converted into numerical data, which can be consumed by AI/ ML algorithms.
Label/ Integer Encoding of Words is one of the initial methodologies used to encode the reverse_vocab_word_index into numerical data. In this methodology, we assign a numerical value to every word in the corpus starting with zero.
About Code: This code demonstrates the  Label or Integer Encoding with two simple example corpus
'''

from collections import Counter
import itertools
from util import naive_clean_text

def naive_label_or_integer_encoding(corpus):
    split_docs_words= [naive_clean_text(doc).split() for doc in corpus]
    #print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    #print("word_counts", "\n", word_counts)

    label_or_integer_encoding = {x: i for i, x in enumerate(word_counts)}
    print("label_or_integer_encoding\n", label_or_integer_encoding)

#corpus =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game."]
corpus = ["I love playing football.", "Indians love playing Cricket."]
naive_label_or_integer_encoding(corpus)

