from collections import Counter
import itertools
from util import naive_clean_text

def naive_label_or_integer_encoding(docs):
    split_docs_words= [naive_clean_text(doc).split() for doc in docs]
    #print("split_docs_words", "\n", split_docs_words)
    word_counts = Counter(itertools.chain(*split_docs_words))
    #print("word_counts", "\n", word_counts)

    label_or_integer_encoding = {x: i for i, x in enumerate(word_counts)}
    print("label_or_integer_encoding\n", label_or_integer_encoding)

#docs =  ["My name is John, What is your name?", "Bill is a very good person. He likes playing soccer.", "What is your favorite game? I love Football. Football is a great game."]
docs = ["I love playing football.", "Indians love playing Cricket."]
naive_label_or_integer_encoding(docs)

