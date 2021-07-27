'''
Author: Gyan Mittal
Corresponding Document: https://gyan-mittal.com/nlp-ai-ml/nlp-word-encoding/
About Code: it is utility file for various encoding concepts discussed at above link
'''
import re

# Clean the text after converting it to lower case
def naive_clean_text(text):
    text = text.strip().lower() #Convert to lower case
    text = re.sub(r"[^A-Za-z0-9]", " ", text) #replace all the characters with space except mentioned here
    return text
