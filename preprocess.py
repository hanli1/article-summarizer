from nltk.corpus import stopwords
from unidecode import unidecode
import re
import string
import nltk
import nltk.data

regex = re.compile('[%s]' % re.escape(string.punctuation)) 

def remove_non_ascii(text):
    return unidecode(unicode(text, encoding="utf-8", errors='ignore'))

def process_sentences(sentences):
    new_sentences = []

    for sentence in sentences:
        new_sentence = []
        for token in nltk.word_tokenize(sentence):
            new_token = regex.sub('', token)
            # remove punctuation and stop words
            if new_token != '' and new_token != stopwords.words('english'):
                new_sentence.append(new_token)

        new_sentences.append(new_sentence)


    return new_sentences

def text_to_sentences(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    return sentences

def sentences_to_tokens(sentences):
    token_sentences = [nltk.word_tokenize(x) for x in sentences]
    return token_sentences