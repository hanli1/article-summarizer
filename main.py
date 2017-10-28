import nltk
import nltk.data
from nltk.corpus import wordnet
import numpy as np
from nltk.corpus import stopwords

from unidecode import unidecode
def remove_non_ascii(text):
    return unidecode(unicode(text, encoding = "utf-8"))

def word_similarity(w1, w2):
    # pick for word sense for now
    x = wordnet.synsets(w1)
    y = wordnet.synsets(w2)
    if len(x) == 0 or len(y) == 0:
        return 0

    val = x[0].path_similarity(y[0])
    if not val:
        # comparing across different POS, don't do anything for now
        return 0

    return val

def word_order_similarity(s1_tokens, s2_tokens):
    joined_set = list(set(s1_tokens).union(set(s2_tokens)))

    def generate_r_vector(tokens):
        r = [x for x in range(len(s1_tokens))]
        for i, word in enumerate(joined_set):
            if word in tokens:
                r[tokens.index(word)] = i
            else:
                max_sim = -1
                found_word = None
                for token in tokens:
                    sim = word_similarity(word, token)
                    # print word + " " + token
                    # print sim
                    if sim > max_sim:
                        max_sim = sim
                        found_word = token

                r[tokens.index(found_word)] = i
        return r

    r1 = np.array(generate_r_vector(s1_tokens))
    r2 = np.array(generate_r_vector(s2_tokens))

    return 1 - (np.linalg.norm(r1 - r2))/(np.linalg.norm(r1 + r2))


import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation)) 

def preprocess(sentences):
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

with open("sample.txt", "r") as f:
    text = remove_non_ascii(f.read())
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)

    # print preprocess(["I don't like cats"])

    print word_order_similarity(nltk.word_tokenize("A quick brown dog jumps over the lazy fox"), nltk.word_tokenize("A quick brown fox jumps over the lazy dog"))
    print word_order_similarity(nltk.word_tokenize("I do not get how this works"), nltk.word_tokenize("nobody likes natural language processing"))
    print word_order_similarity(nltk.word_tokenize("cat likes dog"), nltk.word_tokenize("dog likes cat"))