from nltk.corpus import wordnet
import nltk
import numpy as np
import preprocess
import time

from sklearn.feature_extraction.text import TfidfVectorizer

def word_similarity(w1, w2):
    # pick for word sense for now
    x = wordnet.synsets(w1)
    y = wordnet.synsets(w2)
    if len(x) == 0 or len(y) == 0:
        return 0

    val = x[0].wup_similarity(y[0])
    if not val:
        # comparing across different POS, don't do anything for now
        return 0

    return val

def word_order_similarity(s1_tokens, s2_tokens):
    joined_set = list(set(s1_tokens).union(set(s2_tokens)))
    def generate_r_vector(tokens):
        r = [0 for x in range(max(len(s1_tokens), len(s2_tokens)))]
        for i, word in enumerate(joined_set):
            if word in tokens:
                # print tokens
                # print joined_set
                r[tokens.index(word)] = i
            else:
                max_sim = -1
                found_word = None
                for token in tokens:
                    sim = word_similarity(word, token)
                    if sim > max_sim:
                        max_sim = sim
                        found_word = token

                r[tokens.index(found_word)] = i
        return r

    r1 = np.array(generate_r_vector(s1_tokens))
    r2 = np.array(generate_r_vector(s2_tokens))

    return 1 - (np.linalg.norm(r1 - r2))/(np.linalg.norm(r1 + r2))

def word_order_summary(sentences, num_sentences=1):
    all_maxes = []
    token_sentences = preprocess.sentences_to_tokens(sentences)

    for i in range(len(token_sentences)):
        total = 0
        for j in range(len(token_sentences)):
            if i != j:
                total += word_order_similarity(token_sentences[i], token_sentences[j])

        all_maxes.append((sentences[i], total))

    all_maxes = sorted(all_maxes, key=lambda tup: tup[1], reverse=True)

    summary = ""
    for i in range(num_sentences):
        if i >= len(all_maxes):
            break
        summary += all_maxes[i][0]

    return summary

    # for i, j in all_maxes:
    #     print i

def cosine_similarity(sentences):
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform(sentences)
    ans = (tfidf* tfidf.T).A
    
    return ans

def cosine_similarity_two_sentences(s1, s2):
    return cosine_similarity([s1, s2])[0][1]

def cosine_summary(sentences, num_sentences=1):
    sentence_likely = np.sum(cosine_similarity(sentences), axis=1)
    index_and_likely = [(sentences[i], x) for i, x in enumerate(sentence_likely)]
    index_and_likely = sorted(index_and_likely, key=lambda tup: tup[1], reverse=True)

    summary = ""
    for i in range(num_sentences):
        if i >= len(index_and_likely):
            break
        summary += index_and_likely[i][0]

    return summary

def edit_distance(w1, w2):
    return nltk.edit_distance(w1, w2)

def semantic_similarity(sentences):
    pass

def main(): 
    # print word_order_similarity(nltk.word_tokenize("A quick brown dog jumps over the lazy fox"), nltk.word_tokenize("A quick brown fox jumps over the lazy dog"))
    # print word_order_similarity(nltk.word_tokenize("I do not get how this works"), nltk.word_tokenize("nobody likes natural language processing"))
    # print word_order_similarity(nltk.word_tokenize("hello world I am"), nltk.word_tokenize("back to haunt everything"))
    # print word_order_similarity(nltk.word_tokenize("cat likes dog"), nltk.word_tokenize("dog likes cat"))
    # print word_order_similarity(nltk.word_tokenize("i like cats"), nltk.word_tokenize("you hate dogs"))
    a = time.time()
    print(word_similarity("cat", "dog"))
    print(time.time() - a)

    a = time.time()
    print(word_similarity("cat", "dog"))
    print(time.time() - a)

    a = time.time()
    print(word_similarity("cat", "dog"))
    print(time.time() - a)


    a = time.time()
    print(word_similarity("animal", "cow"))
    print(time.time() - a)


    a = time.time()
    print(word_similarity("duck", "pig"))
    print(time.time() - a)
    #test_sample()

if __name__ == '__main__':
    main()



