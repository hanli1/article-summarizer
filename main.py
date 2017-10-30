from nltk.corpus import wordnet
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

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
    print "start"
    joined_set = list(set(s1_tokens).union(set(s2_tokens)))
    print "done creating set"
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
    print "calculated r1"
    r2 = np.array(generate_r_vector(s2_tokens))
    print "calculated r2"

    return 1 - (np.linalg.norm(r1 - r2))/(np.linalg.norm(r1 + r2))

def cosine_similarity(sentences):
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform(sentences)
    ans = (tfidf* tfidf.T).A
    
    return ans

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


def main(): 
    pass

if __name__ == '__main__':
    main()

# with open("sample.txt", "r") as f:
    # print preprocess(["I don't like cats"])

    # print word_order_similarity(nltk.word_tokenize("A quick brown dog jumps over the lazy fox"), nltk.word_tokenize("A quick brown fox jumps over the lazy dog"))
    # print word_order_similarity(nltk.word_tokenize("I do not get how this works"), nltk.word_tokenize("nobody likes natural language processing"))
    # print word_order_similarity(nltk.word_tokenize("hello world I am"), nltk.word_tokenize("back to haunt everything"))
    # print word_order_similarity(nltk.word_tokenize("cat likes dog"), nltk.word_tokenize("dog likes cat"))

    # all_maxes = []
    # for i in range(len(token_sentences)):
    #     total = 0
    #     for j in range(len(token_sentences)):
    #         if i != j:
    #             total += word_order_similarity(token_sentences[i], token_sentences[j])

    #     all_maxes.append((sentences[i], total))

    # all_maxes = sorted(all_maxes, key=lambda tup: tup[1], reverse=True)

    # print all_maxes[0]
    # for i, j in all_maxes:
    #     print i