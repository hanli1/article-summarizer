import nltk
import numpy as np
import csv
import utils
import evaluate
from tqdm import tqdm

class SentenceNode:

    def __init__(self, sentence, node_id):
        self.sentence = sentence
        self.node_id = node_id
        self.neighbors = []
        

class LexRank:
    COSINE_SIMILARITY_CUTOFF = 0.8
    PAGE_RANK_DAMPING = 0.85
    PAGE_RANK_EPSILON = 0.00000001

    def __init__(self,
                corpus,
                cosine_similarity_cutoff=0.85, 
                page_rank_damping=0.85, 
                page_rank_epislon=0.00000001,
                sentence_representation="embedding"):
        self.corpus = corpus
        self.cosine_similarity_cutoff = cosine_similarity_cutoff
        self.page_rank_damping = page_rank_damping
        self.page_rank_epislon = page_rank_epislon
        self.sentence_representation = sentence_representation
        self.current_sentence_page_ranks = {}
        self.current_sentence_graph_nodes = []
        self.current_node_id_to_sentence = {}
        self.embeddings = {}
        if self.sentence_representation == "embedding":
            self.initialize_lexicon()
            self.initialize_embeddings()
        elif self.sentence_representation == "tf-idf":
            pass

    def initialize_embeddings(self):
        self.embeddings = {}
        with open("glove/glove.6B.300d.txt", 'r') as f:
            for line in f.readlines():
                line_components = line.split()
                if line_components[0] in self.lexicon:
                    word = line_components[0]
                    word_embedding = line_components[1:301]
                    self.embeddings[word] = np.array([float(i) for i in word_embedding])

    def initialize_lexicon(self):
        self.lexicon = set()
        for document in self.corpus:
            tokens = nltk.word_tokenize(document)
            for token in tokens:
                if not token in self.lexicon:
                    self.lexicon.add(token.lower())

    def get_cosine_similarity(self, vector1, vector2):
        return np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def tokenize_into_sentences(self, document):
        return nltk.sent_tokenize(document)

    def page_rank(self, sentence_graph_nodes):
        n = len(sentence_graph_nodes)
        page_rank_distribution = np.empty(shape=(n, 1))
        page_rank_distribution.fill(1.0 / n)
        timestep_difference = float("inf")

        while timestep_difference > LexRank.PAGE_RANK_EPSILON:
            new_page_rank_distribution = np.empty_like(page_rank_distribution)
            new_page_rank_distribution[:] = page_rank_distribution
            for node in sentence_graph_nodes:
                neighbor_sum = 0.0
                for node2 in node.neighbors:
                    neighbor_sum += (page_rank_distribution[node2.node_id][0] / len(node2.neighbors))
                new_page_rank_distribution[node.node_id][0] = ((1.0 - self.page_rank_damping) / n) + \
                    self.page_rank_damping * neighbor_sum
            timestep_difference = np.linalg.norm(new_page_rank_distribution - page_rank_distribution)
            page_rank_distribution = new_page_rank_distribution

        return page_rank_distribution.flatten()

    def get_embedding(self, token):
        if token not in self.embeddings:
            self.embeddings[token] = np.random.uniform(low=-1.0, high=1.0, size=(300,))
        return self.embeddings[token]

    def get_sentence_representation(self, sentence):
        if self.sentence_representation == "embedding":
            sentence_tokens = nltk.word_tokenize(sentence)
            word_embeddings = [self.get_embedding(token) for token in sentence_tokens]
            return np.mean(word_embeddings, axis=0)
        elif self.sentence_representation == "tf-idf":
            pass

    def compute_sentence_page_rank_ordering(self, document):
        sentences = self.tokenize_into_sentences(document)
        self.current_sentence_page_ranks = {}
        self.current_sentence_graph_nodes = []
        self.current_node_id_to_sentence = {}

        for i in range(len(sentences)):
            self.current_sentence_graph_nodes.append(SentenceNode(sentences[i], i))
            self.current_node_id_to_sentence[i] = sentences[i]
        for i in range(len(self.current_sentence_graph_nodes)):
            current_node_neighbors = []
            for j in range(len(self.current_sentence_graph_nodes)):
                if i != j:
                    sentence1_vector = self.get_sentence_representation(self.current_sentence_graph_nodes[i].sentence)
                    sentence2_vector = self.get_sentence_representation(self.current_sentence_graph_nodes[j].sentence)
                    cosine_similarity = self.get_cosine_similarity(sentence1_vector, sentence2_vector)
                    if cosine_similarity > self.cosine_similarity_cutoff:
                        current_node_neighbors.append(self.current_sentence_graph_nodes[j])
            self.current_sentence_graph_nodes[i].neighbors = current_node_neighbors
        page_rank_distribution = self.page_rank(self.current_sentence_graph_nodes)
        self.current_sentence_page_ranks = {}
        for i in range(len(page_rank_distribution)):
            self.current_sentence_page_ranks[i] = page_rank_distribution[i]

    def get_summary_sentences(self, sentence_count, block=False):
        if block:
            largest_page_rank = 0.0
            largest_block_id = 0
            for i in range(len(self.current_sentence_graph_nodes) - sentence_count + 1):
                current_page_rank = 0.0
                for j in range(i, i + sentence_count):
                    current_page_rank += self.current_sentence_page_ranks[j]
                if current_page_rank > largest_page_rank:
                    largest_page_rank = current_page_rank
                    largest_block_id = i
            block_summary = []
            for k in range(largest_block_id, largest_block_id + sentence_count):
                block_summary.append(self.current_node_id_to_sentence[k])
            return " ".join(block_summary)
        else:
            page_rank_and_sentences = []
            for i in self.current_sentence_page_ranks:
                page_rank_and_sentences.append((self.current_node_id_to_sentence[i],
                                                self.current_sentence_page_ranks[i]))
            sorted_page_rank_and_sentences = sorted(page_rank_and_sentences, key=lambda x: x[1], reverse=True)
            top_sentences = list(map(lambda x: x[0], sorted_page_rank_and_sentences))
            return " ".join(top_sentences[:sentence_count])


def evaluate_lex_rank(text_and_summary):
    corpus = list(map(lambda x: x[0], text_and_summary))
    lex_rank_parameter_sets = []
    cosine_similarity_cutoffs = [0.70]
    for cutoff in cosine_similarity_cutoffs:
        lex_rank_parameter_sets.append({
           "cosine_similarity_cutoff": cutoff 
        })
    for parameter_set in lex_rank_parameter_sets:
        lex_rank = LexRank(corpus=corpus, cosine_similarity_cutoff=parameter_set["cosine_similarity_cutoff"])
        total_precision = 0.0
        total_recall = 0.0
        for (text, summary) in tqdm(text_and_summary):
            lex_rank.compute_sentence_page_rank_ordering(text)
            summary_sentences = lex_rank.get_summary_sentences(2, block=False)
            (precision, recall) = evaluate.rouge1_precision_and_recall(summary_sentences, summary)
            total_precision += precision
            total_recall += recall
        final_precision = total_precision / len(text_and_summary)
        final_recall = total_recall / len(text_and_summary)
        final_f1 = (2 * final_precision * final_recall) / (final_precision + final_recall)
        print("Lex Rank with Cosine Similarity Cutoff {}, Precision: {}, Recall: {}, F1:{}", \
            parameter_set["cosine_similarity_cutoff"], final_precision, final_recall, final_f1)



if __name__ == '__main__':
    evaluate_lex_rank(utils.get_kaggle_data())



