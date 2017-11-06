import csv
import utils
import main
import preprocess
from collections import Counter
import nltk

def calculate_accuracy(summary_function, num_sentences, samples=float('inf')):
    text_and_summary = utils.get_kaggle_data()

    total_accuracy = 0
    for i, (text, actual) in enumerate(text_and_summary):
        if i >= samples:
            break

        predicted = summary_function(preprocess.text_to_sentences(text), num_sentences)

        accuracy = main.cosine_similarity_two_sentences(predicted, actual)
        total_accuracy += accuracy
        # print "evaluated sample: " + str(i)

    print total_accuracy/float(len(text_and_summary))

def test_sample():
    with open("data/sample.txt", "r") as f:
        text = f.read()
        text = preprocess.remove_non_ascii(text)
        sentences = preprocess.text_to_sentences(text)
        summary = main.word_order_summary(sentences, num_sentences=2)        
        print summary

def rouge1_precision_and_recall(predicted_summary, actual_summary):
    prediction_tokens = nltk.word_tokenize(predicted_summary)
    actual_tokens = nltk.word_tokenize(actual_summary)
    prediction_counter = Counter(prediction_tokens)
    actual_counter = Counter(actual_tokens)
    total_overlap = 0
    for key in prediction_counter:
        if key in actual_counter:
            total_overlap += min(prediction_counter[key], actual_counter[key])
    precision = float(total_overlap) / len(prediction_tokens)
    recall = float(total_overlap) / len(actual_tokens)
    return (precision, recall)

if __name__ == '__main__':
    calculate_accuracy(main.word_order_summary, 2, samples=1)
