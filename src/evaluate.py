import csv

import main
import preprocess

def calculate_accuracy(summary_function, num_sentences, samples=float('inf')):
    kaggle = []

    with open('data/kaggle.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            summary = row[4]
            text = row[5]
            if summary == "text":
                continue

            if text == "":
                # dataset messed up
                continue

            summary = preprocess.remove_non_ascii(summary)
            text = preprocess.remove_non_ascii(text)
            kaggle.append((summary, text))

    total_accuracy = 0
    for i, (actual, text) in enumerate(kaggle):
        if i >= samples:
            break

        predicted = summary_function(preprocess.text_to_sentences(text), num_sentences)

        accuracy = main.cosine_similarity_two_sentences(predicted, actual)
        total_accuracy += accuracy
        # print "evaluated sample: " + str(i)

    print total_accuracy/float(len(kaggle))

def test_sample():
    with open("data/sample.txt", "r") as f:
        text = f.read()
        text = preprocess.remove_non_ascii(text)
        sentences = preprocess.text_to_sentences(text)
        summary = main.word_order_summary(sentences, num_sentences=2)        
        print summary

# test_sample()

# calculate_accuracy(main.cosine_summary, 2, samples=1000)
calculate_accuracy(main.word_order_summary, 2, samples=1)