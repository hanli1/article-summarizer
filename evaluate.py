import csv

import main
import preprocess

def calculate_accuracy():
    kaggle = []

    with open('kaggle.csv', 'rb') as f:
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
    for actual, text in kaggle:
        predicted = main.cosine_summary(preprocess.text_to_sentences(text), num_sentences=2)

        pair = [predicted, actual]

        accuracy = main.cosine_similarity(pair)[0][1]
        total_accuracy += accuracy

    print total_accuracy/float(len(kaggle))



calculate_accuracy()