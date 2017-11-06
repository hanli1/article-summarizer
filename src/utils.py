import csv
import preprocess    

def get_kaggle_data():
    text_and_summary = []
    with open('data/kaggle.csv', 'r') as f:
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
            text_and_summary.append((text, summary))
    return text_and_summary
