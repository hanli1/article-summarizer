import csv
import preprocess    
import os.path

def get_kaggle_data():
    article_and_summary = []
    path = os.path.join(os.path.dirname(__file__), '../data/kaggle.csv')
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            summary = row[4]
            article = row[5]
            if summary == "text":
                continue
            if article == "":
                # dataset messed up
                continue

            summary = preprocess.remove_non_ascii(summary)
            article = preprocess.remove_non_ascii(article)
            article_and_summary.append((article, summary))
    return article_and_summary
