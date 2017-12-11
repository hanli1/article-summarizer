import os
import sys
import json
from text_summarization.lex_rank import LexRank
from datetime import datetime
import pytz

def populate_database(data_files):
    # delete all rows
    NewsArticle.objects.all().delete()
    url_set = set()

    all_articles = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            current_articles = json.load(f)
            all_articles += current_articles
    lex_rank = LexRank(list(map(lambda x: " ".join(x["text"]), all_articles)))
    datetime_obj = None
    current_index = 1
    for i, article in enumerate(all_articles):
        try:
            if current_index <= 998:
                continue
            if "article_url" not in article:
                continue
            plain_url = article["article_url"].replace("?", "")
            if(plain_url in url_set):
                continue
            url_set.add(plain_url)

            datetime_obj = None
            if article["organization"] == "BBC":
                datetime_obj = datetime.strptime(article["date"], '%d %B %Y')
            # elif article["organization"] == "ABC News":
            else:
                date_components = article["date"].replace(",", "").split()
                datetime_obj = datetime.strptime(" ".join(date_components[:3]), '%b %d %Y')
            datetime_obj = datetime_obj.replace(tzinfo=pytz.UTC)
            summary = lex_rank.get_summary_sentences(" ".join(article["text"]), 5)
            NewsArticle.objects.get_or_create(
                date=datetime_obj,
                title=article["title"],
                organization=article["organization"],
                author=article["author"],
                original_article_link=plain_url,
                text=(" ".join(article["text"])),
                short_summary=(" ".join(summary[:1])),
                medium_summary=(" ".join(summary[:3])),
                long_summary=(" ".join(summary[:5]))
            )
            print("Finished Article {} out of {}".format(i+1, len(all_articles)))
        except Exception as e:
            print "\tEncountered exception for {}".format(current_index)
            continue
    
    print "There are {} articles in the DB".format(NewsArticle.objects.count())

# Start execution here!
if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'website.local_settings')

    import django
    django.setup()
    from article_summarizer_app.models import NewsArticle
    populate_database(["data/bbc_new_1001.json", "data/abc_new_1001.json"])
