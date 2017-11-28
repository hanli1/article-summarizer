import os
import sys
import json
from text_summarization.lex_rank import LexRank


def populate_database(data_files):
    all_articles = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            current_articles = json.load(f)
            all_articles += current_articles
    lex_rank = LexRank(list(map(lambda x: x["text"], all_articles)))
    for article in all_articles:
        summary = lex_rank.get_summary_sentences(article["text"], 5)
        NewsArticle.objects.get_or_create(
            date=article["date"],
            title=article["title"],
            organization=article["organization"],
            author=article["author"],
            original_article_link=article["article_url"],
            text=article["text"],
            short_summary=summary[:1],
            medium_summary=summary[:3],
            long_summary=summary[:5]
        )


# Start execution here!
if __name__ == '__main__':
  reload(sys)
  sys.setdefaultencoding("utf-8")
  os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'website.localsettings')

  import django
  django.setup()
  from article_summarizer_app.models import NewsArticle
  populate_database(["../data/bbc_1001.json"])
