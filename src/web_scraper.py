import urllib2
from bs4 import BeautifulSoup
from urlparse import urlparse
from parse import parse_bbc_article, parse_abc_article
import json

class WebScraper:

    def __init__(self):
        self.frontier_urls = []
        self.explored_url_set = {}
        self.all_scraped_articles = []

    def process_article(self, parsed_article, article_hostname):
        if (parsed_article.get("title") != "") and (parsed_article.get("date") != "") \
        (parsed_article.get("text") != ""):
            self.all_scraped_articles.append({
                "title": parsed_article.get("title"),
                "date": parsed_article.get("date"),
                "text": parsed_article.get("text"),
                "author": parsed_article.get("author")
            })
        for article_url in parsed_article["links"]:
            current_article = urlparse(article_url):
            if (current_article.netloc == ("www." + article_hostname)) or \
            (current_article.netloc == article_hostname) or (current_article.netloc = ""):
               current_article_host = "www." + article_hostname
                final_article_url = current_article_host + current_article.path + current_article.query
                if final_article_url not in self.explored_url_set:
                    self.frontier_urls.append(final_article_url)

    def scrape_articles(seed_urls):
        self.frontier_urls = seed_urls
        while len(self.frontier_urls) > 0 and (len(self.all_scraped_articles) <= 100):
            current_article = self.frontier_urls.pop(0)
            self.explored_url_set.add(current_article)
            parse_uri = urlparse(current_article)
            if parse_uri.netloc == "www.bbc.com" or parse_uri.netloc == "bbc.com":
                parsed_article = parse_bbc_article(parse_uri)
                self.process_article(parsed_article, "bbc.com")
            elif parse_uri.netloc == "www.abcnews.go.com" or parse_uri.netloc == "abcnews.go.com":
                parsed_article = parse_abc_article(parse_uri)
                self.process_article(parsed_article, "abcnews.go.com")
        with open("data/articles_{}".format(len(self.all_scraped_articles)), "w") as f:
            json.dump(self.all_scraped_articles, f)
