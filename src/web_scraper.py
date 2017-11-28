import urllib2
from bs4 import BeautifulSoup
from urlparse import urlparse
from parse import parse_bbc_article, parse_abc_article
import json

class WebScraper:

    def __init__(self):
        self.frontier_urls = []
        self.seen_url_set = set()
        self.all_scraped_articles = []

    def process_article(self, parsed_article, article_hostname):
        if (parsed_article.get("title") != "") and (parsed_article.get("date") != "") and \
        (parsed_article.get("text") != ""):
            self.all_scraped_articles.append({
                "title": parsed_article.get("title"),
                "date": parsed_article.get("date"),
                "text": parsed_article.get("text"),
                "author": parsed_article.get("author"),
                "url": parsed_article.get("url")
            })
            print("GOT ARTICLE {}".format(len(self.all_scraped_articles)))
        for article_url in parsed_article["links"]:
            current_article = urlparse(article_url)
            #print("ARTICLE URL:{}".format(article_url))
            if (current_article.netloc == ("www." + article_hostname)) or \
            (current_article.netloc == article_hostname) or (current_article.netloc == ""):
                current_article_host = "http://www." + article_hostname
                final_article_url = current_article_host + current_article.path + "?" + \
                current_article.query
                #print("FINAL ARTICLE URL:{}".format(final_article_url))
                if final_article_url not in self.seen_url_set:
                    self.seen_url_set.add(final_article_url)
                    self.frontier_urls.append(final_article_url)

    def scrape_articles(self, seed_urls):
        self.frontier_urls = seed_urls
        a = 0
        while len(self.frontier_urls) > 0 and (len(self.all_scraped_articles) <= 2000):
            current_article = self.frontier_urls.pop(0)
            parse_uri = urlparse(current_article)
            #print("HOSTY: {}".format(parse_uri.netloc))
            if parse_uri.netloc == "www.bbc.com" or parse_uri.netloc == "bbc.com":
                #print("HELLO")
                parsed_article = parse_bbc_article(current_article)
                self.process_article(parsed_article, "bbc.com")
            elif parse_uri.netloc == "www.abcnews.go.com" or parse_uri.netloc == "abcnews.go.com":
                #print("HELLO")
                parsed_article = parse_abc_article(current_article)
                self.process_article(parsed_article, "abcnews.go.com")
            #print("FRONTIER:{}".format(self.frontier_urls))
            a += 1
        #print("ITERATIONS: {}".format(a))
        print("ENDING FRONTIER SIZE: {}".format(len(self.frontier_urls)))
        with open("data/articles_{}".format(len(self.all_scraped_articles)), "w") as f:
            json.dump(self.all_scraped_articles, f)

if __name__ == '__main__':
    web_scraper = WebScraper()
    web_scraper.scrape_articles(["http://www.bbc.com/news/world-us-canada-41954436", \
        "http://www.bbc.com/news/technology-41942310", \
        "http://www.bbc.com/sport/rugby-union/41934487", \
        "http://www.bbc.com/news/entertainment-arts-41935971"])
