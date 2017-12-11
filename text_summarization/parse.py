import urllib2
from bs4 import BeautifulSoup
import re


def parse_bbc_article(url):
    try:
        print("URL:{}".format(url))
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        story = soup.find('div', attrs={'class': 'story-body'})

        title = story.find('h1').text
        date = story.find('div', attrs={'class': 'date'}).text

        text = [x.text for x in story.find('div', attrs={'class': 'story-body__inner'}).findAll("p")]

        links = []
        for l in [x['href'] for x in soup.find_all('a', href=True)]:
            m = re.search(r'\d{8}$', l)
            if m and "mailto" not in l and "twitter" not in l and "account" not in l:
                links.append(l)

        return {
            'title': title,
            'author': "BBC",
            'date': date,
            'text': text,
            'links': links,
            'article_url': url
        }
    except Exception:
        return {
            'title': "",
            'author': "",
            'date': "",
            'text': "",
            'links': "",
            'article_url': ""
        }

def parse_abc_article(url):
    links = []
    try:
        print("URL:{}".format(url))
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')

        links = [x['href'] for x in soup.find('section', attrs={'id': 'news-feed'}).find_all('a', href=True) if "photos" not in x['href'] and "video" not in x['href']]
        # for l in [x['href'] for x in soup.find_all('a', href=True)]:
        #     m = re.search(r'\d{8}$', l)
        #     if m and "mailto" not in l and "twitter" not in l and "account" not in l:
        #         links.append(l)

        title = soup.find('header', attrs={'class': 'article-header'}).find('h1').text
        date = soup.find('span', attrs={'class': 'timestamp'}).text
        author = ", ".join([x.text.replace("\n", "").replace("By ", "").title() for x in soup.findAll('div', attrs={'rel': 'author'})])

        text = " ".join([x.text for x in soup.findAll(itemprop="articleBody")])

        
        return {
            'title': title,
            'author': author,
            'date': date,
            'text': text,
            'links': links,
            'article_url' : url
        }
    except Exception:
        return {
            'title': "",
            'author': "",
            'date': "",
            'text': "",
            'links': links,
            'article_url': ""
        }

# parse_bbc_article("http://www.bbc.com/news/world-europe-41798254")
# parse_bbc_article("http://www.bbc.com/news/technology-41942306")
# parse_abc_article("http://abcnews.go.com/Politics/roy-moore-promises-revelations-motivations-sexual-misconduct-allegations/story?id=51083664")