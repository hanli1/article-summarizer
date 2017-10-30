import urllib2
from bs4 import BeautifulSoup

def parse_bbc_article(url):
    page = urllib2.urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')
    story = soup.find('div', attrs={'class': 'story-body'})

    title = story.find('h1').text
    date = story.find('div', attrs={'class': 'date'}).text

    all_paragraphs = story.find('div', attrs={'class': 'story-body__inner'}).findAll("p")
    text = ""
    for p in all_paragraphs:
        text += p.text

    return {
        'title': title,
        'date': date,
        'text': text
    }

parse_bbc_article("http://www.bbc.com/news/world-europe-41798254")