from django.shortcuts import render_to_response
from django.core.paginator import Paginator, EmptyPage
from models import NewsArticle
from django.http import JsonResponse
<<<<<<< HEAD
=======
from django.core.cache import cache
import os
import numpy as np
>>>>>>> 406090732a226c656135e83cc3a182bd77649b38

from text_summarization.lex_rank import LexRank 
from text_summarization import utils 

def index(request):
    output_list = ''
    output=''
    return render_to_response('index.html')

def api_articles_list(request):
    page_count = request.GET.get('page_count')
    text_query = request.GET.get('text_query')
    page_size = 10
    last_page_fetched = "false"
    articles = None
    try:
        if text_query:
            """
            Implement for Text Search
            """
            pass
        else:
            articles = NewsArticle.objects.order_by('-date')
        articles_paginator = Paginator(articles, page_size)
        if page_count:
            page_count = int(page_count)
        else:
            page_count = 1
        articles_page = articles_paginator.page(page_count)
        if page_count == articles_paginator.num_pages:
            last_page_fetched = "true"
    except (ValueError, EmptyPage):
        articles_page = []

    articles_list = []
    for article in articles_page:
        articles_list.append({
            "news_article_id": article.news_article_id,
            "date": "{}/{}/{}".format(article.date.month, article.date.day, article.date.year),
            "title": article.title,
            "organization" : article.organization,
            "author": article.author,
            "original_article_link": article.original_article_link,
            "summary": article.short_summary
        })

    return JsonResponse({"articles_list": articles_list, "last_page_fetched": last_page_fetched})


def api_article_summary(request):
<<<<<<< HEAD
    news_article_id = request.GET.get('news_article_id')
    summary_type = request.GET.get('summary_type')
    try:
        news_article = NewsArticle.objects.get(news_article_id=news_article_id)
        summary = None
        if summary_type == "long":
            summary = news_article.long_summary
        elif summary_type == "medium":
            summary = news_article.medium_summary
        else:
            summary = news_article.short_summary
        return JsonResponse({"summary": summary})
    except (ValueError, NewsArticle.DoesNotExist):
        return JsonResponse({"error_message":"Couldn't fetch article summary"}, status=400)


=======
    pass

def api_summarize(request):
    text = request.GET.get('text')
    length = request.GET.get('length')

    if not cache.get('lex_rank'):
        corpus = list(map(lambda x: x[0], utils.get_kaggle_data()))
        cache.set('lex_rank', LexRank(corpus))
    lex_rank = cache.get('lex_rank')

    summary = lex_rank.get_summary_sentences(text, 2)
    print summary
    return JsonResponse({"result": summary})
>>>>>>> 406090732a226c656135e83cc3a182bd77649b38
