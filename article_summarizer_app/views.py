from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from django.template import loader
from .form import QueryForm
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import simplejson as json
from django.http import JsonResponse
import os
import numpy as np

from text_summarization.lex_rank import LexRank 

def index(request):
    output_list = ''
    output=''
    return render_to_response('index.html')

def api_articles_list(request):
    pass


def api_article_summary(request):
    pass

def api_summarize(request):
    text = request.GET.get('text')
    length = request.GET.get('length')
    
    lex_rank = LexRank(corpus=text)
    summary = lex_rank.get_summary_sentences(text, 2)
    print summary
    return JsonResponse({"result": summary})