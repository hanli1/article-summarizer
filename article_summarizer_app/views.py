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


def index(request):
    output_list = ''
    output=''
    return render_to_response('index.html')


def search(request):
    search = request.GET.get('query')
    return JsonResponse({"results": search + " djkf"})


def api_articles_list(request):
    pass


def api_article_summary(request):
    pass
