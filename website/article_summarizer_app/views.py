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

# Create your views here.
def index(request):
    output_list = ''
    output=''
    return render_to_response('project_template/index.html')

def search(request):
    search = request.GET.get('query')
    return JsonResponse({"results": search + " djkf"})
