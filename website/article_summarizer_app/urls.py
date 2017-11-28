from django.conf.urls import url

from . import views

app_name = 'pt'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^api/articles_list', views.api_articles_list, name='api_articles_list'),
    url(r'^api/article_summary', views.api_article_summary, name='api_article_summary')
]