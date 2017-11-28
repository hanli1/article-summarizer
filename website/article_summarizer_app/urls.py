from django.conf.urls import url

from . import views

app_name = 'pt'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^api/search', views.search, name='search')
]