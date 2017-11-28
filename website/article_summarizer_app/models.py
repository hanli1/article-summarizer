from __future__ import unicode_literals

from django.db import models
import uuid


class NewsArticle(models.Model):
    """
    Model for a home on the site
    """
    news_article_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    date = models.DateTimeField()
    title = models.CharField(max_length=1000)
    publisher = models.CharField(max_length=200)
    author = models.CharField(max_length=200)
    original_article_link = models.URLField(max_length=500)
    picture_link = models.URLField(max_length=500)
    text = models.TextField()
    short_summary = models.TextField()
    medium_summary = models.TextField()
    long_summary = models.TextField()