$( document ).ready(function() {
  var pageCount = 1; //Page count for pagination of articles
  var lastPageFetched = false; //Whether or not the last page of the pagination of articles has been fetched
  var pageFetchInProgress = false; //Whether or not a page of articles is currently being fetched
  var currentDate = ""; //The current date of the articles
  var textQuery = ""; //The text query entered in the search box
  var hideArticleList = false; //Whether or not the article list is hidden
  var hideDynamicInput = true; //Whether or not the dynamic input page is hidden
  var hideAbout = true; //Whether or not the about page is hidden

  $("#summarize-button").click(send_summarize_query);

  function send_summarize_query() {
    data = {
      text: $("#dynamic-text-area").val(),
      summary_length: $('input[name="dynamic-summary-length"]:checked').val(),
      summary_style: $('input[name="dynamic-summary-style"]:checked').val()
    }
    $.get('api/summarize', data, function (response) {
      var summary = response["result"];
      $("#summary-display-area").text(summary);
    });
  }

  $('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
    var target = $(e.target).attr("href"); // activated tab
  });

  $("#search-button").click(send_search_query);
    function send_search_query(){
        textQuery = $("#search-input").val();
        pageCount = 1;
        var container = $("#article-results");
        var loader = $('.spinner').eq(0).clone();
        loader.css("display", "block");
        container.empty();   
        container.append(loader);   
        fetchArticleList(textQuery);
    }
  function fetchArticleList(textQuery) {
    $.ajax({
      url: "api/articles_list",
      data: {
        page_count: pageCount,
        text_query: textQuery
      },
      success: function (responseData) {
        var container = $("#article-results");
        var currentArticlesSelector = $('.article-wrapper');
        var currentDatesSelector = $('.date-wrapper');
        var templateArticleWrapper = currentArticlesSelector.eq(0);
        var previousArticleWrapper = currentArticlesSelector.eq(currentArticlesSelector.length - 1);
        var templateDateWrapper = currentDatesSelector.eq(0);

        var loader = $('.spinner').eq(0).clone();
        loader.css("display", "block");

        $('.spinner').eq($('.spinner').length - 1).css("display", "none");

        if(pageCount == 1)
            container.empty();
 
        for (var i = 0; i < responseData.articles_list.length; i++) {
          var currentArticle = responseData.articles_list[i];
          var currentArticleWrapper = templateArticleWrapper.clone();
          currentArticleWrapper.data("news_article_id", currentArticle.news_article_id);
          currentArticleWrapper.find(".article-title").first().text(currentArticle.title);
          currentArticleWrapper.find(".article-title").first().attr("href", currentArticle.original_article_link);
          currentArticleWrapper.find(".article-organization").first().text(currentArticle.organization);
          currentArticleWrapper.find(".article-author").first().text(currentArticle.author);
          currentArticleWrapper.find(".article-summary").first().text(currentArticle.summary);
          currentArticleWrapper.css("display", "block");
          currentArticleWrapper.find(".short-summary-radio-button").first().prop("checked", true);
          currentArticleWrapper.find(".top-sentences-radio-button").first().prop("checked", true);
          if (currentArticle.date != currentDate) {
            var currentDateWrapper = templateDateWrapper.clone();
            currentDateWrapper.text(currentArticle.date);
            container.append(currentDateWrapper);
            container.append(currentArticleWrapper);
            currentDate = currentArticle.date;
          } else {
            container.append(currentArticleWrapper);
          }
          previousArticleWrapper = currentArticleWrapper;
        }
        if(responseData.articles_list.length != 0)
            container.append(loader);
        pageCount = pageCount + 1;
        if (responseData.last_page_fetched == "true") {
          lastPageFetched = true;
        }
        pageFetchInProgress = false;

      }
    });
  }

  $(window).scroll(function(){
    if (!hideArticleList) {
      var windowScroll = $(this).scrollTop();
      var articleWrappers = $('.article-wrapper');
      var homeWrappersHeight = (articleWrappers.length - 1) * articleWrappers.first().height();
      if (((homeWrappersHeight - windowScroll) <= 400) && !lastPageFetched && !pageFetchInProgress) {
        pageFetchInProgress = true;
        fetchArticleList(textQuery);
      }
    }
  });

  function updateArticleSummary(articleWrapper, summaryLength, summaryStyle) {
    $.ajax({
      url: "api/article_summary",
      data: {
        news_article_id: articleWrapper.data("news_article_id"),
        summary_length: summaryLength,
        summary_style: summaryStyle
      },
      success: function (responseData) {
        articleWrapper.find(".article-summary").first().text(responseData.summary);
      }
    })
  }

  $('#article-results').on('change', 'input[name="summary-length-radio-button"]', function() {
    var articleWrapper = $(this).parents(".article-wrapper").first();
    var summaryLength = $(this).val();
    var summaryStyle =$(this).parents(".article-wrapper").find('input[name="summary-style-radio-button"]:checked').val();
    console.log(summaryStyle);
    updateArticleSummary(articleWrapper, summaryLength, summaryStyle);
  });

  $('#article-results').on('change', 'input[name="summary-style-radio-button"]', function() {
    var articleWrapper = $(this).parents(".article-wrapper").first();
    var summaryLength = $(this).parents(".article-wrapper").find('input[name="summary-length-radio-button"]:checked').val();
    var summaryStyle = $(this).val();
    console.log(summaryLength);
    updateArticleSummary(articleWrapper, summaryLength, summaryStyle);
  });

  fetchArticleList(textQuery);
});