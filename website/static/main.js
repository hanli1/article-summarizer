$( document ).ready(function() {
  var pageCount = 1; //Page count for pagination of articles
  var lastPageFetched = false; //Whether or not the last page of the pagination of articles has been fetched
  var pageFetchInProgress = false; //Whether or not a page of articles is currently being fetched
  var currentDate = ""; //The current date of the articles
  var textQuery = ""; //The text query entered in the search box


  $("#search-button").click(send_search_query);
  
  function send_search_query(){
    $("#search-button").blur();
    // on search button click send query to django backend
    data = { 
        query: "Hello"
    };
    $.get('api/search', data, function(response){
      var results = response["results"];
      console.log(results);
    });
  }

    $('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
      var target = $(e.target).attr("href"); // activated tab
    });

    function fetchArticleList(textQuery) {
      $.ajax({
        url: "api/articles_list",
        data: {
          page_count: pageCount,
          text_query: textQuery
        },
        success: function (responseData) {
          var currentArticlesSelector = $('.article-wrapper');
          var currentDatesSelector = $('.date-wrapper');
          var templateArticleWrapper = currentArticlesSelector.eq(0);
          var previousArticleWrapper = currentArticlesSelector.eq(currentArticlesSelector.length - 1);
          var templateDateWrapper = currentDatesSelector.eq(0);

          for (var i = 0; i < responseData.articles_list.length; i++) {
            var currentArticle = responseData.articles_list[i];
            var currentArticleWrapper = templateArticleWrapper.clone();
            currentArticleWrapper.data("news_article_id", currentArticle.news_article_id);
            currentArticleWrapper.find(".article-title").first().text(currentArticle.title);
            currentArticleWrapper.find(".article-organization").first().text(currentArticle.organization);
            currentArticleWrapper.find(".article-author").first().text(currentArticle.author);
            currentArticleWrapper.find(".article-original-link").first().text(currentArticle.original_article_link);
            currentArticleWrapper.find(".article-original-link").first().attr("href", currentArticle.original_article_link);
            currentArticleWrapper.find(".article-summary").first().text(currentArticle.summary);
            currentArticleWrapper.css("display", "block");
            currentArticleWrapper.find(".short-summary-radio-button").first().prop("checked", true);
            currentArticleWrapper.find(".top-sentences-radio-button").first().prop("checked", true);
            if (currentArticle.date != currentDate) {
              var currentDateWrapper = templateDateWrapper.clone();
              currentDateWrapper.text(currentArticle.date);
              currentDateWrapper.insertAfter(previousArticleWrapper);
              currentArticleWrapper.insertAfter(currentDateWrapper);
              currentDate = currentArticle.date;
            } else {
              currentArticleWrapper.insertAfter(previousArticleWrapper);
            }
            previousArticleWrapper = currentArticleWrapper;
          }

          pageCount = pageCount + 1;
          if (responseData.last_page_fetched == "true") {
            lastPageFetched = true;
          }
          pageFetchInProgress = false;
        }
      });
    }

    $(window).scroll(function(){
        var windowScroll = $(this).scrollTop();
        var articleWrappers = $('.article-wrapper');
        var homeWrappersHeight = (articleWrappers.length - 1) * articleWrappers.first().height();
        if  (((homeWrappersHeight - windowScroll) <= 400) && !lastPageFetched && !pageFetchInProgress) {
           pageFetchInProgress = true;
           fetchArticleList(textQuery);
        }
    });

  $('#article-results').on('change', 'input[name="summary-radio-button"]', function() {
    var articleWrapper = $(this).parents(".article-wrapper").first();
    radioValue = $(this).val();
    $.ajax({
      url: "api/article_summary",
      data: {
        news_article_id: articleWrapper.data("news_article_id"),
        summary_type: radioValue
      },
      success: function (responseData) {
        articleWrapper.find(".article-summary").first().text(responseData.summary);
      }
    })
  });

  fetchArticleList(textQuery);
});