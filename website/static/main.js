$( document ).ready(function() {
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
      var target = $(e.target).attr("href") // activated tab
    });
});