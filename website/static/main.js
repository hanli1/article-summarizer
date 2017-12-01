$( document ).ready(function() {
  $("#summarize-button").click(send_summarize_query);
  function send_summarize_query(){
    $("#summarize-button").blur();
    data = {
        text: $("#dynamic-text-area").val(),
        length: "short"
    }
    $.get('api/summarize', data, function(response){
      var summary = response["result"];
      $("#summary-display-area").text(summary);
    });
  }

    $('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
      var target = $(e.target).attr("href") // activated tab
    });
});