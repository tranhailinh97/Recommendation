import debounce from '/static/lodash.js';

$(document).ready(function () {
    const debouncedFetch = debounce(() => {
        $('.js-data-example-ajax').select2({
            ajax: ({
                url: '/get_songs',
                type: 'GET',
                dataType: 'json',
                processResults: function (data) {
                    var songSelect = $('.js-data-example-ajax');

                    // Add options for each song
                    data.songs.forEach((song, index) => {
                        var option = new Option(data.title[index],
                            song, false, false);
                        songSelect.append(option);
                    });

                    // Initialize Select2
                    songSelect.select2({
                        placeholder: "Select your song",
                        multiple: true,
                        closeOnSelect: false
                    })
                }
            })
        });
    });

    debouncedFetch();

    function matchCustom(params, data) {

        // If there are no search terms, return all of the data
        if ($.trim(params.term) === '') {
            return data;
        }

        // Do not display the item if there is no 'text' property
        if (typeof data.text === 'undefined') {
            return null;
        }

        // `params.term` should be the term that is used for searching
        // `data.text` is the text that is displayed for the data object
        if (data.text.indexOf(params.term) > -1) {
            var modifiedData = $.extend({}, data, true);
            modifiedData.text += ' (matched)';

            // You can return modified objects from here
            // This includes matching the `children` how you want in nested data sets
            return modifiedData;
        }

        // Return `null` if the term should not be displayed
        return null;
    }

    function getRecommendations() {
        var songSelect = document.getElementById("songSelect");
        var selectedUser = songSelect.options[songSelect.selectedIndex].value;

        // Call the API to get the user's playlist and recommendations
        fetch('/get_user_and_recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: selectedUser
                })
            })
            .then(response => response.json())
            .then(data => {
                var userSongsBody = document.getElementById("userSongsBody");
                userSongsBody.innerHTML = "";
                data.user_songs.forEach((userSong, index) => {
                    var row = userSongsBody.insertRow(index);
                    var cell = row.insertCell(0);
                    cell.innerHTML = userSong;
                });

                var recommendationsBody = document.getElementById("recommendationsBody");
                recommendationsBody.innerHTML = "";
                data.recommendations.forEach((recommendation, index) => {
                    var row = recommendationsBody.insertRow(index);
                    var cell = row.insertCell(0);
                    cell.innerHTML = recommendation.join(" - Score: ");
                });
            });
    }
});