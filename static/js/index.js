window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options_teaser_computer = {
      slidesToScroll: 1,
      slidesToShow: 2,
      loop: true,
      infinite: true,
      autoplay: false,
      autoplaySpeed: 3000,
    };

    function checkDeviceAndDisplayVideos() {
        var isMobile = window.innerWidth <= 768 || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

        if (isMobile) {
            $('#teaser-videos-mobile').css('display', 'block');
            $('#teaser-carousel-computer').css('display', 'none');
        } else {
            $('#teaser-videos-mobile').css('display', 'none');
            $('#teaser-carousel-computer').css('display', 'block');

            var teaserCarousels = bulmaCarousel.attach('#teaser-carousel-computer', options_teaser_computer);

            for(var i = 0; i < teaserCarousels.length; i++) {
              // Add listener to  event
              teaserCarousels[i].on('before:show', state => {
                console.log(state);
              });
            }
        }
    }

    checkDeviceAndDisplayVideos();
    $(window).resize(function() {
        checkDeviceAndDisplayVideos();
    });


    var options_qual = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var qualCarousels = bulmaCarousel.attach('#qual-carousel', options_qual);

    // Loop on each carousel initialized
    for(var i = 0; i < qualCarousels.length; i++) {
    	// Add listener to  event
    	qualCarousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    
})