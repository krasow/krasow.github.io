/* Google Analytics. Include as a real <script src> on each page — not via an
   innerHTML-injected fragment, where the script would never execute. */
(function () {
  var GA_ID = 'G-HRZSME3FC1';

  // Load the gtag.js library (the request GA looks for).
  var s = document.createElement('script');
  s.async = true;
  s.src = 'https://www.googletagmanager.com/gtag/js?id=' + GA_ID;
  document.head.appendChild(s);

  window.dataLayer = window.dataLayer || [];
  window.gtag = function () { dataLayer.push(arguments); };
  gtag('js', new Date());
  gtag('config', GA_ID);
})();
