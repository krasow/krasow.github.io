// Dark/light theme toggle.
// The initial theme is set synchronously by an inline snippet in each page's
// <head> (to avoid a flash); this file handles toggling, persistence, and
// keeping the button icon in sync. Uses event delegation so it works after the
// navbar is injected dynamically.
(function () {
  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') || 'light';
  }

  function syncIcon() {
    var icon = document.querySelector('#themeToggle i');
    if (!icon) return;
    // Show the icon for the theme you'd switch TO.
    var dark = currentTheme() === 'dark';
    icon.className = dark ? 'fa fa-sun-o' : 'fa fa-moon-o';
  }

  function applyTheme(theme) {
    var d = document.documentElement;
    d.setAttribute('data-theme', theme);
    // Keep the inline background (set pre-paint to avoid FOUC) in sync — an
    // inline style would otherwise override the stylesheet's --bg-page rule.
    d.style.backgroundColor = theme === 'dark' ? '#121212' : '#ffffff';
    try { localStorage.setItem('theme', theme); } catch (e) {}
    syncIcon();
  }

  document.addEventListener('click', function (e) {
    if (!e.target.closest('#themeToggle')) return;
    e.stopPropagation();
    applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  });

  // The navbar is injected asynchronously, so watch for the button to appear
  // and sync its icon to the active theme once it's in the DOM.
  syncIcon();
  if (document.querySelector('#themeToggle')) return;
  var observer = new MutationObserver(function () {
    if (document.querySelector('#themeToggle')) {
      syncIcon();
      observer.disconnect();
    }
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
