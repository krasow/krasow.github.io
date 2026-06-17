// Mobile navbar toggle — uses event delegation so it works after dynamic injection
document.addEventListener('click', function (e) {
  var nav = document.getElementById('myTopnav');
  if (!nav) return;

  var toggleBtn = e.target.closest('#navbarToggle');

  if (toggleBtn) {
    e.stopPropagation();
    nav.classList.toggle('open');
    toggleBtn.innerHTML = nav.classList.contains('open') ? '&times;' : '&#9776;';
    return;
  }

  // Close when tapping outside the navbar
  if (!e.target.closest('#myTopnav')) {
    nav.classList.remove('open');
    var btn = document.getElementById('navbarToggle');
    if (btn) btn.innerHTML = '&#9776;';
  }
});
