// Homepage terminal widget: hover or arrow keys switch the visible response.
(function () {
  const termCmds = document.querySelectorAll('.terminal-split .command');

  function activateTerminal(idx) {
    termCmds.forEach((c) => c.classList.remove('active'));
    document
      .querySelectorAll('.terminal-split .response')
      .forEach((r) => r.classList.remove('active'));
    termCmds[idx].classList.add('active');
    document.querySelector(`.terminal-split .response[data-idx="${idx}"]`).classList.add('active');
  }

  termCmds.forEach((cmd) => {
    cmd.addEventListener('mouseenter', () => activateTerminal(+cmd.dataset.idx));
  });

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    const active = document.querySelector('.terminal-split .command.active');
    if (!active) return;
    e.preventDefault();
    const cur = +active.dataset.idx;
    const next =
      e.key === 'ArrowDown' ? Math.min(cur + 1, termCmds.length - 1) : Math.max(cur - 1, 0);
    activateTerminal(next);
  });
})();
