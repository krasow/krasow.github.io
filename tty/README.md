# krasow.dev — in your terminal

The [krasow.dev](https://krasow.dev) website has a built-in terminal. This is
the same terminal, in a real TTY.

```sh
node <(curl -fsSL https://krasow.dev/tty/krasow.mjs)
# or, from a checkout:
node tty/krasow.mjs
```

Try `help`, `ls`, `cat about.pg`, `chat what does he work on?`, or `snake`.

## How it works

It is a **host, not a reimplementation**. `krasow.mjs` loads the website's real
engine modules (`terminal/js/{filesystem,commands,autocomplete}.js`, every
`apps/*`, `chat.js`, `snake.js`) exactly as the browser's `<script>` tags do,
supplies the handful of browser globals they reference (a tiny in-memory DOM,
`localStorage`, `getComputedStyle`, clipboard), and renders their output to the
terminal instead of the DOM. Only `terminal.js`/`resize.js` — the DOM host — are
replaced; this file is their Node equivalent.

Filesystem and page **content** is fetched live from krasow.dev, so the site
stays 100% static hosting. Page text (`cat about.pg`) is pre-parsed at build
time (see `terminal/lib/page-text.mjs`) and served as plain text, so nothing
parses HTML at runtime — in the browser or here.

- No runtime dependencies. Requires Node ≥ 18.
- Point at a local preview with `KRASOW_BASE=http://localhost:8000 node tty/krasow.mjs`.
- Disable color with `NO_COLOR=1`.

Run from a checkout and it loads the engine from `../terminal/js`; run the
shipped single file and it fetches the engine from the live site.
