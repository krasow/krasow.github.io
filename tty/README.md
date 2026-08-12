# krasow.dev — in your terminal

The [krasow.dev](https://krasow.dev) website terminal, in a real TTY.

```sh
curl -fsSL https://krasow.dev/tty/krasow.mjs -o /tmp/krasow.mjs && node /tmp/krasow.mjs
```

Requires Node ≥ 18, no dependencies. Try `help`, `ls`, `cat about.pg`, `snake`.

It's a host, not a reimplementation: it loads the site's real engine modules
(`terminal/js/*`) and renders their output to the terminal instead of the DOM.
Env: `KRASOW_BASE` (content source, default krasow.dev), `NO_COLOR`.
