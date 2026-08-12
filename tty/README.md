# krasow.dev — in your terminal

The terminal from [krasow.dev](https://krasow.dev), running in your own terminal.

```sh
curl -fsSL https://krasow.dev/tty/krasow.mjs -o /tmp/krasow.mjs && node /tmp/krasow.mjs
```

Needs Node 18 or newer, and nothing to install. Once it starts, try `help`,
`ls`, `cat about.pg`, or `snake`.

## How it works

The website's terminal and this one run the same code. Every command — the
filesystem, chat, snake, all of it — lives in `terminal/js/` and is shared. This
file just lets that code run in a real terminal instead of a browser tab, and it
loads its content live from krasow.dev, so the site stays a plain static site.

Set `KRASOW_BASE` to load content from somewhere other than `https://krasow.dev`
— point it at a local copy while developing.
