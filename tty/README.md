# @krasow/tty

Run the interactive terminal from [krasow.dev](https://krasow.dev) in your own
terminal.

```sh
npx @krasow/tty
```

Requires Node.js 18 or newer. Try `help`, `ls`, `cat about.pg`, or `snake` once
it starts.

The terminal engine is bundled in the npm package. The CLI fetches only terminal
content and its validated filesystem manifest from krasow.dev. To use another
HTTPS host, or an HTTP server on localhost, set `KRASOW_BASE`:

```sh
KRASOW_BASE=http://localhost:8000 npx @krasow/tty
```

Downloads are limited to same-origin contact cards, capped at 1 MiB, and never
overwrite an existing file.
