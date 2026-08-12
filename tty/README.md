# @krasow/tty

Run the interactive terminal from [krasow.dev](https://krasow.dev) in your own
terminal.

```sh
npx @krasow/tty
```

Requires Node.js 18 or newer. Try `help`, `ls`, `cat about.pg`, or `snake` once
it starts.

The CLI loads the terminal and its content from krasow.dev. To use another host,
set `KRASOW_BASE`:

```sh
KRASOW_BASE=http://localhost:8000 npx @krasow/tty
```
