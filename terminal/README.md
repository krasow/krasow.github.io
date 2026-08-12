# Terminal

The shared terminal engine powers both the browser terminal at `/terminal/` and
the [`@krasow/tty`](https://www.npmjs.com/package/@krasow/tty) CLI:

```sh
npx @krasow/tty
```

## Layout

- `index.html`, `js/terminal.js`, and `js/resize.js` host the browser UI.
- `js/engine.js`, `js/commands.js`, `js/apps/`, and `js/games/` implement the
  shared shell engine.
- `js/filesystem.js`, `js/autocomplete.js`, and `fs/` provide files and shell
  navigation.

Files under `fs/` map to equivalent virtual paths. A `.pg` file links to a page,
a `.terminal` sidecar links one asset, and `.terminal.json` defines virtual folder
entries. After changing `fs/`, regenerate its manifest from the repository root:

```sh
node terminal/generate-fs-manifest.mjs
```

Browser history, theme, terminal size, game scores, and virtual removals are
stored in `localStorage`; `reset` clears them.
