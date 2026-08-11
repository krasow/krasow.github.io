# Terminal

The browser terminal and its virtual filesystem are self-contained in this directory.

## Structure

- `index.html` is the terminal page.
- `js/terminal.js` manages input, output, history, completion, and shared filesystem primitives.
- `js/commands.js` registers apps and dispatches commands, responses, and easter eggs.
- `js/apps/` contains commands with meaningful behavior or state; trivial built-ins stay in the
  registry.
- `js/games/` contains terminal games.
- `js/filesystem.js` loads the generated filesystem and manages locally hidden entries.
- `js/resize.js` handles mouse resizing.
- `fs/` mirrors the virtual filesystem.

## Virtual filesystem

Files under `fs/` appear at the equivalent terminal path. For example,
`fs/home/contact.md` becomes `/home/contact.md`. Hidden files remain available but are omitted from
normal `ls` output. Text files work with commands such as `cat`, `grep`, and `wc`.

After changing anything under `fs/`, regenerate the manifest:

```sh
node terminal/generate-fs-manifest.mjs
```

### Pages

A `.pg` file represents a page. Its first line is the destination URL:

```text
/pages/about.html
```

Optional `key=value` lines can provide a different readable source and CSS selector:

```text
/index.html#experience
source=/homepage/experience.html
selector=.holder
```

### Linked files

A `.terminal` sidecar exposes an existing asset without copying it into `fs/`:

```text
# fs/home/contact.vcf.terminal
/assets/documents/Krasowska_David_contact.vcf
```

The terminal displays `contact.vcf`; the descriptor remains hidden. Additional metadata can use
`key=value` lines. Linked files default to `text=false`.

### Virtual folders

A folder-level `.terminal.json` registers entries that point to existing assets:

```json
{
  "entries": [
    { "name": "JULIACON25", "target": "/assets/documents/slides/2025/juliacon.pdf" },
    { "name": "LEGION24", "target": "/assets/documents/slides/2024/legion24.pdf" }
  ]
}
```

The descriptor is hidden, and its folder is registered even when `entries` is empty. Entries
support `target` and an optional `text` flag.

## Local state

Command history, transcript, theme, terminal size, Snake high score, and files hidden with `rm` are
stored in `localStorage`. The `reset` command restores the default local state.
