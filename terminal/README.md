# Terminal

The browser terminal is self-contained in this directory.

Files under `fs/` are exposed at the same absolute path in the virtual filesystem. After adding,
moving, or removing files, regenerate the static-site manifest:

```sh
node terminal/generate-fs-manifest.mjs
```

The generator recursively includes regular files and dot-directories. Text files can be read with
terminal commands such as `cat` and `grep`; other files are still listed and can be opened.
