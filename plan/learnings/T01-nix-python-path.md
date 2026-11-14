# Learning: Do not override PYTHONPATH in the Nix shell

**Date:** 2026-11-14
**Author:** J
**Task:** T01
**Status:** confirmed

## What we learned

In the Nix shell, setting `PYTHONPATH=.` replaces the Nix-provided Python path and breaks imports such as `numpy` and `yaml`.

## Evidence

```sh
nix-shell -p python3Packages.numpy --run 'python -c "import numpy"'
# works

nix-shell -p python3Packages.numpy --run 'PYTHONPATH=. python -c "import numpy"'
# ModuleNotFoundError
```

## Impact

- Run Python from the repo root without overriding `PYTHONPATH`.
- The current directory is already on `sys.path` for `python -m` and scripts in the repo.

## Plan changes

- [x] No code change needed.
- [x] Use `python scripts/...` from the repo root.

## Follow-up

If a custom path is required, append instead of replace:

```sh
PYTHONPATH="$PYTHONPATH:." python ...
```
