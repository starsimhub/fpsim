# FPsim docs

## Tutorials

Please see the `tutorials` subfolder.

## Everything else

This folder includes source code for building the docs. Users are unlikely to need to do this themselves. Instead, view the FPsim docs at https://docs.fpsim.org.

To build the docs, follow these steps:

1.  Make sure dependencies are installed::
    ```
    pip install -r requirements.txt
    ```

2.  Make the documents; run from the `docs` folder:
    - `./build_docs` — copy docs to a cache (`.docs_build`), execute notebooks there, then build. Notebooks in the repo are never modified. Takes a few minutes.
    - `./build_docs never` — build without executing notebooks (quick, ~15 s). Uses `mkdocs build` with `execute: false`.
    Or from the repo root: `mkdocs build` (no notebook execution).

3.  The built site is in `site/` at the repo root.
