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

2.  Build the documents from the repo root::
    ```
    mkdocs build
    ```

3.  The built site is in `site/` at the repo root.
