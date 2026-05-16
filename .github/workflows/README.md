# Setting up PyPI publishing

In addition to running CI/CD tests, FPsim versions can also be released on PyPI via a GitHub Action. This is controlled by `pypi_release.yml`.

To set up the integration, see [this article](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).

In short, the steps are:
1. Log into the owner's PyPI account
2. Go to https://pypi.org/manage/project/fpsim/settings/publishing
3. Fill in the form (in this case: `fpsim`, `fpsim`, `fpsim`, `pypi_release.yml`, `pypi`)
4. Click "Add"

That's it! This shouldn't need to be done again, unless the owner or workflow name changes.

To use, go to the FPsim GitHub repo, go to "Actions", click on the PyPI Release workflow, and click "Run".