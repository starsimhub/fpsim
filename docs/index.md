# FPsim

[![tests](https://github.com/fpsim/fpsim/actions/workflows/tests.yaml/badge.svg)](https://github.com/fpsim/fpsim/actions/workflows/tests.yaml)
[![PyPI](https://img.shields.io/pypi/v/fpsim?label=PyPI)](https://pypi.org/project/fpsim/)

FPsim is the Institute for Disease Modeling's family planning simulator, an agent-based model for family planning research. FPsim uses the [Starsim](https://starsim.org) architecture, and belongs to the Starsim model suite which also includes [Covasim](https://covasim.org), [HPVsim](https://hpvsim.org), and [STIsim](https://stisim.org).

FPsim is designed as an open-source tool for family planning research. It uses a life-course approach, which allows researchers to examine how compounding and temporal effects unfold over women's lives, and how these individual-level changes lead to macro-level outcomes.

## Requirements

Python 3.9-3.13.

## Installation

FPsim is most easily installed via PyPI: `pip install fpsim`.

FPsim can also be installed locally. To do this, clone first this repository, then run `pip install -e .` (don't forget the dot at the end!).

## Usage and documentation

Documentation is available at https://docs.fpsim.org.

## Contributing

Questions or comments can be directed to [info@starsim.org](mailto:info@starsim.org), or on this project's [GitHub](https://github.com/fpsim/fpsim) page.

## Disclaimer

The code in this repository was developed by IDM and other collaborators to support our joint research on family planning. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. Note that FPsim depends on a number of user-installed Python packages that can be installed automatically via `pip install`. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the MIT License.
