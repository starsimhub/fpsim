# FPsim: Family Planning Simulator

This repository contains the code for the Institute for Disease Modeling's family planning simulator, FPsim.

**FPsim is currently under development**.

## User guide

FPsim is designed as an open-source tool for family planning research.
However, it is not a silver bullet tool. It is designed to answer
complex questions about emerging dynamics in complex social and behavioral systems. Its strength stems from a life-course approach,
which allows researchers to examine how compounding and temporal effects unfold over women's lives, and how these individual-level changes lead to macro-level outcomes.

Before using FPsim, please refer to the following guidelines:

 * FPsim is only as good as the data and assumptions provided. Be sure you are familiar with both before using FPsim.
 * FPsim is not a replacement for good data. The model cannot tell you what demand for a hypothetical method will be.
 * FPsim is not a replacement for descriptive statistics. Before using FPsim, assess your primary research question(s). Can they be answered using descriptive statistics?
 * FPsim cannot predict exogenous events. Use caution when interpreting and presenting results. For example, FPsim cannot predict regional conflicts or pandemics, nor their impacts on FP services.


## Repo structure

The structure is as follows:

- FPsim, in the folder `fpsim`, is a standalone Python library for performing family planning analyses.
- Within `fpsim`, the `locations` folder contains parameters and input data for all countries currently calibrated.
- Docs are in the `docs` folder.
- Examples are in the `examples` folder.
- Tests are in the `tests` folder.


## Installation

Run `pip install fpsim` to install and its dependencies from PyPI. Alternatively, clone the repository and run `pip install -e .` (including the final dot!). To be able to run tests and build the docs as well, use `pip install -e .[dev]`.


## Documentation

Documentation is available at https://docs.fpsim.org.


## Disclaimer

The code in this repository was developed by IDM and other collaborators to support our joint research on family planning. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. Note that FPsim depends on a number of user-installed Python packages that can be installed automatically via `pip install`. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the MIT License. See the contributing and code of conduct READMEs for more information.
