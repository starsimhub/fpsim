FPsim: Family Planning Simulator
================================

This repository contains the code for the Institute for Disease Modeling's family planning simulator, FPsim. 

User guide
------------
FPsim is designed as an open-source tool for family planning research. 
However, it is not a silver bullet tool. It is designed to answer
complex questions about emerging dynamics in complex social and behavioral systems. Its strength stems from a life-course approach, 
which allows researchers to examine how compounding and temporal effects unfold over women's lives, and how these individual-level changes lead to macro-level outcomes.

Before using FPsim, please refer to the following guidelines:

 * FPsim is only as good as the data and assumptions provided. Be sure you are familiar with both before using FPsim.
 * FPsim is not a replacement for good data. The model cannot tell you what demand for a hypothetical method will be.
 * FPsim is not a replacement for descriptive statistics. Before using FPsim, assess your primary research question(s). Can they be answered using descriptive statistics? 
 * FPsim cannot predict exogenous events. Use caution when interpreting and presenting results. For example, FPsim cannot predict regional conflicts or pandemics, nor their impacts on FP services.


Repo structure
--------------

The structure is as follows:

- FPsim, in the folder ``fpsim``, is a standalone Python library for performing family planning analyses.
- Within ``fpsim``, the ``locations`` folder contains parameters and input data for all countries currently calibrated.
- Docs, including tutorials, are in the ``docs`` folder.
- Examples are in the ``examples`` folder.
- Tests are in the ``tests`` folder.


Installation
------------

If you are working with the main branch, you can run ``pip install fpsim`` to install FPsim and its dependencies from PyPI.
Alternatively, clone the repository and run ``pip install -e .`` (including the final dot!). This is particularly useful if you want to collaborate on a branch that has not yet been merged. 


Documentation
-------------

Documentation is available at https://docs.fpsim.org.


Contributing
------------

Style guide
```````````

Please follow the starsim style guide at: https://github.com/amath-idm/styleguide

Issues
```````

* Everything you're working on must be linked to an issue. If you notice that something needs to be done (even small things or things nearly finished) and there isn't an issue for it, create an issue. This helps track who is doing what and why.
* Label issues you are currently working on with ``in progress`` for tracking purposes - and to avoid accidental replication of work.
* High priority issues are organized from top (most urgent) to bottom (least urgent) and can be labelled with ``urgent`` or ``blocking`` as appropriate. If you are working on something that is urgent or blocks other development, please set a reasonable deadline for review (can be updated, of course). 
* The Hydra Head Effect: Often when you solve one issue, two more pop up in its place. When this happens, close the original issue and start new issues (linked) to be triaged. 
* If your issue has more than two distinct tasks associated with it, please include a check list in the text, so that we can track which components of the issue have been resolved and which need to be supported. 
* If your issue is a bug that was not caught by test, and includes a specific expected value that can be hard-checked, please either include or request a test patch so that a test fails due to the bug

Pull requests
`````````````

* ALL PRs should be linked to at least one issue. As above, if you're working on a PR and there's no issue associated with it, you can create an issue.  
* All PRs should have a non-author assigned for review. If the PR review is assigned to more than one person, use the comment section to assign a main reviewer. Use your best judgement here, as roles shift, but in general: 

   - @MOBrien-IDM as FPsim lead (approval required to merge)
   - Anyone you've worked with on this issue 1:1
   - @cliffckerr to ensure new feature performance and compatibility with FPsim
   - @mzimmermann-IDM for subject matter expertise, economic and empowerment questions, questions about modeling best practices
   - @avictorious for questions about historical FPsim decisions and subject matter expertise

* Keep PRs as small as possible: e.g., one issue, one PR. Small PRs are easier to review and merge. 
* At times there may be a backlog of issues, but there should never be a big backlog of PRs. (If you're unsure whether to make a PR, write a detailed issue first.)

   - What if there are two people working on PRs at the same time?
   - Take a look at the issue priority. The PR addressing the higher priority issue should merge first. Make sure you pull the new master after that merge before you push changes for your PR. If both issues are high priority, the one with more time-sensitive commits should be merged first. If you're unsure, ask. 

* If we do have a backlog of PRs, it's fine to make a new branch off your current PR, and make a new PR from that. These "cumulative PRs are not ideal, but they are better than creating merge conflicts with yourself!
* Before starting work, always ensure you've pulled from master. If you spend more than a few days on your PR, make sure you pull from master regularly. Before making a PR, ensure that your branch is up to date with master.
* Please create a draft PR on an active branch as soon as you're ready. Be generous in creating draft PRs. It helps with transparency and allows for quicker support if you run into a problem.
* Make sure tests pass on your PR. If they don't, mark the PR as draft until they do.
* Even if your work isn't ready for a PR, push it regularly. A guiding principle is to commit every few minutes and push to your branch every 1-2 hours.
* Every PR that adds a new feature or new functionality which can be hard-checked (so, excluding plotting functionality etc.) should include a corresponding unittest

Testing
````````

Development and debugging
+++++++++++++++++++++++++
- Developers are responsible for ensuring the functionality of new features they develop
   * Debugging and testing code are core features of ensuring functionality
   * When debugging in active development mode, ensure that your new feature is compatible with not only a single run of FPsim, but also the multisim scenarios
   * Ensure new features are compatible with introducing a novel method in scenarios
   * Use example_scens.py to quickly debug your new feature during development

Test coverage
+++++++++++++
- Every time a new feature is added, the developer should develop a unittest which checks the basic implementation of the feature.
- A unittest is simply a function starting with "test" that implements a feature as succinctly as possibly, and checks the expected output with an assertion.
- If you're having trouble starting a unittest feel free to look at some examples `here <https://github.com/amath-idm/fp_analyses/blob/master/tests/test_scenarios.py>`_.
- `Some test suites <https://github.com/amath-idm/fp_analyses/blob/master/tests/test_states.py>`_ organize the tests into a class with a configuration function called ``setUp()``. After implementing a unittest in such a class you may want to take advantage of the shared assets defined in ``setUp()`` to minimize the number of lines of code in your test.
- The new unittest should follow style guidelines laid out in the `starsim style guide <https://github.com/amath-idm/styleguide/tree/testing>`_.
- The new test should contain a docstring that details what is being tested, how it is tested (what it's being checked against), and the expected value.
- The test should display error message information that is sufficient to create a bug report (summary, expected value, and actual value).


Disclaimer
----------

The code in this repository was developed by IDM and other collaborators to support our joint research on family planning. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. Note that FPsim depends on a number of user-installed Python packages that can be installed automatically via ``pip install``. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the MIT License. See the contributing and code of conduct READMEs for more information.
