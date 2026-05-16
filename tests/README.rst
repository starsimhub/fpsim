=================
Integration tests
=================

This folder contains the core tests for FPsim. 

Run ``pip install -r requirements_test.txt`` before running the tests.

Recommended usage is ``./check_coverage`` or ``./run_tests``. You can also use ``pytest`` to run all the tests in the folder. Description of other scripts included for convenience are below.


check_coverage
--------------

Run all tests with parallelization, determine code coverage, and create an HTML report.


check_baseline
--------------

Just check if the baselines match.


run_tests
---------

Run all tests, with parallelization, and showing how long each test took.


update_baseline
---------------

The test ``test_baselines.py`` checks to see if results changed unintentionally. If you *intended* to change them, run this script to update the saved results.