"""
Smoke-test the scripts in examples/.

The examples are user-facing documentation, but nothing executed them, so several
broke silently during the v3.6.0 refactor (removed cum_births, fp.pars and
fp.MultiSim). Running them here means that breakage shows up as a test failure.

Each script runs in a subprocess so a script calling plt.show() or sys.exit()
cannot disturb the test session.
"""

import os
import sys
import subprocess
import sciris as sc
import pytest

examples_dir = sc.thispath(__file__).parent / 'examples'

# The calibration examples run full Optuna studies, which are far too slow for CI
skip_scripts = ['example_calib_auto.py', 'example_calib_manual.py']

scripts = sorted(p.name for p in examples_dir.glob('*.py') if p.name not in skip_scripts)


@pytest.mark.parametrize('script', scripts)
def test_example(script):
    """ Run an example script and check it exits cleanly """
    env = dict(os.environ, MPLBACKEND='Agg')  # Never open a window
    proc = subprocess.run([sys.executable, script], cwd=str(examples_dir), env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        errormsg = f'Example {script} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}'
        raise AssertionError(errormsg)
    return


if __name__ == '__main__':
    for script in scripts:
        print(f'Running {script}...')
        test_example(script)
    print('Done.')
