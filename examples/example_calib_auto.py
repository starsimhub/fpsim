'''
A sample script that uses the Calibration class to generate the optimal set of free parameters, runs a sim with that set
of free params, and then generates plots showing the discrepancies between the model vs data.

Note: running a calibration does not guarantee a good fit! You must ensure that
you run for a sufficient number of iterations, have enough free parameters, and
that the parameters have wide enough bounds. To modify calibration settings and see
further documentation, see calibration.py. By default, the target parameters that
the calibration classes uses are listed in the default_flags in experiment.py (which
is used by the calibration class to determine goodness of fit).

PRIOR TO RUNNING:
1. Be sure to set the plotting config variables in the first section below (country, figs directory, save option) as well as
any sim parameters and free params (for calibration) in the 'run_sim' function

2. Ensure that fpsim/locations contains both a directory for the country being calibrated, and ensure this location directory
 contains a corresponding location file (i.e. 'ethiopia.py') and 'data/' subdirectory

3. In order to run this script, the country data must be stored in the country directory mentioned above and with the
following naming conventions:

ageparity.csv' # Age-parity distribution file
use.csv' # Dichotomous contraceptive method use
birth_spacing_dhs.csv'  # Birth-to-birth interval data
afb.table.csv'  # Ages at first birth in DHS for women age 25-50
cpr.csv'  # Contraceptive prevalence rate data; from UN Data Portal
asfr.csv'  # Age-specific data fertility rate data
mix.csv'  # Contraceptive method mix
tfr.csv'  # Total fertility rate data
popsize.csv'  # Population by year

4. Ensure that the data in the aforementioned files are formatted properly (see files in locations/kenya/data for reference)

5. Verify desired settings in calibration.py (i.e. number of trials)
'''

import fpsim as fp
from fpsim import plotting as plt
import pylab as pl

# Name of the country being calibrated. To note that this should match the name of the country data folder
country = 'kenya'

# Modify the figs directory and ensure it exists
plt.Config.set_figs_directory('./figs_auto_calib')
plt.Config.do_save = True


def run_sim():
        # Set up sim for country
        pars = dict(location=country,
                    n_agents = 100, # Small population size
                    end_year = 2020 # 1960 - 2020 is the normal date range
                    )

        # Free parameters for calibration
        freepars = dict(
                fecundity_low = [0.95, 0.925, 0.975],       # [best, low, high]
                fecundity_high = [1.05, 1.025, 1.075],
                exposure_factor = [1.0, 0.95, 1.0],
        )

        # Last free parameter, postpartum sexual activity correction or 'birth spacing preference'. Pulls values from {location}/data/birth_spacing_pref.csv by default
        # Set all to 1 to reset. Option to use 'optimize-space-prefs.py' script in this directory to determine values
        pars['spacing_pref']['preference'][:3] =  1  # Spacing of 0-6 months
        pars['spacing_pref']['preference'][3:6] = 1  # Spacing of 9-15 months
        pars['spacing_pref']['preference'][6:9] = 1  # Spacing of 18-24 months
        pars['spacing_pref']['preference'][9:] =  1  # Spacing of 27-36 months

        # Here we are not specifying a contraceptive module, so by default the model uses StandardChoice,
        # in which contraceptive choice is based on age, education, wealth, parity, and prior use. Only other free
        # parameters are age-based exposure and parity-based exposure (which you can adjust manually in {country}.py)
        # as well as primary_infertility (set to 0.05 by default)

        calibration = fp.Calibration(pars, calib_pars=freepars, n_trials=2)
        calibration.calibrate()
        calibration.summarize()
        fig = calibration.plot_best()
        pl.savefig("calib_best.png", bbox_inches='tight', dpi=100)

        fig = calibration.plot_trend()
        pl.savefig("calib_trend.png", )

        pars.update(calibration.best_pars)
        sim = fp.Sim(pars=pars)
        sim.run()

        return sim


if __name__ == '__main__':
        # Run the simulation
        sim = run_sim()
        sim.plot()

        # Set options for plotting
        plt.plot_calib(sim)
