'''
A sample script that sets free parameters to fixed values, which are used to run a sim and generate plots showing the
discrepancies between the model and data.

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
'''
import numpy as np
import fpsim as fp
from fpsim import plotting as plt

# Name of the country being calibrated. To note that this should match the name of the country data folder
country = 'kenya'

plt.Config.set_figs_directory('./figs_manual_calib')
plt.Config.do_save = True
plt.Config.show_rmse = True


def run_sim():
        # Set up sim for country
        pars = fp.pars(location=country)
        pars['n_agents'] = 100  # Small population size
        pars['end_year'] = 2020  # 1960 - 2020 is the normal date range

        # Free parameters for calibration
        pars['fecundity_low'] = .8
        pars['fecundity_high'] = 3.25
        pars['exposure_factor'] = 3.5

        # Last free parameter, postpartum sexual activity correction or 'birth spacing preference'. Pulls values from {location}/data/birth_spacing_pref.csv by default
        # Set all to 1 to reset. Option to use 'optimize-space-prefs.py' script in this directory to determine values
        pars['spacing_pref']['preference'][:3] =  1  # Spacing of 0-6 months
        pars['spacing_pref']['preference'][3:6] = 1  # Spacing of 9-15 months
        pars['spacing_pref']['preference'][6:9] = 1  # Spacing of 18-24 months
        pars['spacing_pref']['preference'][9:] =  1  # Spacing of 27-36 months

        # Adjust contraceptive choice parameters
        cm_pars = dict(
            prob_use_year=2020,  # Time trend intercept
            prob_use_trend_par=0.03,   # Time trend parameter
            force_choose=False,        # Whether to force non-users to choose a method ('False' by default)
            method_weights=np.array([0.34, .7, 0.6, 0.74, 0.76, 1, 1.63, 0.65, 9.5])  # Weights for the methods in method_list in methods.py (excluding 'none', so starting with 'pill' and ending in 'othmod').
        )
        method_choice = fp.SimpleChoice(pars=cm_pars, location=country)   # Specifying contraceptive choice module (see RandomChoice, SimpleChoice, or StandardChoice in methods.py)

        # Only other free parameters are age-based exposure and parity-based exposure (which you can adjust manually in {country}.py) as well as primary_infertility (set to 0.05 by default)

        # Print out free params being used
        print("FREE PARAMETERS BEING USED:")
        print(f"Fecundity range: {pars['fecundity_low']}-{pars['fecundity_high']}")
        print(f"Exposure factor: {pars['exposure_factor']}")
        print(f"Birth spacing preference: {pars['spacing_pref']['preference']}")
        print(f"Age-based exposure and parity-based exposure can be adjusted manually in {country}.py")
        print(f"Contraceptive choice parameters: {cm_pars}")

        # Run the sim
        sim = fp.Sim(pars=pars, contraception_module=method_choice, analyzers=fp.cpr_by_age())
        sim.run()

        return sim


if __name__ == '__main__':
        # Run the simulation
        sim = run_sim()
        sim.plot()

        # Set options for plotting
        plt.plot_calib(sim)     # Function to plot the primary calibration targets (method mix, use, mcpr, tfr, birth spacing, afb, and asfr)
        plt.plot_cpr_by_age(sim)    # Function to plot cpr by age when this analyzer is used (useful for analysis and debugging)
