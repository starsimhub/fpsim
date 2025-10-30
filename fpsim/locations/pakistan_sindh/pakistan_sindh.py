"""
Set the parameters for Pakistan Sindh.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():

    print("INFO: Currently FPsim-Sindh uses PMA data from Rajasthan India as a placeholder. This calibration is not externally validated, and users should proceed with caution in using and interpreting FPsim for Sindh.")

    pars = {}
    pars['fecundity_low'] = 0.1
    pars['fecundity_high'] = 0.9
    pars['exposure_factor'] = 4
    pars['prob_use_year'] = 2020
    pars['prob_use_trend_par'] = 0.01
    pars['prob_use_intercept'] = -2.7
    pars['method_weights'] = np.array([0.06, 2.3, 20, 2, 2.5, 1.5, 3, 0.1, 0.01])
    pars['dur_postpartum'] = 23

    spacing_pref_array = np.ones(18, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:3] =  1
    spacing_pref_array[3:6] = 1
    spacing_pref_array[6:9] = 5
    spacing_pref_array[9:] =  0.1

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    pars['exposure_age'] = np.array([[0,     5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                        [0.2, 0.2, 0.2,  1 ,0.9, 0.8,0.8,1.25,1, 0.7,0.5,0.5, 0.5]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars


def dataloader(location='pakistan_sindh'):
    return fpld.DataLoader(location=location)