"""
Set the parameters for Pakistan Sindh.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():

    print("INFO: Currently FPsim-Sindh uses PMA data from Rajasthan India as a placeholder. This calibration is not externally validated, and users should proceed with caution in using and interpreting FPsim for Sindh.")

    pars = {}
    pars['exposure_factor'] = 0.9
    pars['prob_use_intercept'] = -0.3
    pars['prob_use_trend_par'] = -0.12
    pars['fecundity_low'] = 0.6881
    pars['fecundity_high'] = 1.5827
    pars['method_weights'] = np.array([0.44, 2.3, 10, 3, 3, 2, 2, 0.01, 0.01])
    pars['dur_postpartum'] = 18

    spacing_pref_array = np.ones(19, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:4] =  1.0   # 0-12 months
    spacing_pref_array[4:8] = 1.0   # 12-24 months
    spacing_pref_array[8:13] = 1.0  # 24-39 months
    spacing_pref_array[13:] = 0.15  # 39-54+ months — suppress

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0, 0.1737, 0.2362, 0.4481, 0.6, 0.85, 0.9, 1.0, 0.9, 0.7, 0.5, 0.25, 0.1]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars


def dataloader(location='pakistan_sindh'):
    return fpld.DataLoader(location=location)
