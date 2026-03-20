"""
Set the parameters for FPsim, specifically for Ethiopia.
"""
import numpy as np
from pathlib import Path
from fpsim import defaults as fpd
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 1.2
    pars['prob_use_intercept'] = -1.0
    pars['prob_use_trend_par'] = 0.04
    pars['fecundity_low'] = 0.6498
    pars['fecundity_high'] = 1.7560
    pars['method_weights'] = np.array([0.2, 0.2, 100, 1, 0.15, 0.5, 0.05, 0.5, 1])
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0, 0.4854, 0.4431, 0.35, 1.0, 0.9, 0.95, 1.0, 1.2, 1.1, 0.8, 0.4, 0.15]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])
    pars['dur_postpartum'] = 18

    # 18 bins (0-48 months, 3-month intervals). Suppress >48 via last bin.
    spacing_pref_array = np.ones(17, dtype=float)
    spacing_pref_array[:4] =  1.0    # 0-12 months
    spacing_pref_array[4:8] = 1.0    # 12-24 months
    spacing_pref_array[8:13] = 1.0   # 24-39 months
    spacing_pref_array[13:] = 0.15   # 39-48+ months — suppress to reduce >48 births

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars


# %% Make and validate parameters
def dataloader(location='ethiopia'):
    return fpld.DataLoader(location=location)
