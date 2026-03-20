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
    pars['exposure_factor'] = 0.85000
    pars['prob_use_intercept'] = -1.2
    pars['prob_use_trend_par'] = 0.05200
    pars['fecundity_low'] = 0.5981
    pars['fecundity_high'] = 1.5018
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.1633, 0.4832, 0.5831, 0.8000, 1.2000, 1.4000, 1.3000, 1.7000, 1.7000, 1.1000, 0.6000, 0.3000]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9506, 0.4868, 0.2425, 0.2182, 0.0577, 0.1803, 0.0778]])
    pars['method_weights'] = np.array([0.05, 0.05, 150, 1, 0.05, 0.5, 0.05, 0.2, 1])
    pars['dur_postpartum'] = 18
    spacing_pref_array = np.ones(17, dtype=float)
    spacing_pref_array[:2] =  0.3    # 0-6 months — suppress very short intervals
    spacing_pref_array[2:5] = 0.5    # 6-15 months — suppress to reduce 12-24mo births
    spacing_pref_array[5:13] = 1.0   # 15-39 months — normal (maps to 24-48mo births)
    spacing_pref_array[13:] = 0.15   # 39+ months — suppress >48mo births

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars


# %% Make and validate parameters
def dataloader(location='ethiopia'):
    return fpld.DataLoader(location=location)
