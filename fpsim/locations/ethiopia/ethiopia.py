"""
Set the parameters for FPsim, specifically for Ethiopia.
"""
import numpy as np
from pathlib import Path
from fpsim import defaults as fpd
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    pars = {}
    pars['fecundity_low'] = 0.5
    pars['fecundity_high'] = 0.8
    pars['exposure_factor'] = 3.0
    pars['prob_use_year'] = 2020
    pars['prob_use_trend_par'] = 0.00005
    pars['prob_use_intercept'] = 0.05
    pars['method_weights'] = np.array([
        0.008,    # Pill
        0.0008,   # IUDs
        68.0,     # Injectables
        0.4,      # Condoms
        0.001,    # BTL
        0.2,      # Withdrawal
        0.002,    # Implants
        0.003,    # Other traditional methods
        0.1       # Other modern methods
    ])
    pars['dur_postpartum'] = 23

    spacing_pref_array = np.ones(18, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:3] =  0.1
    spacing_pref_array[3:6] = 2.5
    spacing_pref_array[6:9] = 1.8
    spacing_pref_array[9:] =  2
    
    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    pars['exposure_age'] = np.array([[0,     5,  10, 12.5, 15, 18, 20, 25,  30, 35,  40, 45,    50],
                                        [0.1, 0.1, 0.5,  3,   2,  0.5,0.5,1.5, 1.5, 1.5,   1.5,  1.5,   0.5]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [5, 5, 5, 5, 1, 1, 1, 0.8, 0.5, 0.3, 0.01, 0.01, 0.01, 0.01]])

    return pars

# %% Make and validate parameters
def dataloader(location='ethiopia'):
    return fpld.DataLoader(location=location)