"""
Set the parameters for Cotedivoire.
"""
import numpy as np
import os
import pandas as pd
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    pars = {}
    pars['fecundity_low'] = 0.7
    pars['fecundity_high'] = 1.1
    pars['exposure_factor'] = 2
    pars['prob_use_year'] = 2020
    pars['prob_use_trend_par'] = 0.01
    pars['prob_use_intercept'] = 0.5
    pars['method_weights'] = np.array([8, 4, 5, 20, 2, 2, 0.02, 0.015, 5])
    pars['dur_postpartum'] = 23

    spacing_pref_array = np.ones(13, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:3] =  1.0  # Spacing of 0-6 months
    spacing_pref_array[3:4] = 1.0  # Spacing of 9 months
    spacing_pref_array[4:9] = 0.3  # Spacing of 12-24 months
    spacing_pref_array[9:17] =  1.1  # Spacing of 27-48 months
    spacing_pref_array[17:] =  1.0  # Spacing of 51-56 months

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    pars['exposure_age'] = np.array([[0,    5,  10, 12.5, 15, 18, 20, 25,  30,  35, 40, 45, 50],
                                        [0.1, 0.1, 0.5,  0.1,0.1, 0.6,1.4,0.56,0.5,0.3,0.2,0.5,0.2]])  # <<< USER-EDITABLE: Can be modified for calibration
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])  # <<< USER-EDITABLE: Can be modified for calibration


    return pars



def dataloader(location='cotedivoire'):
    return fpld.DataLoader(location=location)
