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
    pars['exposure_factor'] = 1.2985
    pars['prob_use_intercept'] = -1.8967
    pars['prob_use_trend_par'] = 0.0487
    pars['fecundity_low'] = 0.6814
    pars['fecundity_high'] = 1.1504
    pars['method_weights'] = np.array([8, 4, 5, 20, 2, 2, 3, 5, 5])
    pars['dur_postpartum'] = 18

    spacing_pref_array = np.ones(19, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:4] =  1.0    # 0-12 months
    spacing_pref_array[4:8] = 0.1    # 12-24 months — suppress (model over)
    spacing_pref_array[8:13] = 1.0   # 24-39 months
    spacing_pref_array[13:] = 0.3    # 39-54+ months — suppress (data: 27% >48)

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0, 0.1325, 0.4272, 0.3416, 0.7857, 0.5719, 2.3741, 0.6273, 0.932, 0.8956, 1.4544, 0.456, 0.49]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars



def dataloader(location='cotedivoire'):
    return fpld.DataLoader(location=location)
