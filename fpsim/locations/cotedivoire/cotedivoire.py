"""
Set the parameters for Cotedivoire.
"""
import numpy as np
import os
import pandas as pd
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 0.76
    pars['prob_use_intercept'] = -0.6357
    pars['prob_use_trend_par'] = -0.0050
    pars['fecundity_low'] = 0.8
    pars['fecundity_high'] = 1.6
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.3000, 0.3900, 0.4700, 1.0000, 1.3000, 1.65000, 1.7000, 1.6000, 1.9000, 1.4000, 0.8000, 0.4500]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.7742, 0.4464, 0.2942, 0.3817, 0.2869, 0.1676, 0.0534]])
    pars['method_weights'] = np.array([8, 4, 4, 20, 2, 2, 2, 3, 5])
    pars['dur_postpartum'] = 18
    spacing_pref_array = np.ones(19, dtype=float)
    spacing_pref_array[:2] =  0.5    # 0-6 months pp — suppress very short intervals
    spacing_pref_array[2:5] = 0.35    # 6-15 months pp — suppress to reduce 12-24mo births
    spacing_pref_array[5:13] = 0.4    # 15-24 months pp — slight suppress
    spacing_pref_array[13:] = 0.2    # 39+ months pp — suppress >48mo births

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars



def dataloader(location='cotedivoire'):
    return fpld.DataLoader(location=location)
