"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 4.65) """
    pars = {}
    pars['exposure_factor'] = 0.5543
    pars['prob_use_intercept'] = -0.4945
    pars['prob_use_trend_par'] = 0.0355
    pars['fecundity_low'] = 0.8390
    pars['fecundity_high'] = 1.3105
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.4729, 0.4282, 0.5124, 1.2211, 1.8405, 1.5522, 1.0511, 1.2528, 1.1136, 0.4651, 0.1810, 0.2428]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.8253, 0.7074, 0.4202, 0.3672, 0.0402, 0.1769, 0.0723]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9097, 0.8193, 0.7290, 0.6387, 0.5483, 0.4580, 0.3677])
    }
    return pars


def dataloader(location='nigeria_kano'):
    return fpld.DataLoader(location=location)
