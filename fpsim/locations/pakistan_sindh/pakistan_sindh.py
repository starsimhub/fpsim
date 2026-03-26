"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 4.23) """
    pars = {}
    pars['exposure_factor'] = 0.7165
    pars['prob_use_intercept'] = -2.5276
    pars['prob_use_trend_par'] = 0.0101
    pars['fecundity_low'] = 0.5669
    pars['fecundity_high'] = 1.6169
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.2535, 0.2845, 0.9287, 1.4366, 0.3015, 1.8251, 0.8591, 0.7632, 1.2604, 0.8532, 0.4214, 0.2945]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9860, 0.2184, 0.5439, 0.2706, 0.1680, 0.0884, 0.0315]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.8882, 0.7764, 0.6646, 0.5528, 0.4410, 0.3292])
    }
    return pars


def dataloader(location='pakistan_sindh'):
    return fpld.DataLoader(location=location)
