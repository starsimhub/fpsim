"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 4.10) """
    pars = {}
    pars['exposure_factor'] = 0.5014
    pars['prob_use_intercept'] = -2.3977
    pars['prob_use_trend_par'] = 0.0394
    pars['fecundity_low'] = 0.8021
    pars['fecundity_high'] = 2.2159
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.0123, 0.4958, 0.4754, 1.3013, 0.4910, 0.9354, 2.4407, 2.4752, 1.6760, 0.3966, 0.7695, 0.1951]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.7835, 0.2006, 0.2132, 0.0894, 0.2841, 0.0996, 0.0050]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.8432, 0.6863])
    }
    return pars


def dataloader(location='senegal'):
    return fpld.DataLoader(location=location)
