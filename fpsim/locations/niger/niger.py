"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 3.88) """
    pars = {}
    pars['exposure_factor'] = 0.8999
    pars['prob_use_intercept'] = -1.9358
    pars['prob_use_trend_par'] = 0.0356
    pars['fecundity_low'] = 0.5429
    pars['fecundity_high'] = 1.1194
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.2204, 0.1198, 0.7899, 0.6227, 0.9572, 1.2558, 1.0130, 1.4562, 1.2273, 1.3061, 0.4147, 0.2101]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9842, 0.6869, 0.5148, 0.1039, 0.0557, 0.1343, 0.0061]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.8621, 0.7241, 0.5862, 0.4482, 0.3103])
    }
    pars['method_weights'] = np.array([0.1, 0.1, .3, 0.5, 0.5, 0.4, 0.002, 0.4, 150])
    pars['dur_postpartum'] = 18
    return pars


def dataloader(location='niger'):
    return fpld.DataLoader(location=location)
