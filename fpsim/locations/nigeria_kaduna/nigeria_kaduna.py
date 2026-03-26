"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 4.93) """
    pars = {}
    pars['exposure_factor'] = 0.5498
    pars['prob_use_intercept'] = -1.1521
    pars['prob_use_trend_par'] = 0.0341
    pars['fecundity_low'] = 0.8885
    pars['fecundity_high'] = 1.0921
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.2635, 0.4746, 0.5635, 0.6925, 1.0049, 1.5918, 1.9706, 2.4885, 0.2524, 0.5178, 0.3040, 0.3797]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.4067, 0.3524, 0.4418, 0.0307, 0.0381, 0.0650, 0.0038]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9125, 0.8250, 0.7376, 0.6501, 0.5626, 0.4751, 0.3876, 0.3002])
    }
    return pars


def dataloader(location='nigeria_kaduna'):
    return fpld.DataLoader(location=location)
