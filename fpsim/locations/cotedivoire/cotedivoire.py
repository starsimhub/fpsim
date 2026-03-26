"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 4.15) """
    pars = {}
    pars['exposure_factor'] = 0.7132
    pars['prob_use_intercept'] = -1.5681
    pars['prob_use_trend_par'] = 0.0279
    pars['fecundity_low'] = 0.7938
    pars['fecundity_high'] = 1.3277
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.2397, 0.3609, 0.2406, 1.3727, 0.3985, 1.1786, 1.2907, 1.7837, 0.5751, 1.3807, 0.4991, 0.4638]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.7670, 0.3396, 0.5252, 0.0555, 0.1520, 0.0822, 0.0442]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.8660, 0.7321, 0.5981, 0.4641, 0.3302])
    }
    return pars


def dataloader(location='cotedivoire'):
    return fpld.DataLoader(location=location)
