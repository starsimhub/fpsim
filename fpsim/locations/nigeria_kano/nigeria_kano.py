"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 4.41) """
    pars = {}
    pars['exposure_factor'] = 0.6864
    pars['prob_use_intercept'] = -0.1233
    pars['prob_use_trend_par'] = 0.0411
    pars['fecundity_low'] = 0.6076
    pars['fecundity_high'] = 1.8138
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.2952, 0.2026, 0.6005, 0.4100, 1.8059, 1.1097, 0.9556, 1.9052, 0.9882, 1.3647, 0.8703, 0.2729]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.6776, 0.5768, 0.5221, 0.1509, 0.0906, 0.1891, 0.0914]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9156, 0.8312, 0.7469, 0.6625, 0.5781])
    }
    pars['method_weights'] = np.array([0.75, 1, 3, 1.5, 0.5, 0.3, 8, 2, 3])
    pars['dur_postpartum'] = 18
    return pars


def dataloader(location='nigeria_kano'):
    return fpld.DataLoader(location=location)
