"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 3.57) """
    pars = {}
    pars['exposure_factor'] = 0.7864
    pars['prob_use_intercept'] = -1.5564
    pars['prob_use_trend_par'] = 0.0489
    pars['fecundity_low'] = 0.8819
    pars['fecundity_high'] = 1.8301
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.3548, 0.0419, 0.1592, 1.7080, 1.3984, 1.7495, 1.1565, 0.9603, 1.4821, 1.2862, 0.1208, 0.4354]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9649, 0.6006, 0.2696, 0.0263, 0.1728, 0.0227, 0.0839]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.7956, 0.5911, 0.3867])
    }
    pars['method_weights'] = np.array([0.05, 0.05, 150, 1, 0.05, 0.5, 0.05, 0.2, 1])
    pars['dur_postpartum'] = 18
    return pars


def dataloader(location='ethiopia'):
    return fpld.DataLoader(location=location)
