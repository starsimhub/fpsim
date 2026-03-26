"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 4.79) """
    pars = {}
    pars['exposure_factor'] = 0.8873
    pars['prob_use_intercept'] = -1.5857
    pars['prob_use_trend_par'] = 0.0005
    pars['fecundity_low'] = 0.8239
    pars['fecundity_high'] = 2.2813
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.1803, 0.3574, 0.6376, 1.4622, 0.4466, 1.4889, 0.9518, 1.7930, 0.7145, 1.3235, 0.4917, 0.0212]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9006, 0.5405, 0.3950, 0.0514, 0.1169, 0.1966, 0.0303]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9273, 0.8546, 0.7818, 0.7091, 0.6364, 0.5637, 0.4910, 0.4183])
    }
    return pars


def dataloader(location='niger'):
    return fpld.DataLoader(location=location)
