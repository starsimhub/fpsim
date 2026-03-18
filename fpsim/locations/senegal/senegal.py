"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 0.8487
    pars['prob_use_intercept'] = -2.7443
    pars['prob_use_trend_par'] = 0.0437
    pars['fecundity_low'] = 0.5889
    pars['fecundity_high'] = 2.2754
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0, 0.4975, 0.0179, 0.531, 0.5407, 1.0439, 1.3717, 1.1333, 2.1776, 1.2859, 1.0328, 0.7105, 0.0147]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])
    pars['method_weights'] = np.array([1, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 0.8, 1])
    pars['dur_postpartum'] = 15

    # 17 bins: 0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48 months
    # Weights derived from DHS data, declining at >36 months to reduce >48 spacing
    spacing_pref_array = np.array([1.0, 1.0, 1.0, 1.0,   # 0-12 months
                                   1.0, 1.0,               # 12-18 months
                                   1.0, 1.0,               # 18-24 months
                                   1.0, 1.0, 1.0, 1.0,     # 24-36 months
                                   0.8, 0.5, 0.3, 0.15,    # 36-48 months — gradual suppress
                                   0.05])                    # 48+ months — suppress

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars


def dataloader(location='senegal'):
    return fpld.DataLoader(location=location)
