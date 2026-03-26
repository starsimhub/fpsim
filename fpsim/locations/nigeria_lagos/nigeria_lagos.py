"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 5.87) """
    pars = {}
    pars['exposure_factor'] = 0.5006
    pars['prob_use_intercept'] = -2.8774
    pars['prob_use_trend_par'] = -0.0140
    pars['fecundity_low'] = 0.8156
    pars['fecundity_high'] = 1.1296
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.0996, 0.4939, 0.5096, 0.9153, 1.2144, 0.7362, 0.5352, 2.6467, 1.1547, 0.7532, 0.5379, 0.0715]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.4880, 0.7188, 0.4333, 0.2994, 0.1473, 0.1091, 0.0887]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9508, 0.9016, 0.8524, 0.8032, 0.7540, 0.7048, 0.6556, 0.6064])
    }
    return pars


def dataloader(location='nigeria_lagos'):
    return fpld.DataLoader(location=location)
