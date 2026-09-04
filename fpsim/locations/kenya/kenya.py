"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 3.44) """
    pars = {}
    pars['exposure_factor'] = 0.8066
    pars['prob_use_intercept'] = -1.8269
    pars['prob_use_trend_par'] = 0.0499
    pars['fecundity_low'] = 0.7444
    pars['fecundity_high'] = 1.6498
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.3939, 0.4225, 0.1188, 1.6794, 1.4618, 1.0967, 0.8179, 2.5009, 1.0118, 1.0756, 0.9667, 0.1535]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.7215, 0.2341, 0.0924, 0.0860, 0.1926, 0.1398, 0.0486]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9027, 0.8054, 0.7081, 0.6109, 0.5136, 0.4163])
    }
    pars['method_weights'] = np.array([0.8, 0.3, 3.5, 1, 1, 1, 1.2, 0.3, 3])
    pars['dur_postpartum'] = 18
    return pars


def dataloader(location='kenya'):
    return fpld.DataLoader(location=location)
