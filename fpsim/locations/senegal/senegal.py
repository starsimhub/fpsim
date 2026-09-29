"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 3.46) """
    pars = {}
    pars['exposure_factor'] = 0.6391
    pars['prob_use_intercept'] = -2.2957
    pars['prob_use_trend_par'] = 0.0457
    pars['fecundity_low'] = 0.5792
    pars['fecundity_high'] = 1.3420
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.1204, 0.0557, 0.1193, 1.0122, 2.1370, 2.1849, 1.1103, 1.8062, 1.3417, 1.3308, 0.3158, 0.4621]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.4978, 0.2611, 0.5468, 0.2652, 0.0851, 0.1545, 0.0821]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.8621, 0.7242, 0.5863, 0.4484, 0.3104])
    }
    pars['method_weights'] = np.array([1, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 0.8, 1])
    pars['dur_postpartum'] = 15
    return pars


def dataloader(location='senegal'):
    return fpld.DataLoader(location=location)
