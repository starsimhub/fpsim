"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 3.50) """
    pars = {}
    pars['exposure_factor'] = 0.5382
    pars['prob_use_intercept'] = -2.2959
    pars['prob_use_trend_par'] = 0.0429
    pars['fecundity_low'] = 0.5237
    pars['fecundity_high'] = 2.4048
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.2574, 0.4966, 0.3587, 1.9006, 1.6208, 1.1313, 1.7979, 2.3366, 1.8868, 0.6719, 0.0974, 0.4532]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9175, 0.4272, 0.2851, 0.2592, 0.1574, 0.0382, 0.0623]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9581, 0.9162, 0.8743, 0.8324, 0.7905, 0.7486, 0.7067, 0.6648])
    }
    return pars


def dataloader(location='kenya'):
    return fpld.DataLoader(location=location)
