"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 3.61) """
    pars = {}
    pars['exposure_factor'] = 0.6764
    pars['prob_use_intercept'] = -2.2705
    pars['prob_use_trend_par'] = 0.0350
    pars['fecundity_low'] = 0.7465
    pars['fecundity_high'] = 1.8148
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.4162, 0.2118, 0.1729, 0.7320, 1.2515, 1.1850, 1.4270, 1.5731, 1.5892, 1.2648, 0.7770, 0.3831]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.7429, 0.4523, 0.3250, 0.0760, 0.2530, 0.1082, 0.0343]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9222, 0.8445, 0.7667, 0.6889, 0.6112, 0.5334, 0.4556, 0.3779, 0.3001, 0.2223, 0.1446])
    }
    pars['method_weights'] = np.array([0.44, 2.3, 10, 3, 2, 2, 2, 0.01, 0.01])
    pars['dur_postpartum'] = 18
    return pars


def dataloader(location='pakistan_sindh'):
    return fpld.DataLoader(location=location)
