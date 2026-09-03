"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 4.80) """
    pars = {}
    pars['exposure_factor'] = 0.5691
    pars['prob_use_intercept'] = -2.9265
    pars['prob_use_trend_par'] = 0.0462
    pars['fecundity_low'] = 0.5880
    pars['fecundity_high'] = 2.3660
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.4367, 0.0670, 0.5746, 1.0448, 2.0639, 1.1694, 0.5192, 1.5062, 1.8515, 0.6090, 0.2209, 0.4469]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9011, 0.5163, 0.1980, 0.2699, 0.2485, 0.0895, 0.0652]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9275, 0.8551, 0.7826, 0.7101, 0.6377, 0.5652, 0.4927, 0.4202, 0.3478])
    }
    pars['method_weights'] = np.array([0.6, 0.4, 0.4, 0.9, 10, 1.5, 1, 10, 8])
    pars['dur_postpartum'] = 18
    return pars


def dataloader(location='nigeria_lagos'):
    return fpld.DataLoader(location=location)
