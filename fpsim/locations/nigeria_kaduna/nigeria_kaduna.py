"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 4.21) """
    pars = {}
    pars['exposure_factor'] = 0.5120
    pars['prob_use_intercept'] = -1.4378
    pars['prob_use_trend_par'] = 0.0360
    pars['fecundity_low'] = 0.8686
    pars['fecundity_high'] = 1.1469
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.4889, 0.2977, 0.6769, 0.8114, 1.2165, 1.4691, 1.0280, 2.5040, 0.7251, 1.4297, 0.7276, 0.3814]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.6450, 0.4593, 0.0587, 0.3468, 0.2540, 0.0246, 0.0316]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9150, 0.8300, 0.7450, 0.6600, 0.5751])
    }
    pars['method_weights'] = np.array([0.1, 0.2, 5, 25, 5, 0.01, 0.1, 10, 18])
    pars['dur_postpartum'] = 18
    return pars


def dataloader(location='nigeria_kaduna'):
    return fpld.DataLoader(location=location)
