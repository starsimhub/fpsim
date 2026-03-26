"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 4.33) """
    pars = {}
    pars['exposure_factor'] = 0.8091
    pars['prob_use_intercept'] = -1.6412
    pars['prob_use_trend_par'] = 0.0496
    pars['fecundity_low'] = 0.8172
    pars['fecundity_high'] = 1.2190
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.4673, 0.2921, 0.2989, 0.8648, 1.2860, 2.0729, 1.0649, 0.9525, 1.7373, 0.7432, 0.2116, 0.4565]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.3756, 0.2023, 0.2719, 0.1126, 0.2724, 0.1041, 0.0887]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9000, 0.8000, 0.7000, 0.6000, 0.5000])
    }
    return pars


def dataloader(location='ethiopia'):
    return fpld.DataLoader(location=location)
