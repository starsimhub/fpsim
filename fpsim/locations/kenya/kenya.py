"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 1.179
    pars['prob_use_intercept'] = -2.0024
    pars['prob_use_trend_par'] = 0.0496
    pars['fecundity_low'] = 0.7632
    pars['fecundity_high'] = 2.4112
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0, 0.4035, 0.0312, 0.2581, 0.6449, 0.5961, 0.9335, 0.8552, 2.3911, 0.9851, 1.1123, 0.9312, 0.0341]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])
    pars['method_weights'] = np.array([0.8, 0.3, 3.5, 1, 1, 1, 1.2, 0.3, 3])
    pars['dur_postpartum'] = 18

    # 19 bins (0-54 months, 3-month intervals). Suppress >48 to reduce long birth intervals.
    spacing_pref_array = np.ones(19, dtype=float)
    spacing_pref_array[:4] =  0.5   # 0-12 months
    spacing_pref_array[4:8] = 0.01   # 12-24 months — suppress (model over)
    spacing_pref_array[8:16] = 2  # 24-48 months
    spacing_pref_array[16:] = 0.2   # >48 months — suppress

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars


def dataloader(location='kenya'):
    return fpld.DataLoader(location=location)
