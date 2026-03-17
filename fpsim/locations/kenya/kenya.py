"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 0.97
    pars['prob_use_intercept'] = -0.9
    pars['prob_use_trend_par'] = -0.01
    pars['fecundity_low'] = 0.85
    pars['fecundity_high'] = 2.5
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0, 0.4706, 0.2, 0.65, 0.8, 0.9, 0.95, 1.2, 1.15, 1.5352, 0.9445, 0.207, 0.1546]])
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
