"""
Set the parameters for Nigeria Kaduna.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 0.800
    pars['prob_use_intercept'] = -1.2000
    pars['prob_use_trend_par'] = 0.0
    pars['fecundity_low'] = 0.7520
    pars['fecundity_high'] = 1.3364
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.2900, 0.2600, 0.5800, 0.5000, 0.8000, 1.0500, 1.2500, 1.2000, 1.7000, 1.4000, 0.8000, 0.5000]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.3141, 0.4816, 0.2244, 0.3646, 0.1998, 0.0390, 0.0110]])
    pars['method_weights'] = np.array([0.1, 0.2, 5, 25, 5, 0.01, 0.1, 10, 18])
    pars['dur_postpartum'] = 18
    spacing_pref_array = np.ones(19, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:4] =  0.8   # 0-12 months
    spacing_pref_array[4:8] = 0.8   # 12-24 months
    spacing_pref_array[8:13] = 1.2  # 24-39 months
    spacing_pref_array[13:] = 0.15  # 39-54+ months — suppress (data: 12% >48)

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars


def dataloader(location='nigeria_kaduna'):
    return fpld.DataLoader(location=location)
