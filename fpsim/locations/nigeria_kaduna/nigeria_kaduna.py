"""
Set the parameters for Nigeria Kaduna.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    pars = {}
    pars['exposure_factor'] = 0.7125
    pars['prob_use_intercept'] = -1.7094
    pars['prob_use_trend_par'] = 0.0138
    pars['fecundity_low'] = 0.5559
    pars['fecundity_high'] = 1.5778
    pars['method_weights'] = np.array([0.1, 0.2, 5, 25, 5, 0.01, 0.1, 10, 18])
    pars['dur_postpartum'] = 18

    spacing_pref_array = np.ones(19, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:4] =  1.0   # 0-12 months
    spacing_pref_array[4:8] = 1.0   # 12-24 months
    spacing_pref_array[8:13] = 1.0  # 24-39 months
    spacing_pref_array[13:] = 0.15  # 39-54+ months — suppress (data: 12% >48)

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0, 0.1227, 0.2044, 0.9651, 1.3271, 1.0578, 0.8303, 1.2483, 1.753, 1.7814, 1.4106, 0.7539, 0.2443]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars


def dataloader(location='nigeria_kaduna'):
    return fpld.DataLoader(location=location)
