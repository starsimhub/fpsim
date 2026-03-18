"""
Set the parameters for Nigeria Kano.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    pars = {}
    pars['exposure_factor'] = 0.9467
    pars['prob_use_intercept'] = -0.2254
    pars['prob_use_trend_par'] = 0.023
    pars['fecundity_low'] = 0.804
    pars['fecundity_high'] = 1.6524
    pars['method_weights'] = np.array([0.75, 1, 3, 1.5, 0.5, 0.3, 8, 2, 3])
    pars['dur_postpartum'] = 18

    spacing_pref_array = np.ones(19, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:4] =  1.0   # 0-12 months
    spacing_pref_array[4:8] = 1.0   # 12-24 months
    spacing_pref_array[8:13] = 1.0  # 24-39 months
    spacing_pref_array[13:] = 0.1   # 39-54+ months — suppress (data: only 9% >48)

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0, 0.1574, 0.1013, 0.1763, 0.8715, 1.8229, 1.7624, 0.6649, 1.4302, 1.8135, 0.194, 0.1178, 0.4176]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars


def dataloader(location='nigeria_kano'):
    return fpld.DataLoader(location=location)
