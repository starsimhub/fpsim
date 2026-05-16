"""
Set the parameters for Niger.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 1.0
    pars['prob_use_intercept'] = -0.9
    pars['prob_use_trend_par'] = -0.0152
    pars['fecundity_low'] = 0.5989
    pars['fecundity_high'] = 2.3477
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.3300, 0.4500, 0.6100, 1.0000, 1.2000, 1.3000, 1.4000, 1.5000, 1.4000, 1.0000, 0.6500, 0.3500]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.6987, 0.5770, 0.4872, 0.3658, 0.2544, 0.0214, 0.0748]])
    pars['method_weights'] = np.array([0.1, 0.1, .3, 0.5, 0.5, 0.4, 0.002, 0.4, 150])
    pars['dur_postpartum'] = 18
    spacing_pref_array = np.ones(19, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:4] =  0.9   # 0-12 months
    spacing_pref_array[4:8] = 0.9   # 12-24 months
    spacing_pref_array[8:13] = 1.1  # 24-39 months
    spacing_pref_array[13:] = 0.1   # 39-54+ months — suppress (data: only 9% >48)

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars


def dataloader(location='niger'):
    return fpld.DataLoader(location=location)
