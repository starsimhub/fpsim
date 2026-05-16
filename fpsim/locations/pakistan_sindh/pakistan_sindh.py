"""
Set the parameters for Pakistan Sindh.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 0.9000
    pars['prob_use_intercept'] = -0.200
    pars['prob_use_trend_par'] = -0.150
    pars['fecundity_low'] = 0.6339
    pars['fecundity_high'] = 2.3721
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.1900, 0.2900, 0.1300, 0.8000, 1.2000, 1.0000, 1.0000, 1.4000, 1.5000, 1.1000, 0.6500, 0.3500]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9344, 0.1332, 0.2479, 0.3279, 0.2390, 0.0553, 0.0551]])
    pars['method_weights'] = np.array([0.44, 2.3, 10, 3, 2, 2, 2, 0.01, 0.01])
    pars['dur_postpartum'] = 18
    spacing_pref_array = np.ones(19, dtype=float)
    spacing_pref_array[:2] =  0.5    # 0-6 months pp — suppress very short intervals
    spacing_pref_array[2:5] = 0.7    # 6-15 months pp — mild suppress to reduce 12-24mo births
    spacing_pref_array[5:13] = 1.0   # 15-39 months pp — normal (maps to 24-48mo births)
    spacing_pref_array[13:] = 0.15   # 39+ months pp — suppress >48mo births

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars


def dataloader(location='pakistan_sindh'):
    return fpld.DataLoader(location=location)
