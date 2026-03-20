"""
Set the parameters for Nigeria Kano.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 0.85000
    pars['prob_use_intercept'] = 0.25
    pars['prob_use_trend_par'] = 0.00
    pars['fecundity_low'] = 0.5931
    pars['fecundity_high'] = 2.0963
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.4000, 0.3600, 0.6300, 0.4500, 0.8500, 1.0500, 1.2500, 1.5000, 1.5500, 1.2500, 0.7500, 0.4500]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5715, 0.3088, 0.4949, 0.3469, 0.1863, 0.1165, 0.0586]])
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
    return pars


def dataloader(location='nigeria_kano'):
    return fpld.DataLoader(location=location)
