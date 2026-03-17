"""
Set the parameters for Nigeria Lagos.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    pars = {}
    pars['exposure_factor'] = 0.9250
    pars['prob_use_intercept'] = -.3
    pars['prob_use_trend_par'] = -0.135
    pars['fecundity_low'] = 0.7165
    pars['fecundity_high'] = 1.4219
    pars['method_weights'] = np.array([0.6, 0.4, 0.4, 0.9, 10, 1.7, 1, 10, 6])
    pars['dur_postpartum'] = 18

    spacing_pref_array = np.ones(19, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:3] =  1
    spacing_pref_array[3:6] = 0.5
    spacing_pref_array[6:9] = 0.8
    spacing_pref_array[9:13] = 1.0  # 27-39 months
    spacing_pref_array[13:] =  0.3  # 39-54+ months — suppress

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0, 0.2655, 0.0927, 0.4971, 1.0241, 1.2026, 0.9, 1.3, 1.2, 1.0, 0.6, 0.4, 0.15]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars


def dataloader(location='nigeria_lagos'):
    return fpld.DataLoader(location=location)
