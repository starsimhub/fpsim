"""
Set the parameters for Nigeria Lagos.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    pars = {}
    pars['fecundity_low'] = 0.5
    pars['fecundity_high'] = 1.0
    pars['exposure_factor'] = 3
    pars['prob_use_year'] = 2020
    pars['prob_use_trend_par'] = 0.0001
    pars['prob_use_intercept'] = -3.4
    pars['method_weights'] = np.array([0.2, 0.2, 0.1, 0.2, 30, 0.5, 1, 50, 5])
    pars['dur_postpartum'] = 23

    spacing_pref_array = np.ones(18, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:3] =  1
    spacing_pref_array[3:6] = 0.5
    spacing_pref_array[6:9] = 0.8
    spacing_pref_array[9:] =  2
    
    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30,  35, 40, 45,  50],
                                        [1, 1, 1,  2,    2 ,0.3,0.8,1.3,0.4,0.6,0.3,0.4, 0.5]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars


def dataloader(location='nigeria_lagos'):
    return fpld.DataLoader(location=location)