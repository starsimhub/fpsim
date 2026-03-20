"""
Set the parameters for Nigeria Lagos.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 1
    pars['prob_use_intercept'] = 0.0
    pars['prob_use_trend_par'] = -0.14
    pars['fecundity_low'] = 0.6041
    pars['fecundity_high'] = 1.8852
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.1000, 0.3400, 0.7000, 0.8000, 1.0000, 1.2000, 1.3000, 1.0000, 1.5000, 1.2000, 0.7000, 0.3500]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.3783, 0.4045, 0.5020, 0.2438, 0.1445, 0.1725, 0.0263]])
    pars['method_weights'] = np.array([0.6, 0.4, 0.4, 0.9, 10, 1.5, 1, 10, 8])
    pars['dur_postpartum'] = 18
    spacing_pref_array = np.ones(19, dtype=float)
    spacing_pref_array[:2] =  0.5    # 0-6 months — suppress very short intervals
    spacing_pref_array[2:5] = 0.7    # 6-15 months — suppress to reduce 12-24mo births
    spacing_pref_array[5:13] = 1.0   # 15-39 months — normal (maps to 24-48mo births)
    spacing_pref_array[13:] = 0.5   # 39+ months — suppress >48mo births

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars


def dataloader(location='nigeria_lagos'):
    return fpld.DataLoader(location=location)
