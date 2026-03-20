"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 0.8700
    pars['prob_use_intercept'] = -1.2438
    pars['prob_use_trend_par'] = 0.0050
    pars['fecundity_low'] = 0.8035
    pars['fecundity_high'] = 1.1014
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.4000, 0.0300, 0.1900, 0.7000, 0.5500, 1.2500, 1.400, 1.4500, 1.900, 1.800, 0.9000, 0.4500]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5543, 0.7648, 0.1689, 0.0231, 0.2988, 0.0110, 0.0238]])
    pars['method_weights'] = np.array([1, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 0.8, 1])
    pars['dur_postpartum'] = 15
    spacing_pref_array = np.ones(17, dtype=float)
    spacing_pref_array[:2] =  0.3    # 0-6 months pp — suppress very short intervals
    spacing_pref_array[2:5] = 0.45    # 6-15 months pp — suppress to reduce 12-24mo births
    spacing_pref_array[5:12] = 0.9   # 15-36 months pp — normal (maps to 24-48mo births)
    spacing_pref_array[12:] = 0.1    # 36+ months pp — suppress >48mo births

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars


def dataloader(location='senegal'):
    return fpld.DataLoader(location=location)
