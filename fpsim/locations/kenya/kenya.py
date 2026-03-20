"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}
    pars['exposure_factor'] = 0.9000
    pars['prob_use_intercept'] = -0.4
    pars['prob_use_trend_par'] = -0.02
    pars['fecundity_low'] = 0.7778
    pars['fecundity_high'] = 1.8333
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.4000, 0.3600, 0.6400, 0.7000, 1.3000, 1.3000, 1.4000, 1.8000, 1.9000, 1.4000, 0.8000, 0.4000]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.3388, 0.6675, 0.4723, 0.0333, 0.2638, 0.0251, 0.0130]])
    pars['method_weights'] = np.array([0.8, 0.3, 3.5, 1, 1, 1, 1.2, 0.3, 3])
    pars['dur_postpartum'] = 18
    spacing_pref_array = np.ones(19, dtype=float)
    spacing_pref_array[:4] =  0.3    # 0-12 months pp — suppress short intervals
    spacing_pref_array[4:8] = 0.05   # 12-24 months pp — strong suppress (model way over)
    spacing_pref_array[8:16] = 1.5   # 24-48 months pp — boost preferred spacing
    spacing_pref_array[16:] = 0.15   # >48 months pp — suppress

    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    return pars


def dataloader(location='kenya'):
    return fpld.DataLoader(location=location)
