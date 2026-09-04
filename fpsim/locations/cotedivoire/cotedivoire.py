"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters (mismatch: 3.76) """
    pars = {}
    pars['exposure_factor'] = 0.7970
    pars['prob_use_intercept'] = -1.3772
    pars['prob_use_trend_par'] = 0.0381
    pars['fecundity_low'] = 0.8333
    pars['fecundity_high'] = 1.8155
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                      [1.0000, 0.3834, 0.1220, 0.2280, 1.0743, 0.6357, 0.7075, 2.1815, 0.9376, 1.3849, 0.5389, 0.4066, 0.0552]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9300, 0.7063, 0.1584, 0.2007, 0.2554, 0.1298, 0.0241]])
    pars['spacing_pref'] = {
        'preference': np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9672, 0.9343, 0.9015, 0.8687, 0.8359, 0.8030, 0.7702, 0.7374, 0.7046, 0.6717, 0.6389])
    }
    pars['method_weights'] = np.array([8, 4, 4, 20, 2, 2, 2, 3, 5])
    pars['dur_postpartum'] = 18
    return pars


def dataloader(location='cotedivoire'):
    return fpld.DataLoader(location=location)
