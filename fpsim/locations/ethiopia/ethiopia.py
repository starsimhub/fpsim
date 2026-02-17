"""
Set the parameters for FPsim, specifically for Ethiopia.
"""
import numpy as np
from pathlib import Path
from fpsim import defaults as fpd
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    """ Make a dictionary of location-specific parameters """
    pars = {}

    # FP parameters
    pars['exposure_age'] = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars


# %% Make and validate parameters
def dataloader(location='ethiopia'):
    return fpld.DataLoader(location=location)
