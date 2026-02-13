'''
Handle sim parameters
'''

import sciris as sc
import starsim as ss
import fpsim as fp

__all__ = ['SimPars', 'FPPars', 'make_sim_pars', 'make_fp_pars', 'par_keys', 'sim_par_keys', 'all_pars', 'mergepars']


# %% Parameter creation functions

class SimPars(ss.SimPars):
    """
    Dictionary with all parameters used within an FPsim.
    All parameters that don't vary across geographies are defined explicitly here.
    Keys for all location-specific parameters are also defined here with None values.
    """
    def __init__(self, **kwargs):

        # Initialize the parent class
        super().__init__()

        # Basic parameters
        self.n_agents = 1_000  # Number of agents
        self.start = 2000   # Start year of simulation
        self.stop = 2020   # End year of simulation
        self.dt = 1/12      # The simulation timestep in 'unit's
        self.unit = 'year'   # The unit of time for the simulation
        self.rand_seed = 1      # Random seed
        self.verbose = 1/12   # Verbosity level
        self.use_aging = True   # Whether to age the population
        self.location = None  # Default location
        self.test = False

        # Update with any supplied parameter values and generate things that need to be generated
        self.update(kwargs)
        return


def make_sim_pars(**kwargs):
    """ Shortcut for making a new instance of SimPars """
    return SimPars(**kwargs)


class FPPars(ss.Pars):
    def __init__(self, location=None, **kwargs):
        super().__init__()

        # Settings - what aspects are being modeled - TODO, remove
        self.use_partnership = 0

        # Age limits (in years)
        self.method_age = 15
        self.age_limit_fecundity = 50
        self.max_age = 99

        # Durations (in months)
        self.end_first_tri = 3      # Months
        self.dur_pregnancy = ss.uniform(low=ss.months(9), high=ss.months(9))
        self.dur_breastfeeding = ss.normal(loc=ss.months(24), scale=ss.months(6))
        self.dur_postpartum = None      # Updated by data, do not modify
        self.max_lam_dur = 5            # Duration of lactational amenorrhea (months)
        self.short_int = ss.months(24)  # Duration of a short birth interval between live births (months)

        # Parameters related to the likelihood of conception
        self.LAM_efficacy = 0.98   # From Cochrane review: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6823189/
        self.primary_infertility = 0.05

        # Parameters typically tuned during calibration
        self.maternal_mortality_factor = 1
        self.fecundity_low = 0.7
        self.fecundity_high = 1.1  # Personal fecundity distribution
        self.exposure_factor = 1  # Overall exposure factor, to be calibrated
        self.exposure_age = dict(age    =[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                 rel_exp=[1, 1,  1,    1,  1,  1,  1,  1,  1,  1,  1,  1,  1])
        self.exposure_parity = dict(parity =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                    rel_exp=[1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01])

        ###################################
        # Context-specific data-derived parameters, all need to be loaded from data
        ###################################
        self.abortion_prob = None
        self.twins_prob = None
        self.maternal_mortality = None
        self.infant_mortality = None
        self.miscarriage_rates = None
        self.stillbirth_rate = None
        self.age_fecundity = None
        self.fecundity_ratio_nullip = None
        self.lactational_amenorrhea = None
        self.sexual_activity = None
        self.sexual_activity_pp = None
        self.debut_age = None
        self.spacing_pref = None
        self.age_partnership = None
        self.region = None
        self.regional = None

        self.update(kwargs)

        if location is not None:
            self.update_location(location=location)
        else:
            # Process parameters even when no location is specified
            self.process_parameters()

        return

    def update_location(self, location=None):
        """
        Update the location-specific FP parameters
        """
        location_module = fp.get_dataloader(location)
        location_pars = location_module.make_fp_pars()
        self.update(**location_pars)
        self.process_parameters()
        return

    def process_parameters(self):
        """
        Process parameters after all updates are complete.
        Convert fecundity_low/fecundity_high to ss.uniform distribution.
        """
        # Convert fecundity bounds to distribution
        if hasattr(self, 'fecundity_low') and hasattr(self, 'fecundity_high'):
            self.fecundity = ss.uniform(low=self.fecundity_low, high=self.fecundity_high)
        elif not hasattr(self, 'fecundity'):
            # Fallback to default if no fecundity parameters are set
            self.fecundity = ss.uniform(low=0.7, high=1.1)
        return


def make_fp_pars(location=None):
    """ Shortcut for making a new instance of FPPars """
    return FPPars(location=location)


def mergepars(*args, _copy=False, **kwargs):
    """
    Merge all parameter dictionaries into a single dictionary with nested merging.
    This is used to initialize the SimPars class with all relevant parameters.

    Unlike sc.mergedicts, this function recursively merges nested dictionaries instead
    of replacing them entirely. This allows partial dictionary specifications in
    calibration parameters (e.g., only specifying preference values in spacing_pref
    while preserving interval, n_bins, months from defaults).

    Args:
        _copy (bool): whether to deep copy the input dictionaries (default False, same as sc.mergedicts)
        *args: dictionaries to merge
        **kwargs: additional parameters (for compatibility with sc.mergedicts)
    """
    # Convert any Pars objects to dicts while preserving types like sc.objdict
    if _copy:
        dicts = [sc.dcp(arg) if arg is not None else {} for arg in args if arg is not None]
    else:
        dicts = [dict(arg) if arg is not None else {} for arg in args if arg is not None]

    if len(dicts) < 2:
        # Single dict or empty - use standard merging
        return sc.mergedicts(*dicts, _copy=_copy, **kwargs) if dicts else {}

    # Start with the first dictionary
    result = sc.dcp(dicts[0]) if _copy else dicts[0]

    # Recursively merge each subsequent dictionary
    for next_dict in dicts[1:]:
        for key, value in next_dict.items():
            if (key in result and
                isinstance(value, dict) and
                isinstance(result[key], dict)):
                # Both are dicts - recursively merge
                result[key] = mergepars(result[key], value, _copy=_copy, **kwargs)
            else:
                # Not both dicts - use the new value
                result[key] = sc.dcp(value) if _copy else value

    return result


# Shortcut for accessing default keys
par_keys = make_fp_pars().keys()
sim_par_keys = make_sim_pars().keys()


# ALL PARS
def all_pars(location=None):
    """
    Return a dictionary with all parameters used within an FPsim.
    This includes both simulation parameters and family planning parameters.
    """
    sim_pars = make_sim_pars()
    fp_pars = make_fp_pars()
    contra_pars = fp.ContraPars()
    edu_pars = fp.EduPars()
    death_pars = fp.DeathPars()
    people_pars = fp.PeoplePars()
    mergedpars = mergepars(sim_pars, fp_pars, contra_pars, edu_pars, death_pars, people_pars)
    mergedpars['location'] = location
    # Load location-specific parameters if a location is provided
    if location is not None:
        data_pars = fp.DataLoader(location=location).load(return_data=True)
        for modpars in data_pars.values():
            mergedpars.update(modpars)
    return mergedpars

