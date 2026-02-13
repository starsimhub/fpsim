"""
Defines the People class
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc
import fpsim as fp
from . import demographics as fpdmg
import starsim as ss

# Specify all externally visible things this file defines
__all__ = ['People', 'PeoplePars', 'make_people_pars']


# %% Define classes
class PeoplePars(ss.Pars):
    """
    Parameters for the deaths module.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.age_pyramid = None
        self.urban_prop = None
        self.wealth_quintile = None
        self.update(kwargs)
        return


def make_people_pars():
    """ Shortcut for making a new instance of ContraPars """
    return PeoplePars()


class People(ss.People):
    """
    Class for all the people in the simulation.
    Inherits from starsim.People, and adds a few additional states and methods specific to family planning simulations.
    Args:
        n_agents (int): Number of agents to create. If None, will be set later.
        pars (dict): Dictionary of parameters used to initialize states
        **kwargs: Additional keyword arguments passed to the parent class initializer.
    """

    def __init__(self, n_agents=None, pars=None, **kwargs):

        # Person defaults
        self.person_defaults = [
            ss.BoolState('partnered', default=False),  # Will remain at these values if use_partnership is False
            ss.FloatArr('partnership_age', default=-1),  # Will remain at these values if use_partnership is False
            ss.BoolState('urban', default=ss.bernoulli(p=0.5)),  # Urban/rural
            ss.FloatArr('wealthquintile', default=ss.choice(a=5)),  # Wealth quintile
        ]

        # Process age/sex data
        age_data = None
        f_frac = 0.5
        if pars.get('age_pyramid') is not None:
            ages = pars['age_pyramid'][:, 0]
            age_counts = pars['age_pyramid'][:, 1] + pars['age_pyramid'][:, 2]
            age_data = np.array([ages, age_counts]).T
            f_frac = pars['age_pyramid'][:, 2].sum() / pars['age_pyramid'][:, 1:3].sum()

        # Initialization
        super().__init__(n_agents, age_data, extra_states=self.person_defaults, **kwargs)
        self.female.default.set(p=f_frac)
        self.set_urban(pars)
        self.set_wealthquintile(pars)

        return

    def init_vals(self, uids=None):
        super().init_vals()

        sim = self.sim
        fp_pars = sim.pars.fp

        if uids is None:
            uids = self.alive.uids

        # Partnership
        if fp_pars['use_partnership']:
            fpdmg.init_partnership_states(uids)

        # Store keys
        self._keys = [s.name for s in self.states.values()]

        return

    @property
    def ever_used_contra(self):
        return self.sim.connectors.fp.ever_used_contra  # TODO, fix

    @property
    def parity(self):
        return self.sim.connectors.fp.parity  # TODO, fix

    def set_urban(self, pars):
        """ Get initial distribution of urban """
        if pars.get('urban_prop') is None:
            return
        urban_prop = pars['urban_prop']
        self.urban.default.set(p=urban_prop)
        return

    def set_wealthquintile(self, pars):
        if pars.get('wealth_quintile') is None:
            return
        wq = pars['wealth_quintile']
        wq_quintiles = wq['quintile']
        wq_probs = wq['percent']
        self.wealthquintile.default.set(p=wq_probs, a=wq_quintiles)
        return

    def update_age_bin_totals(self, uids):
        """
        Count how many total live women in each 5-year age bin 10-50, for tabulating ASFR
        """
        if uids is None:
            uids = self.alive.uids

        for key, (age_low, age_high) in fp.age_bin_map.items():
            this_age_bin = uids[(self.age[uids] >= age_low) & (self.age[uids] < age_high)]
            self.sim.results[f'total_women_{key}'][self.sim.ti] += len(this_age_bin)

        return

    def update_results(self):
        """Calculate and return the results for this specific time step"""
        # TODO: this is commented out because the base class calculated deaths in a way that overwrites the
        # FPsim way. In FPsim, deaths are removed at the beginning of the timestep not the end, so we can't
        # calculate deaths using np.count_nonzero(self.ti_dead == ti)
        # super().update_results()  # Updates n_alive and other base results
        ti = self.sim.ti
        res = self.sim.results
        res.n_alive[ti] = np.count_nonzero(self.alive)
        # res.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        res.cum_deaths[ti] = np.sum(res.new_deaths[:ti]) # TODO: inefficient to compute the cumulative sum on every timestep!
        self._step_results_wq()  # Updates wealth quintile results
        return

    def _step_results_wq(self):
        """" Calculate step results on wealthquintile """
        for i in range(1, 6):
            self.sim.results[f'n_wq{i}'][self.sim.ti] = np.sum((self.wealthquintile == i) & self.female)
        return

    @staticmethod
    def cond_prob(a, b):
        """ Calculate conditional probability. This should be moved somewhere else. """
        return np.sum(a & b) / np.sum(b)

    def int_age(self, uids=None):
        """ Return ages as an integer """
        if uids is None:
            return np.array(self.age, dtype=np.int64)
        return np.array(self.age[uids], dtype=np.int64)

    def int_age_clip(self, uids=None):
        """ Return ages as integers, clipped to maximum allowable age for pregnancy """
        if uids is None:
            return np.minimum(self.int_age(), fp.max_age_preg)
        return np.minimum(self.int_age(uids), fp.max_age_preg)

    def update_post(self):
        """ Final updates at the very end of the timestep """
        sim = self.sim
        if sim.pars.use_aging:
            self.age[self.alive.uids] += sim.t.dt_year
            # there is a max age for some of the stats, so if we exceed that, reset it
            self.age[self.alive.uids] = np.minimum(self.age[self.alive.uids], self.sim.pars.fp['max_age'])
        return

    def compute_method_usage(self):
        """
        Computes method mix proportions from a sim object
        Returns:
            list of lists where list[years_after_start][method_index] == proportion of
            fecundity aged women using that method on that year
        """

        min_age = fp.min_age
        max_age = self.sim.pars.fp['age_limit_fecundity']

        # filtering for women with appropriate characteristics
        bool_list_uids = (self.alive & (self.female) & (self.age >= min_age) * (self.age <= max_age)).uids
        filtered_methods = self.method[bool_list_uids]

        unique, counts = np.unique(filtered_methods, return_counts=True)
        count_dict = dict(zip(unique, counts))

        # Initialize result list with zeros for each method
        cm = self.sim.connectors.contraception
        result = [0] * (len(cm.methods))
        for method in count_dict:
            result[int(method)] = count_dict[int(method)] / len(filtered_methods)

        return result

    def plot(self, fig_args=None, hist_args=None):
        ''' Plot histograms of each quantity '''

        fig_args  = sc.mergedicts(fig_args)
        hist_args = sc.mergedicts(dict(bins=50), hist_args)
        keys = [key for key in self.__dict__.keys() if isinstance(self.__dict__[key], ss.FloatArr)]
        nkeys = len(keys)
        rows,cols = sc.get_rows_cols(nkeys)

        fig = pl.figure(**fig_args)

        for k,key in enumerate(keys):
            pl.subplot(rows,cols,k+1)
            try:
                data = np.array(self[key], dtype=float)
                mean = data.mean()
                label = f'mean: {mean}'
                pl.hist(data, label=label, **hist_args)
                pl.title(key)
                pl.legend()
            except:
                pl.title(f'Could not plot {key}')

        return fig