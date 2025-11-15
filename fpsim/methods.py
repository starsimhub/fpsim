"""
Contraceptive methods

Idea:
    A method selector should be a generic class (an intervention?). We want the existing matrix-based
    method and the new duration-based method to both be instances of this.

"""

# >> Imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import sciris as sc
import starsim as ss
from scipy.special import expit
from scipy.stats import fisk
from . import defaults as fpd
from . import locations as fplocs

__all__ = ['Method', 'NewMethodConfig', 'make_methods', 'make_method_list', 'ContraPars', 'make_contra_pars', 'ContraceptiveChoice', 'RandomChoice', 'SimpleChoice', 'StandardChoice']


# >> Base definition of contraceptive methods -- can be overwritten by locations
class Method:
    def __init__(self, name=None, label=None, idx=None, efficacy=None, modern=None, dur_use=None, csv_name=None):
        self.name = name
        self.label = label or name
        self.csv_name = csv_name or label or name
        self.idx = idx
        self.efficacy = efficacy
        self.modern = modern
        self.dur_use = dur_use

    def gamma_scale_callback(self, sim, uids):
        """ Sample from gamma distribution with age factors """
        ppl = sim.people
        if sim.connectors.contraception.age_bins is not None:
            age_bins = np.digitize(ppl.age[uids], sim.connectors.contraception.age_bins)
            scale = 1 / np.exp(self.dur_use.base_scale + self.dur_use.age_factors[age_bins])
        else:
            scale = 1 / np.exp(self.dur_use.base_scale)
        return scale

    def expon_scale_callback(self, sim, uids):
        """ Sample from exponential distribution with age factors """
        ppl = sim.people
        if sim.connectors.contraception.age_bins is not None:
            age_bins = np.digitize(ppl.age[uids], sim.connectors.contraception.age_bins)
            scale = 1 / np.exp(self.dur_use.base_scale + self.dur_use.age_factors[age_bins])
        else:
            scale = 1 / np.exp(self.dur_use.base_scale)
        return scale

    def lognorm_mean_callback(self, sim, uids):
        """ Sample from lognormal distribution with age factors """
        ppl = sim.people
        if sim.connectors.contraception.age_bins is not None:
            # Use age bins to apply age factors
            age_bins = np.digitize(ppl.age[uids], sim.connectors.contraception.age_bins)
            age_factors = self.dur_use.age_factors[age_bins]
            mean = np.exp(self.dur_use.base_mean + age_factors[age_bins])
        else:
            # If no age bins, just use the base mean
            mean = np.exp(self.dur_use.base_mean)
        return mean

    def llogis_scale_callback(self, sim, uids):
        """ Sample from log-logistic distribution with age factors """
        ppl = sim.people
        if sim.connectors.contraception.age_bins is not None:
            age_bins = np.digitize(ppl.age[uids], sim.connectors.contraception.age_bins)
            scale = np.exp(self.dur_use.base_scale + self.dur_use.age_factors[age_bins])
        else:
            scale = np.exp(self.dur_use.base_scale)
        return scale

    def weibull_scale_callback(self, sim, uids):
        """ Sample from Weibull distribution with age factors """
        ppl = sim.people
        if sim.connectors.contraception.age_bins is not None:
            age_bins = np.digitize(ppl.age[uids], sim.connectors.contraception.age_bins)
            scale = np.exp(self.dur_use.base_scale + self.dur_use.age_factors[age_bins])
        else:
            scale = np.exp(self.dur_use.base_scale)
        return scale

    def set_dur_use(self, dist_type, par1=None, par2=None, age_factors=None, **kwargs):
        """
        Set the duration of use for this method.
        Args:
            dist_type: Type of distribution to use (e.g., 'lognorm', 'gamma', etc.)
            par1: First parameter for the distribution (e.g., mean for lognorm, shape for gamma)
            par2: Second parameter for the distribution (e.g., std for lognorm, scale for gamma)
            age_factors: Optional age factors to apply to the duration
            kwargs: Additional parameters for the distribution
        """

        if dist_type == 'lognorm':
            # Lognormal distribution
            self.dur_use = ss.lognorm_ex(mean=self.lognorm_mean_callback, std=np.exp(par2))
            self.dur_use.base_mean = par1
            self.dur_use.base_std = par2

        elif dist_type == 'gamma':
            # Gamma distribution
            self.dur_use = ss.gamma(a=np.exp(par1), scale=self.gamma_scale_callback)
            self.dur_use.base_a = par1
            self.dur_use.base_scale = par2

        elif dist_type == 'llogis':

            self.dur_use = Fisk(c=np.exp(par1), scale=self.llogis_scale_callback)
            self.dur_use.base_c = par1  # This is the scale parameter for the log-logistic distribution
            self.dur_use.base_scale = par2

        elif dist_type == 'weibull':

            self.dur_use = ss.weibull(c=par1, scale=self.weibull_scale_callback)
            self.dur_use.base_c = par1
            self.dur_use.base_scale = par2  # This is the scale parameter for the Weibull distribution

        elif dist_type == 'exponential':
            # Exponential distribution

            self.dur_use = ss.expon(scale=self.expon_scale_callback)
            self.dur_use.base_scale = par1

        if age_factors is not None:
            self.dur_use.age_factors = age_factors


# Helper function for setting lognormals - now returns Starsim distribution  
def ln(a, b): return ss.lognorm_ex(mean=a, std=b)


def make_method_list():
    method_list = [
        Method(name='none',     efficacy=0,     modern=False, dur_use=ln(2, 3), label='None'),
        Method(name='pill',     efficacy=0.945, modern=True,  dur_use=ln(2, 3), label='Pill'),
        Method(name='iud',      efficacy=0.986, modern=True, dur_use=ln(5, 3), label='IUDs', csv_name='IUD'),
        Method(name='inj',      efficacy=0.983, modern=True, dur_use=ln(2, 3), label='Injectables', csv_name='Injectable'),
        Method(name='cond',     efficacy=0.946, modern=True,  dur_use=ln(1, 3), label='Condoms', csv_name='Condom'),
        Method(name='btl',      efficacy=0.995, modern=True, dur_use=ln(50, 3), label='BTL', csv_name='F.sterilization'),
        Method(name='wdraw',    efficacy=0.866, modern=False, dur_use=ln(1, 3), label='Withdrawal', csv_name='Withdrawal'), #     # 1/2 periodic abstinence, 1/2 other traditional approx.  Using rate from periodic abstinence
        Method(name='impl',     efficacy=0.994, modern=True, dur_use=ln(2, 3), label='Implants', csv_name='Implant'),
        Method(name='othtrad',  efficacy=0.861, modern=False, dur_use=ln(1, 3), label='Other traditional', csv_name='Other.trad'),
        Method(name='othmod',   efficacy=0.880, modern=True, dur_use=ln(1, 3), label='Other modern', csv_name='Other.mod'),
    ]
    idx = 0
    for method in method_list:
        method.idx = idx
        idx += 1
    return sc.dcp(method_list)


def make_method_map(method_list):
    method_map = {method.label: method.idx for method in method_list}
    return method_map


def make_methods(method_list=None):
    if method_list is None: method_list = make_method_list()
    return ss.ndict(method_list, type=Method)


InitialShareInput = Union[float, int, np.ndarray, Callable[[], float], ss.Dist]


@dataclass(slots=True)
class NewMethodConfig:
    """Configuration payload used when dynamically adding a new contraceptive method."""

    method: Method
    copy_from_row: Optional[str]
    copy_from_col: Optional[str]
    initial_share: InitialShareInput = 0.1
    renormalize: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.method, Method):
            raise TypeError('`method` must be a fpsim.Method instance')
        if self.copy_from_row is None or self.copy_from_col is None:
            raise ValueError('copy_from_row and copy_from_col must be provided')
        self.initial_share = self._coerce_initial_share(self.initial_share)

    @staticmethod
    def _coerce_initial_share(value: InitialShareInput) -> float:
        """Convert initial_share inputs (float/int/callable/distribution) to a bounded float."""
        # Numpy scalar/array
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError('initial_share ndarray must contain a single value')
            value = value.item()

        # Starsim/scipy distribution objects (have .rvs)
        if hasattr(value, 'rvs') and callable(getattr(value, 'rvs')):
            sampled = value.rvs()
            return NewMethodConfig._coerce_initial_share(sampled)

        # Callables returning a value
        if callable(value):
            sampled = value()
            return NewMethodConfig._coerce_initial_share(sampled)

        # Numeric types
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise TypeError('initial_share must be a float, int, numpy scalar, callable, or distribution')

        if not (0.0 <= numeric <= 1.0):
            raise ValueError('initial_share must be between 0 and 1')
        return numeric


class Fisk(ss.Dist):
    """ Wrapper for scipy's fisk distribution to make it compatible with starsim """
    def __init__(self, c=0.0, scale=1.0, **kwargs):
        super().__init__(distname='fisk', dist=fisk, c=c, scale=scale, **kwargs)
        return


# >> Define parameters
class ContraPars(ss.Pars):
    def __init__(self, **kwargs):
        super().__init__()

        # Methods
        self.methods = make_method_list()  # Default methods

        # Probabilities and choices
        self.p_use = ss.bernoulli(p=0.5)
        self.force_choose = False  # Whether to force non-users to choose a method
        self.method_mix = 'uniform'  #np.array([1/self.n_methods]*self.n_methods)
        self.method_weights = None  #np.ones(n_methods)

        # mCPR trend
        self.prob_use_year = 2000
        self.prob_use_intercept = 0.0
        self.prob_use_trend_par = 0.0

        # Data pars, all None by default and populated with data
        self.age_spline = None
        self.init_dist = None
        self.dur_use_df = None
        self.contra_use_pars = None
        self.method_choice_pars = None

        # Settings and other misc
        self.max_dur = ss.years(100)  # Maximum duration of use in years
        self.update(kwargs)
        return


def make_contra_pars():
    """ Shortcut for making a new instance of ContraPars """
    return ContraPars()


# >> Define classes to contain information about the way women choose contraception

class ContraceptiveChoice(ss.Connector):
    def __init__(self, pars=None, **kwargs):
        """
        Base contraceptive choice module
        """
        super().__init__(name='contraception')

        # Handle parameters
        default_pars = ContraPars()
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)

        # Copy methods as main attribute
        self.methods = make_methods(self.pars.methods)  # Store the methods as an ndict
        self.n_options = len(self.methods)
        self.n_methods = len([m for m in self.methods if m != 'none'])

        # Process pars
        if self.pars.method_mix == 'uniform':
            self.pars.method_mix = np.array([1/self.n_methods]*self.n_methods)
        if self.pars.method_weights is None:
            self.pars.method_weights = np.ones(self.n_methods)

        self.init_dist = None
        self.data = {}
        
        # Initialize choice distributions for method selection
        self._method_choice_dist = ss.choice(a=self.n_methods, p=np.ones(self.n_methods)/self.n_methods)
        self._jitter_dist = ss.normal(loc=0, scale=1e-4)

        return

    def init_results(self):
        """
        Initialize results for this module
        """
        super().init_results()

        self.define_results(
            ss.Result('n_at_risk_non_users', scale=True, label="Number of non-users at risk of pregnancy (aCPR)"),
            ss.Result('n_at_risk_users', scale=True, label="Number of users at risk of pregnancy (aCPR)"),
            ss.Result('n_non_users', scale=True, label="Number of non-users (CPR)"),
            ss.Result('n_mod_users', scale=True, label="Number of modern contraceptive users (mCPR)"),
            ss.Result('n_users', scale=True, label="Number of contraceptive users (CPR)"),
            ss.Result('mcpr', scale=False, label="Modern contraceptive prevalence rate (mCPR)"),
            ss.Result('cpr', scale=False, label="Contraceptive prevalence rate (CPR)"),
            ss.Result('acpr', scale=False, label="Active contraceptive prevalence rate (aCPR)"),
        )
        return

    @property
    def average_dur_use(self):
        av = 0
        # todo verify property names
        for m in self.methods.values():
            if sc.isnumber(m.dur_use): 
                av += m.dur_use
            elif hasattr(m.dur_use, 'mean'):
                # Starsim distribution object
                av += m.dur_use.mean()
            elif hasattr(m.dur_use, 'scale'):
                # For distributions that use scale parameter as approximation of mean
                av += m.dur_use.scale
        return av / len(self.methods)

    def init_post(self):
        """
         Decide who will start using contraception, when, which contraception method and the
         duration on that method. This method is called by the simulation to initialise the
         people object at the beginning of the simulation and new people born during the simulation.
         """
        super().init_post()
        ppl = self.sim.people
        fecund = ppl.female & (ppl.age < self.sim.pars.fp['age_limit_fecundity'])
        fecund_uids = fecund.uids

        # Look for women who have reached the time to choose
        time_to_set_contra_uids = fecund_uids[(ppl.fp.ti_contra[fecund_uids] == 0)]
        self.init_contraception(time_to_set_contra_uids)
        return

    def init_contraception(self, uids):
        """
        Used for all agents at the start of a sim and for newly active agents throughout
        """
        contra_users, _ = self.get_contra_users(uids)
        self.start_contra(contra_users)
        self.init_methods(contra_users)
        return

    def start_contra(self, uids):
        """ Wrapper method to start contraception for a set of users """
        self.sim.people.fp.on_contra[uids] = True
        self.sim.people.fp.ever_used_contra[uids] = 1
        return

    def init_methods(self, uids):
        # Set initial distribution of methods
        self.sim.people.fp.method[uids] = self.init_method_dist(uids)
        method_dur = self.set_dur_method(uids)
        self.sim.people.fp.ti_contra[uids] = self.ti + method_dur
        return

    def get_method_by_label(self, method_label):
        """ 
        Extract method according to its label / long name or short name.
        
        This method looks for methods by label first, then by name.
        This allows it to work with both standard methods and dynamically added methods.
        """
        return_val = None
        # First try to match by label
        for method_name, method in self.methods.items():
            if method.label == method_label:
                return_val = method
                break
        # If no match, try by name (for dynamically added methods)
        if return_val is None and method_label in self.methods:
            return_val = self.methods[method_label]
        # If still no match, raise error
        if return_val is None:
            errormsg = f'No method matching "{method_label}" found in methods: {list(self.methods.keys())}'
            raise ValueError(errormsg)
        return return_val

    def update_efficacy(self, method_label=None, new_efficacy=None):
        method = self.get_method_by_label(method_label)
        method.efficacy = new_efficacy

    def update_duration(self, method_label=None, new_duration=None):
        method = self.get_method_by_label(method_label)
        method.dur_use = new_duration

    def add_method(self, method):
        self.methods[method.name] = method

    def add_new_method(self, config: NewMethodConfig):
        """
        Add a new contraceptive method and expand the switching matrix dynamically.
        
        This allows adding new methods during a simulation (e.g., when a new product
        becomes available). The switching matrix is expanded by copying transition
        probabilities from existing similar methods.
        
        Args:
            config (NewMethodConfig): Configuration payload describing the method to add
            
        Example:
            # Add a new long-acting injectable method
            new_method = fp.Method(
                name='la_inj', 
                label='Long-acting injectable',
                efficacy=0.99, 
                modern=True,
                dur_use=fp.ln(6, 2)
            )
            cfg = fp.NewMethodConfig(
                method=new_method,
                copy_from_row='inj',  # Copy switching patterns from injectables
                copy_from_col='inj',
                initial_share=0.15
            )
            sim.connectors.contraception.add_new_method(cfg)
        """
        method = config.method
        if method.name in self.methods:
            raise ValueError(f"Method '{method.name}' already exists in the simulation")
        
        # Set the index for the new method
        method.idx = len(self.methods)
        
        # Add method to the methods dict
        self.methods[method.name] = method
        self.n_options = len(self.methods)
        self.n_methods = len([m for m in self.methods if m != 'none'])
        
        # Handle switching matrix expansion if method_choice_pars exists
        if hasattr(self, 'pars') and self.pars.method_choice_pars is not None:
            # This is for modules that use switching matrices (SimpleChoice, StandardChoice)
            self._expand_switching_matrix(config)
        
        # Update method mix if it exists
        if hasattr(self, 'pars') and hasattr(self.pars, 'method_mix'):
            # Add a small entry for the new method and renormalize
            new_mix = np.append(self.pars.method_mix, config.initial_share)
            mix_sum = new_mix.sum()
            if mix_sum > 0:
                self.pars.method_mix = new_mix / mix_sum
        
        # Update method_weights if it exists (used in StandardChoice)
        if hasattr(self, 'pars') and hasattr(self.pars, 'method_weights'):
            # Add weight for the new method (copy from the reference method or use 1.0)
            if config.copy_from_row and config.copy_from_row in self.methods:
                row_idx = self.methods[config.copy_from_row].idx
                if row_idx < len(self.pars.method_weights):
                    new_weight = self.pars.method_weights[row_idx]
                else:
                    new_weight = 1.0
            else:
                new_weight = 1.0
            self.pars.method_weights = np.append(self.pars.method_weights, new_weight)
        
        # Resize method_mix array in FPmod if it exists
        # This is needed because method_mix is pre-allocated with fixed size
        if hasattr(self, 'sim') and hasattr(self.sim, 'connectors'):
            try:
                # FPmod is stored as a connector named 'fp'
                fp_mod = self.sim.connectors['fp']
                old_shape = fp_mod.method_mix.shape
                new_shape = (self.n_options, old_shape[1])
                new_method_mix = np.zeros(new_shape)
                # Copy existing data
                new_method_mix[:old_shape[0], :] = fp_mod.method_mix
                fp_mod.method_mix = new_method_mix
            except (AttributeError, KeyError) as e:
                # If method_mix doesn't exist yet, it will be created with correct size
                pass
        
        return
    
    def _expand_switching_matrix(self, config: NewMethodConfig):
        """
        Expand the switching matrix to accommodate a new method.
        
        This is an internal helper for add_new_method() that handles the complex
        logic of expanding nested switching matrix structures.
        """
        new_method = config.method
        copy_from_row = config.copy_from_row
        copy_from_col = config.copy_from_col
        initial_share = config.initial_share
        renormalize = config.renormalize

        # Get the source methods for copying probabilities
        if copy_from_row not in self.methods:
            raise ValueError(f"copy_from_row '{copy_from_row}' not found in methods")
        if copy_from_col not in self.methods:
            raise ValueError(f"copy_from_col '{copy_from_col}' not found in methods")
        
        method_lookup = self.methods
        row_method = method_lookup[copy_from_row]
        col_method = method_lookup[copy_from_col]
        
        mcp = self.pars.method_choice_pars
        
        # Update each age bin/event in the switching structure
        for event_idx, event_data in mcp.items():
            if not isinstance(event_data, dict):
                print(f"Warning: Event data for event {event_idx} is not a dictionary: {event_data}")
                continue
                
            # Track current method_idx list
            if 'method_idx' in event_data:
                # Add new method index to the list
                current_idx_list = list(event_data['method_idx'])
                event_data['method_idx'] = current_idx_list + [new_method.idx]
            
            # Iterate through age bins
            for age_key, age_data in event_data.items():
                if age_key == 'method_idx':
                    continue
                
                # Handle case where age_data is a numpy array directly (Event 1 structure)
                if isinstance(age_data, np.ndarray):
                    # This is a direct probability vector for choosing methods
                    # We need to append a probability for the new method
                    probs = age_data
                    
                    # Calculate probability for new method based on reference method
                    if col_method.idx < len(probs):
                        new_prob = probs[col_method.idx] * initial_share
                    else:
                        new_prob = initial_share / len(probs)
                    
                    new_probs = np.append(probs, new_prob)
                    
                    if renormalize:
                        new_probs = new_probs / new_probs.sum()
                    
                    event_data[age_key] = new_probs
                    continue
                
                if not isinstance(age_data, dict):
                    continue
                
                # For postpartum events (pp1), structure is different
                if 'probs' in age_data and 'method_idx' in age_data:
                    # This is a direct probability vector
                    probs = np.array(age_data['probs'])
                    method_idx = list(age_data['method_idx'])
                    
                    # Find the probability to copy
                    if col_method.idx in method_idx:
                        col_idx = method_idx.index(col_method.idx)
                        new_prob = probs[col_idx] * initial_share
                    else:
                        new_prob = initial_share / len(probs)
                    
                    # Append new probability
                    new_probs = np.append(probs, new_prob)
                    
                    # Renormalize if requested
                    if renormalize:
                        new_probs = new_probs / new_probs.sum()
                    
                    age_data['probs'] = new_probs.tolist()
                    age_data['method_idx'] = method_idx + [new_method.idx]
                
                # For regular switching structure
                else:
                    # Add a new entry for switching FROM the new method
                    if row_method.name in age_data:
                        # Copy the structure from the row method
                        row_data = age_data[row_method.name]
                        
                        # Check if it's a numpy array (direct probabilities) or dict structure
                        if isinstance(row_data, np.ndarray):
                            # Direct probability array - append staying probability
                            probs = row_data.copy()
                            new_probs = np.append(probs, initial_share)
                            if renormalize:
                                new_probs = new_probs / new_probs.sum()
                            age_data[new_method.name] = new_probs
                        
                        elif isinstance(row_data, dict):
                            # Dictionary structure with 'probs' and 'method_idx' keys
                            new_method_data = sc.dcp(row_data)
                            
                            if 'probs' in new_method_data:
                                probs = np.array(new_method_data['probs'])
                                new_probs = np.append(probs, initial_share)
                                if renormalize:
                                    new_probs = new_probs / new_probs.sum()
                                new_method_data['probs'] = new_probs.tolist()
                            
                            if 'method_idx' in new_method_data:
                                method_idx = list(new_method_data['method_idx'])
                                new_method_data['method_idx'] = method_idx + [new_method.idx]
                            
                            age_data[new_method.name] = new_method_data
                    
                    # Update existing methods to add probability of switching TO new method
                    for origin_method_name, origin_data in age_data.items():
                        if origin_method_name == new_method.name:
                            continue
                        
                        # Handle numpy array structure (direct probabilities)
                        if isinstance(origin_data, np.ndarray):
                            probs = origin_data
                            
                            # Find the reference method's probability to copy
                            # col_method.idx is the index in the probability array
                            if col_method.idx < len(probs):
                                new_prob = probs[col_method.idx] * initial_share
                            else:
                                new_prob = initial_share / len(probs)
                            
                            new_probs = np.append(probs, new_prob)
                            
                            if renormalize:
                                new_probs = new_probs / new_probs.sum()
                            
                            age_data[origin_method_name] = new_probs
                        
                        # Handle dictionary structure
                        elif isinstance(origin_data, dict):
                            if 'probs' not in origin_data or 'method_idx' not in origin_data:
                                continue
                            
                            method_idx = list(origin_data['method_idx'])
                            probs = np.array(origin_data['probs'])
                            
                            # Add probability of switching to new method
                            if col_method.idx in method_idx:
                                col_idx = method_idx.index(col_method.idx)
                                new_prob = probs[col_idx] * initial_share
                            else:
                                new_prob = initial_share / len(probs)
                            
                            new_probs = np.append(probs, new_prob)
                            
                            if renormalize:
                                new_probs = new_probs / new_probs.sum()
                            
                            origin_data['probs'] = new_probs.tolist()
                            origin_data['method_idx'] = method_idx + [new_method.idx]
        
        return

    def remove_method(self, method_label, reassign_to='none'):
        """
        Remove a contraceptive method from the simulation.
        
        This allows removing methods during a simulation (e.g., when a product
        is discontinued or becomes unavailable). The switching matrix is adjusted
        by removing the method's row and column, and users currently on that method
        are reassigned to another method.
        
        Args:
            method_label (str): Label or name of the method to remove
            reassign_to (str): Method to reassign current users to (default: 'none')
            
        Example:
            # Remove a method from the simulation
            sim.connectors.contraception.remove_method('Implants', reassign_to='Injectables')
        """
        # Get the method to remove
        method = self.get_method_by_label(method_label)
        if method is None:
            raise ValueError(f"Method '{method_label}' not found")
        
        # Don't allow removing 'none' method
        if method.name == 'none':
            raise ValueError("Cannot remove the 'none' method")
        
        # Store the index and name before removal for matrix adjustments
        removed_idx = method.idx
        removed_name = method.name
        
        # Reassign people currently using this method
        if hasattr(self, 'sim') and self.sim is not None:
            ppl = self.sim.people
            currently_using = ppl.fp.method == removed_idx
            
            if currently_using.any():
                # Get the reassignment method
                reassign_method = self.get_method_by_label(reassign_to)
                if reassign_method is None:
                    raise ValueError(f"Reassignment method '{reassign_to}' not found")
                
                # Reassign users
                ppl.fp.method[currently_using] = reassign_method.idx
                
                # If reassigning to 'none', turn off contraception
                if reassign_method.name == 'none':
                    ppl.fp.on_contra[currently_using] = False
        
        # Remove method from the methods dict
        del self.methods[method.name]
        
        # Update counters
        self.n_options = len(self.methods)
        self.n_methods = len([m for m in self.methods if m != 'none'])
        
        # Re-index remaining methods (shift indices down for methods after removed one)
        for mname, m in self.methods.items():
            if m.idx > removed_idx:
                m.idx -= 1
        
        # Adjust method indices in people's state
        if hasattr(self, 'sim') and self.sim is not None:
            ppl = self.sim.people
            # Shift down all method indices greater than removed_idx
            higher_indices = ppl.fp.method > removed_idx
            ppl.fp.method[higher_indices] -= 1
        
        # Handle switching matrix contraction if method_choice_pars exists
        if hasattr(self, 'pars') and self.pars.method_choice_pars is not None:
            self._contract_switching_matrix(removed_idx, removed_name)
        
        # Update method mix if it exists
        if hasattr(self, 'pars') and hasattr(self.pars, 'method_mix'):
            # Remove the entry for this method and renormalize
            if removed_idx < len(self.pars.method_mix):
                # For method_mix, we need to handle the fact that 'none' is not included
                # Method indices are 0-based, but method_mix excludes 'none' (idx 0)
                mix_idx = removed_idx - 1 if removed_idx > 0 else None
                if mix_idx is not None and mix_idx >= 0 and mix_idx < len(self.pars.method_mix):
                    self.pars.method_mix = np.delete(self.pars.method_mix, mix_idx)
                    # Renormalize
                    mix_sum = self.pars.method_mix.sum()
                    if mix_sum > 0:
                        self.pars.method_mix = self.pars.method_mix / mix_sum
        
        # Update method_weights if it exists
        if hasattr(self, 'pars') and hasattr(self.pars, 'method_weights'):
            if removed_idx < len(self.pars.method_weights):
                mix_idx = removed_idx - 1 if removed_idx > 0 else None
                if mix_idx is not None and mix_idx >= 0 and mix_idx < len(self.pars.method_weights):
                    self.pars.method_weights = np.delete(self.pars.method_weights, mix_idx)
        
        # Resize method_mix array in FPmod if it exists
        if hasattr(self, 'sim') and hasattr(self.sim, 'connectors'):
            try:
                fp_mod = self.sim.connectors['fp']
                if hasattr(fp_mod, 'method_mix'):
                    old_shape = fp_mod.method_mix.shape
                    new_shape = (self.n_options, old_shape[1])
                    if new_shape[0] < old_shape[0]:
                        # Create new smaller array and copy data
                        new_method_mix = np.zeros(new_shape)
                        # Copy rows before removed index
                        new_method_mix[:removed_idx, :] = fp_mod.method_mix[:removed_idx, :]
                        # Copy rows after removed index (shifted down)
                        if removed_idx < old_shape[0] - 1:
                            new_method_mix[removed_idx:, :] = fp_mod.method_mix[removed_idx+1:, :]
                        fp_mod.method_mix = new_method_mix
            except (AttributeError, KeyError):
                pass
        
        return
    
    def _contract_switching_matrix(self, removed_idx, removed_name):
        """
        Contract the switching matrix to remove a method.
        
        This is an internal helper for remove_method() that handles the complex
        logic of contracting nested switching matrix structures.
        
        Args:
            removed_idx: The index of the method being removed
            removed_name: The name of the method being removed
        """
        mcp = self.pars.method_choice_pars
        
        # Update each age bin/event in the switching structure
        for event_idx, event_data in mcp.items():
            if not isinstance(event_data, dict):
                continue
            
            # Update method_idx list if present
            if 'method_idx' in event_data:
                method_idx_list = list(event_data['method_idx'])
                if removed_idx in method_idx_list:
                    idx_pos = method_idx_list.index(removed_idx)
                    method_idx_list.pop(idx_pos)
                    # Shift down indices greater than removed_idx
                    method_idx_list = [idx - 1 if idx > removed_idx else idx for idx in method_idx_list]
                    event_data['method_idx'] = method_idx_list
            
            # Iterate through age bins
            for age_key, age_data in event_data.items():
                if age_key == 'method_idx':
                    continue
                
                # Handle case where age_data is a numpy array directly (Event 1 structure)
                if isinstance(age_data, np.ndarray):
                    # Remove the probability for the removed method
                    if removed_idx < len(age_data):
                        new_probs = np.delete(age_data, removed_idx)
                        # Renormalize
                        if new_probs.sum() > 0:
                            new_probs = new_probs / new_probs.sum()
                        event_data[age_key] = new_probs
                    continue
                
                if not isinstance(age_data, dict):
                    continue
                
                # For postpartum events (pp1), structure is different
                if 'probs' in age_data and 'method_idx' in age_data:
                    method_idx_list = list(age_data['method_idx'])
                    if removed_idx in method_idx_list:
                        idx_pos = method_idx_list.index(removed_idx)
                        probs = np.array(age_data['probs'])
                        
                        # Remove the probability at that position
                        new_probs = np.delete(probs, idx_pos)
                        # Renormalize
                        if new_probs.sum() > 0:
                            new_probs = new_probs / new_probs.sum()
                        
                        age_data['probs'] = new_probs.tolist()
                        method_idx_list.pop(idx_pos)
                        # Shift down indices
                        method_idx_list = [idx - 1 if idx > removed_idx else idx for idx in method_idx_list]
                        age_data['method_idx'] = method_idx_list
                
                # For regular switching structure - need to remove the method as an origin
                # Use the removed_name parameter instead of searching for it
                if removed_name in age_data:
                    del age_data[removed_name]
                
                # Now update remaining methods to remove probability of switching TO removed method
                for origin_method_name, origin_data in age_data.items():
                    # Handle numpy array structure (direct probabilities)
                    if isinstance(origin_data, np.ndarray):
                        if removed_idx < len(origin_data):
                            new_probs = np.delete(origin_data, removed_idx)
                            # Renormalize
                            if new_probs.sum() > 0:
                                new_probs = new_probs / new_probs.sum()
                            age_data[origin_method_name] = new_probs
                    
                    # Handle dictionary structure
                    elif isinstance(origin_data, dict):
                        if 'probs' not in origin_data or 'method_idx' not in origin_data:
                            continue
                        
                        method_idx_list = list(origin_data['method_idx'])
                        if removed_idx in method_idx_list:
                            idx_pos = method_idx_list.index(removed_idx)
                            probs = np.array(origin_data['probs'])
                            
                            # Remove probability at that position
                            new_probs = np.delete(probs, idx_pos)
                            # Renormalize
                            if new_probs.sum() > 0:
                                new_probs = new_probs / new_probs.sum()
                            
                            origin_data['probs'] = new_probs.tolist()
                            method_idx_list.pop(idx_pos)
                            # Shift down indices
                            method_idx_list = [idx - 1 if idx > removed_idx else idx for idx in method_idx_list]
                            origin_data['method_idx'] = method_idx_list
        
        return

    def get_prob_use(self, uids, event=None):
        pass

    def get_contra_users(self, uids, event=None):
        """ Select contraception users, return boolean array """
        self.get_prob_use(uids, event=event)  # Call this to reset p_use parameter
        users, non_users = self.pars.p_use.split(uids)
        return users, non_users

    def update_contra(self, uids):
        """ Update contraceptive choices for a set of users. """
        sim = self.sim
        ti = self.ti
        ppl = sim.people
        fpppl = ppl.fp  # Shorter name for people.fp

        # If people are 1 or 6m postpartum, we use different parameters for updating their contraceptive decisions
        is_pp1 = (self.ti - fpppl.ti_delivery[uids]) == 1  # Delivered last timestep
        is_pp6 = ((self.ti - fpppl.ti_delivery[uids]) == 6) & ~fpppl.on_contra[uids]  # They may have decided to use contraception after 1m
        pp0 = uids[~(is_pp1 | is_pp6)]
        pp1 = uids[is_pp1]
        pp6 = uids[is_pp6]

        # Update choices for people who aren't postpartum
        if len(pp0):

            # If force_choose is True, then all non-users will be made to pick a method
            if self.pars['force_choose']:
                must_use = pp0[~fpppl.on_contra[pp0]]
                choosers = pp0[fpppl.on_contra[pp0]]

                if len(must_use):
                    self.start_contra(must_use)  # Start contraception for those who must use
                    fpppl.method[must_use] = self.choose_method(must_use)

            else:
                choosers = pp0

            # Get previous users and see whether they will switch methods or stop using
            if len(choosers):

                users, non_users = self.get_contra_users(choosers)

                if len(non_users):
                    fpppl.on_contra[non_users] = False  # Set non-users to not using contraception
                    fpppl.method[non_users] = 0  # Set method to zero for non-users

                # For those who keep using, choose their next method
                if len(users):
                    self.start_contra(users)
                    fpppl.method[users] = self.choose_method(users)

            # Validate
            n_methods = len(self.methods)
            invalid_vals = (fpppl.method[pp0] >= n_methods) * (fpppl.method[pp0] < 0) * (np.isnan(fpppl.method[pp0]))
            if invalid_vals.any():
                errormsg = f'Invalid method set: ti={pp0.ti}, inds={invalid_vals.nonzero()[-1]}'
                raise ValueError(errormsg)

        # Now update choices for postpartum people. Logic here is simpler because none of these
        # people should be using contraception currently. We first check that's the case, then
        # have them choose their contraception options.
        ppdict = {'pp1': pp1, 'pp6': pp6}
        for event, pp in ppdict.items():
            if len(pp):
                if fpppl.on_contra[pp].any():
                    errormsg = 'Postpartum women should not currently be using contraception.'
                    raise ValueError(errormsg)
                users, _ = self.get_contra_users(pp, event=event)
                self.start_contra(users)
                on_contra = pp[fpppl.on_contra[pp]]
                off_contra = pp[~fpppl.on_contra[pp]]

                # Set method for those who use contraception
                if len(on_contra):
                    method_used = self.choose_method(on_contra, event=event)
                    fpppl.method[on_contra] = method_used

                if len(off_contra):
                    fpppl.method[off_contra] = 0
                    if event == 'pp1':  # For women 1m postpartum, choose again when they are 6 months pp
                        fpppl.ti_contra[off_contra] = ti + 5

        # Set duration of use for everyone, and reset the time they'll next update
        durs_fixed = ((self.ti - fpppl.ti_delivery[uids]) == 1) & (fpppl.method[uids] == 0)
        update_durs = uids[~durs_fixed]
        dur_methods = self.set_dur_method(update_durs)

        # Check validity
        if (dur_methods < 0).any():
            raise ValueError('Negative duration of method use')

        fpppl.ti_contra[update_durs] = ti + dur_methods

        return

    def set_dur_method(self, uids, method_used=None):
        dt = self.t.dt_year
        timesteps_til_update = np.full(len(uids), np.round(self.average_dur_use/dt), dtype=int)
        return timesteps_til_update

    def set_method(self, uids):
        """ Wrapper for choosing method and assigning duration of use """
        ppl = self.sim.people.fp
        method_used = self.choose_method(uids)
        ppl.method[uids] = method_used

        # Set the duration of use
        dur_method = self.set_dur_method(uids)
        ppl.ti_contra[uids] = self.ti + dur_method

        return

    def step(self):
        # TODO, could move all update logic to here...
        pass

    def update_results(self):
        """
        Note that we are not including LAM users in mCPR as this model counts
        all women passively using LAM but DHS data records only women who self-report
        LAM which is much lower. Follows the DHS definition of mCPR.
        """
        super().update_results()
        ppl = self.sim.people
        method_age = self.sim.pars.fp['method_age'] <= ppl.age
        fecund_age = ppl.age < self.sim.pars.fp['age_limit_fecundity']
        denominator = method_age * fecund_age * ppl.female * ppl.alive

        # Track mCPR
        modern_methods_num = [idx for idx, m in enumerate(self.methods.values()) if m.modern]
        numerator = np.isin(ppl.fp.method, modern_methods_num)
        n_no_method = np.sum((ppl.fp.method == 0) * denominator)
        n_mod_users = np.sum(numerator * denominator)
        self.results['n_non_users'][self.ti] += n_no_method
        self.results['n_mod_users'][self.ti] += n_mod_users
        self.results['mcpr'][self.ti] += sc.safedivide(n_mod_users, sum(denominator))

        # Track CPR: includes newer ways to conceptualize contraceptive prevalence.
        # Includes women using any method of contraception, including LAM
        numerator = ppl.fp.method != 0
        cpr = np.sum(numerator * denominator)
        self.results['n_users'][self.ti] += cpr
        self.results['cpr'][self.ti] += sc.safedivide(cpr, sum(denominator))

        # Track aCPR
        # Denominator of possible users excludes pregnant women and those not sexually active in the last 4 weeks
        # Used to compare new metrics of contraceptive prevalence and eventually unmet need to traditional mCPR definitions
        denominator = method_age * fecund_age * ppl.female * ~ppl.fp.pregnant * ppl.fp.sexually_active
        numerator = ppl.fp.method != 0
        n_at_risk_non_users = np.sum((ppl.fp.method == 0) * denominator)
        n_at_risk_users = np.sum(numerator * denominator)
        self.results['n_at_risk_non_users'][self.ti] += n_at_risk_non_users
        self.results['n_at_risk_users'][self.ti] += n_at_risk_users
        self.results['acpr'][self.ti] = sc.safedivide(n_at_risk_users, sum(denominator))

        return


class RandomChoice(ContraceptiveChoice):
    """ Randomly choose a method of contraception """
    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)
        self.init_dist = self.pars['method_mix']
        self._method_mix = ss.choice(a=np.arange(1, self.n_methods+1))
        return

    def init_method_dist(self, uids):
        return self.choose_method(uids)

    def choose_method(self, uids, event=None):
        choice_arr = self._method_mix.rvs(uids)
        return choice_arr.astype(int)


class SimpleChoice(RandomChoice):
    """
    Simple choice model where method choice depends on age and previous method.
    Uses location-specific data to set parameters, and needs to be initialized with
    either a location string or a data dictionary.
    """
    def __init__(self, pars=None, location=None, data=None, contra_mod='simple', **kwargs):
        """
        This module can be initialized in several different ways. The pars dictionary includes
        all the parameters needed to run the model, some of which are location-specific and created
        from uploaded csv files. The following options are all valid:
        1. Provide a location string, in which case the relevant data will be loaded automatically
        2. Provide a data dict, which will be used to update the pars dict with the data-derived pars
        3. Provide all parameters directly in the pars dict, including the data-derived ones
        Args:
            pars: ContraPars object or dictionary of parameters
            location: Location string (e.g., 'senegal') to load data from
            data: Data dictionary, if not using location
            contra_mod: Which contraception model to use. Default is 'simple'.
            kwargs: Additional keyword arguments passed to the parent class

        Examples:
            # Initialize with location string
            mod = SimpleChoice(location='senegal')

            # Initialize with data dictionary
            dataloader = fpd.get_dataloader('senegal')
            data = dataloader.load_contra_data('simple')
            mod = SimpleChoice(data=data)

            # Initialize with all parameters directly
            dataloader = fpd.get_dataloader('senegal')
            data = dataloader.load_contra_data('simple')
            pars = fp.ContraPars()
            pars.update(data)
            mod = SimpleChoice(pars=pars)

            # Initialize implicitly within a Sim
            sim = fp.Sim(location='senegal', contra_pars=dict(contra_mod='simple'))
            sim.init()
            mod = sim.connectors.contraception
        """

        super().__init__(pars=pars, **kwargs)

        # Get data if not provided
        if data is None and location is not None:
            dataloader = fpd.get_dataloader(location)
            data = dataloader.load_contra_data(contra_mod, return_data=True)
        self.update_pars(data)
        if self.pars.dur_use_df is not None: self.process_durations()

        self.age_bins = np.sort([fpd.method_age_map[k][1] for k in self.pars.method_choice_pars[0].keys() if k != 'method_idx'])

        return

    def process_durations(self):
        df = self.pars.dur_use_df
        for method in self.methods.values():
            if method.name == 'btl':
                method.dur_use = ss.uniform(low=1000, high=1200)
            else:
                mlabel = method.csv_name

                thisdf = df.loc[df.method == mlabel]
                dist = thisdf.functionform.iloc[0]
                age_ind = sc.findfirst(thisdf.coef.values, 'age_grp_fact(0,18]')

                # Get age factors if they exist for this distribution
                age_factors = None
                if age_ind is not None and age_ind < len(thisdf.estimate.values):
                    age_factors = thisdf.estimate.values[age_ind:]

                if dist in ['lognormal', 'lnorm']:
                    par1 = thisdf.estimate[thisdf.coef == 'meanlog'].values[0]
                    par2 = thisdf.estimate[thisdf.coef == 'sdlog'].values[0]

                elif dist in ['gamma']:
                    par1 = thisdf.estimate[thisdf.coef == 'shape'].values[0]  # shape parameter (log space)
                    par2 = thisdf.estimate[thisdf.coef == 'rate'].values[0]   # rate parameter (log space)

                elif dist == 'llogis':
                    par1 = thisdf.estimate[thisdf.coef == 'shape'].values[0]  # shape parameter (log space)
                    par2 = thisdf.estimate[thisdf.coef == 'scale'].values[0]  # scale parameter (log space)

                elif dist == 'weibull':
                    par1 = thisdf.estimate[thisdf.coef == 'shape'].values[0]  # shape parameter (log space)
                    par2 = thisdf.estimate[thisdf.coef == 'scale'].values[0]  # scale parameter (log space)

                elif dist == 'exponential':
                    par1 = thisdf.estimate[thisdf.coef == 'rate'].values[0]   # rate parameter (log space)

                else:
                    errormsg = f"Duration of use distribution {dist} not recognized"
                    raise ValueError(errormsg)

                method.set_dur_use(dist_type=dist,
                                   par1=par1, par2=par2,
                                   age_factors=age_factors)
        return

    def init_method_dist(self, uids):
        ppl = self.sim.people
        if self.pars.init_dist is not None:
            choice_array = np.zeros(len(uids))

            # Loop over age groups and methods
            for key, (age_low, age_high) in fpd.method_age_map.items():
                this_age_bools = (ppl.age[uids] >= age_low) & (ppl.age[uids] < age_high)
                ppl_this_age = this_age_bools.nonzero()[-1]
                if len(ppl_this_age) > 0:
                    these_probs = self.pars.init_dist[key]
                    these_probs = np.array(these_probs) * self.pars['method_weights']  # Scale by weights
                    these_probs = these_probs/np.sum(these_probs)  # Renormalize
                    self._method_choice_dist.set(a=len(these_probs), p=these_probs)
                    these_choices = self._method_choice_dist.rvs(len(ppl_this_age))  # Choose
                    # Adjust method indexing to correspond to datafile (removing None: Marita to confirm)
                    choice_array[this_age_bools] = np.array(list(self.pars.init_dist.method_idx))[these_choices]
            return choice_array.astype(int)
        else:
            errormsg = f'Distribution of contraceptive choices has not been provided.'
            raise ValueError(errormsg)

    def get_prob_use(self, uids, event=None):
        """
        Return an array of probabilities that each woman will use contraception.
        """
        ppl = self.sim.people
        year = self.t.year

        # Figure out which coefficients to use
        if event is None : p = self.pars.contra_use_pars[0]
        if event == 'pp1': p = self.pars.contra_use_pars[1]
        if event == 'pp6': p = self.pars.contra_use_pars[2]

        # Initialize probability of use
        rhs = np.full_like(ppl.age[uids], fill_value=p.intercept)
        age_bins = np.digitize(ppl.age[uids], self.age_bins)
        for ai, ab in enumerate(self.age_bins):
            rhs[age_bins == ai] += p.age_factors[ai]
            if ai > 1:
                rhs[(age_bins == ai) & ppl.ever_used_contra[uids]] += p.age_ever_user_factors[ai-1]
        rhs[ppl.ever_used_contra[uids]] += p.fp_ever_user

        # The yearly trend
        rhs += (year - self.pars['prob_use_year']) * self.pars['prob_use_trend_par']
        # This parameter can be positive or negative
        rhs += self.pars['prob_use_intercept']
        prob_use = 1 / (1+np.exp(-rhs))

        # Set
        self.pars.p_use.set(p=prob_use)  # Set the probability of use parameter
        return

    def set_dur_method(self, uids, method_used=None):
        """ Time on method depends on age and method """
        ppl = self.sim.people

        dur_method = np.zeros(len(uids), dtype=float)
        if method_used is None: method_used = ppl.fp.method[uids]

        for mname, method in self.methods.items():
            dur_use = method.dur_use
            user_idxs = np.nonzero(method_used == method.idx)[-1]
            users = uids[user_idxs]  # Get the users of this method
            n_users = len(users)

            if n_users:
                if hasattr(dur_use, 'rvs'):
                    # Starsim distribution object
                    dur_method[user_idxs] = dur_use.rvs(users)
                elif sc.isnumber(dur_use):
                    dur_method[user_idxs] = dur_use
                else:
                    errormsg = 'Unrecognized type for duration of use: expecting a Starsim distribution or a number'
                    raise ValueError(errormsg)

        dt = self.t.dt.months
        timesteps_til_update = np.clip(np.round(dur_method/dt), 1, self.pars['max_dur'].years)  # Include a maximum. Durs seem way too high

        return timesteps_til_update

    def choose_method(self, uids, event=None, jitter=1e-4):
        ppl = self.sim.people
        if event == 'pp1': return self.choose_method_post_birth(uids)

        else:
            if event is None:  mcp = self.pars.method_choice_pars[0]
            if event == 'pp6': mcp = self.pars.method_choice_pars[6]

            # Initialize arrays and get parameters
            choice_array = np.zeros(len(uids))

            # Loop over age groups and methods
            for key, (age_low, age_high) in fpd.method_age_map.items():
                match_low_high = (ppl.age[uids] >= age_low) & (ppl.age[uids] < age_high)

                for mname, method in self.methods.items():
                    # Get people of this age who are using this method
                    using_this_method = match_low_high & (ppl.fp.method[uids] == method.idx)
                    switch_iinds = using_this_method.nonzero()[-1]

                    if len(switch_iinds):

                        # Get probability of choosing each method
                        if mname == 'btl':
                            choice_array[switch_iinds] = method.idx  # Continue, can't actually stop this method
                        else:
                            try:
                                these_probs = mcp[key][mname]  # Cannot stay on method
                            except:
                                errormsg = f'Cannot find {key} in method switch for {mname}!'
                                raise ValueError(errormsg)
                            self._jitter_dist.set(scale=jitter)
                            these_probs = [p if p > 0 else p+abs(self._jitter_dist.rvs(1)[0]) for p in these_probs]  # No 0s
                            these_probs = np.array(these_probs) * self.pars['method_weights']  # Scale by weights
                            these_probs = these_probs/sum(these_probs)  # Renormalize
                            self._method_choice_dist.set(a=len(these_probs), p=these_probs)
                            these_choices = self._method_choice_dist.rvs(len(switch_iinds))  # Choose

                            # Adjust method indexing to correspond to datafile (removing None: Marita to confirm)
                            choice_array[switch_iinds] = np.array(list(mcp.method_idx))[these_choices]

        return choice_array.astype(int)

    def choose_method_post_birth(self, uids, jitter=1e-4):
        ppl = self.sim.people
        mcp = self.pars.method_choice_pars[1]
        choice_array = np.zeros(len(uids))

        # Loop over age groups and methods
        for key, (age_low, age_high) in fpd.method_age_map.items():
            match_low_high = (ppl.age[uids] >= age_low) & (ppl.age[uids] < age_high)
            switch_iinds = match_low_high.nonzero()[-1]

            if len(switch_iinds):
                these_probs = mcp[key]
                self._jitter_dist.set(scale=jitter)
                these_probs = [p if p > 0 else p+abs(self._jitter_dist.rvs(1)[0]) for p in these_probs]  # No 0s
                these_probs = np.array(these_probs) * self.pars['method_weights']  # Scale by weights
                these_probs = these_probs/sum(these_probs)  # Renormalize
                self._method_choice_dist.set(a=len(these_probs), p=these_probs)
                these_choices = self._method_choice_dist.rvs(len(switch_iinds))  # Choose
                choice_array[switch_iinds] = np.array(list(mcp.method_idx))[these_choices]

        return choice_array


class StandardChoice(SimpleChoice):
    """
    Default contraceptive choice module.
    Contraceptive choice is based on age, education, wealth, parity, and prior use.
    """
    def __init__(self, pars=None, location=None, data=None, contra_mod='mid', **kwargs):
        super().__init__(pars=pars, location=location, data=data, contra_mod=contra_mod, **kwargs)
        return

    def get_prob_use(self, uids, event=None):
        """
        Return an array of probabilities that each woman will use contraception.
        """
        ppl = self.sim.people
        year = self.t.year

        # Figure out which coefficients to use
        if event is None : p = self.pars.contra_use_pars[0]
        if event == 'pp1': p = self.pars.contra_use_pars[1]
        if event == 'pp6': p = self.pars.contra_use_pars[2]

        # Initialize with intercept
        rhs = np.full_like(ppl.age[uids], fill_value=p.intercept)

        # Add all terms that don't involve age/education level factors
        for term in ['ever_used_contra', 'urban', 'parity', 'wealthquintile']:
            rhs += p[term] * ppl[term][uids]

        # Add age
        int_age = ppl.int_age(uids)
        int_age[int_age < fpd.min_age] = fpd.min_age
        int_age[int_age >= fpd.max_age_preg] = fpd.max_age_preg-1
        dfa = self.pars.age_spline.loc[int_age]
        rhs += p.age_factors[0] * dfa['knot_1'].values + p.age_factors[1] * dfa['knot_2'].values + p.age_factors[2] * dfa['knot_3'].values
        rhs += (p.age_ever_user_factors[0] * dfa['knot_1'].values * ppl.ever_used_contra[uids]
                + p.age_ever_user_factors[1] * dfa['knot_2'].values * ppl.ever_used_contra[uids]
                + p.age_ever_user_factors[2] * dfa['knot_3'].values * ppl.ever_used_contra[uids])

        # Add education levels
        primary = (ppl.edu.attainment[uids] > 1) & (ppl.edu.attainment[uids] <= 6)
        secondary = ppl.edu.attainment[uids] > 6
        rhs += p.edu_factors[0] * primary + p.edu_factors[1] * secondary

        # Add time trend
        rhs += (year - self.pars['prob_use_year'])*self.pars['prob_use_trend_par']
        # This parameter can be positive or negative
        rhs += self.pars['prob_use_intercept']

        # Finish
        prob_use = expit(rhs)
        self.pars.p_use.set(p=prob_use)
        return
