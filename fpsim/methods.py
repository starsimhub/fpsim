"""
Contraceptive methods

Idea:
    A method selector should be a generic class (an intervention?). We want the existing matrix-based
    method and the new duration-based method to both be instances of this.

"""

# %% Imports
import numpy as np
import sciris as sc
import starsim as ss
from scipy.special import expit
from scipy.stats import fisk
from . import defaults as fpd

__all__ = ['Method', 'make_method_list', 'ContraPars', 'make_contra_pars', 'ContraceptiveChoice', 'RandomChoice', 'SimpleChoice', 'StandardChoice']


# %% Base definition of contraceptive methods -- can be overwritten by locations
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


def make_method_list(methods_df):
    """
    Create methods from a DataFrame.
    Args:
        methods_df: DataFrame with columns: name, label, csv_name, efficacy, modern,
    Returns:
        method_list: List of Method objects
    """
    method_list = []

    for idx, row in methods_df.iterrows():
        method = Method(
            name=row['name'],
            label=row['label'],
            csv_name=row['csv_name'],
            idx=idx,  # Use DataFrame index as method index
            efficacy=float(row['efficacy']),
            modern=bool(row['modern']),
        )
        method_list.append(method)

    return method_list


class Fisk(ss.Dist):
    """ Wrapper for scipy's fisk distribution to make it compatible with starsim """
    def __init__(self, c=0.0, scale=1.0, **kwargs):
        super().__init__(distname='fisk', dist=fisk, c=c, scale=scale, **kwargs)
        return


# %% Define parameters
class ContraPars(ss.Pars):
    def __init__(self, **kwargs):
        super().__init__()

        # Methods
        self.methods = None  # Will be populated from methods_df
        self.methods_df = None  # Store the DataFrame for reference

        # Probabilities and choices
        self.p_use = ss.bernoulli(p=0.5)
        self.force_choose = False  # Whether to force non-users to choose a method
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



# %% Define classes to contain information about the way women choose contraception

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

        # Create methods from DataFrame or use default
        if self.pars.methods_df is not None:
            method_list = make_method_list(self.pars.methods_df)
        else:
            raise ValueError('A methods_df must be provided to define contraceptive methods.')

        # Store methods. Note that these can be updated over the course of a simulation, in which case
        # self.methods will change but self.pars.methods_df will not. Always refer to self.methods for
        # the current methods.
        self.methods = ss.ndict(method_list, type=Method)

        # Process pars
        if self.pars.method_weights is None:
            self.pars.method_weights = np.ones(self.n_methods)

        # Validate methods against method_choice_pars
        if self.pars.method_choice_pars is not None:
            self._validate_methods()

        # # Make switching matrix
        # self.switch = Switching(self.methods, create=True)

        self.init_dist = None
        self.data = {}
        
        # Initialize choice distributions for method selection
        # self._method_choice_dist = ss.choice(a=self.n_methods, p=np.ones(self.n_methods)/self.n_methods)
        self._jitter_dist = ss.normal(loc=0, scale=1e-4)

        return

    # ==================================================================================
    # PROPERTIES
    # ==================================================================================
    @property
    def n_options(self):
        """ Number of contraceptive methods defined """
        return len(self.methods)

    @property
    def n_methods(self):
        return len([m for m in self.methods if m != 'none'])

    @property
    def average_dur_use(self):
        av = 0
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

    # ==================================================================================
    # VALIDATION METHODS
    # ==================================================================================
    def _validate_methods(self):
        """
        Validate that methods are consistent with method_choice_pars.
        1. All methods in method_choice_pars must exist in self.methods
        2. Method indices match between methods and method_choice_pars.method_idx
        3. The ordering is consistent
        Raises ValueError with detailed message if validation fails.
        """
        mcp = self.pars.method_choice_pars

        # Get method_idx from method_choice_pars
        mcp_method_idx = None
        for pp in sorted(mcp.keys()):
            if hasattr(mcp[pp], 'method_idx'):
                mcp_method_idx = mcp[pp].method_idx
                break

        if mcp_method_idx is None:
            errormsg = 'method_choice_pars does not contain method_idx array'
            raise ValueError(errormsg)

        # Get methods (excluding 'none' which isn't in switching)
        method_indices = {m.idx: m for m in self.methods.values() if m.name != 'none'}

        # Check that all indices in method_choice_pars exist in methods
        missing_indices = set(mcp_method_idx) - set(method_indices.keys())
        if missing_indices:
            errormsg = (f'Method indices in method_choice_pars not found in methods: {missing_indices}\n'
                       f'Available method indices: {sorted(method_indices.keys())}')
            raise ValueError(errormsg)

        # Check that counts match
        if len(mcp_method_idx) != len(method_indices):
            errormsg = (f'Method count mismatch:\n'
                       f'  method_choice_pars has {len(mcp_method_idx)} methods: {mcp_method_idx}\n'
                       f'  methods has {len(method_indices)} methods (excluding "none"): {sorted(method_indices.keys())}')
            raise ValueError(errormsg)

        # Validate that 'none' has index 0
        if 'none' in self.methods and self.methods['none'].idx != 0:
            errormsg = f'Method "none" must have index 0, got {self.methods["none"].idx}'
            raise ValueError(errormsg)

        # Check for duplicate indices
        all_indices = [m.idx for m in self.methods.values()]
        if len(all_indices) != len(set(all_indices)):
            errormsg = f'Duplicate method indices found'
            raise ValueError(errormsg)

        return

    # ==================================================================================
    # METHOD MANAGEMENT (add_method, get_method, update methods)
    # ==================================================================================
    def get_method_by_label(self, method_label):
        """ Extract method according to its label / long name """
        return_val = None
        for method_name, method in self.methods.items():
            if method.label == method_label:
                return_val = method
        if return_val is None:
            errormsg = f'No method matching {method_label} found.'
            raise ValueError(errormsg)
        return return_val

    def get_method(self, identifier):
        """
        Get a method by name (first) or label (second).
        Args:
            identifier (str): Method name (e.g., 'impl') or label (e.g., 'Implants')
        Returns:
            Method: The matching method object
        """
        # Try by name first
        if identifier in self.methods:
            return self.methods[identifier]

        # Try by label
        for method in self.methods.values():
            if method.label == identifier:
                return method

        # Not found
        available_names = list(self.methods.keys())
        available_labels = [m.label for m in self.methods.values()]
        raise ValueError(f'Method "{identifier}" not found. Available names: {available_names}, labels: {available_labels}')

    def update_efficacy(self, method_label=None, new_efficacy=None):
        method = self.get_method_by_label(method_label)
        method.efficacy = new_efficacy

    def update_duration(self, method_label=None, new_duration=None):
        method = self.get_method_by_label(method_label)
        method.dur_use = new_duration

    def add_method(self, method, copy_from=None):
        """
        Add a new contraceptive method to the simulation.

        Args:
            method: Method object to add (must have name, label, csv_name, efficacy, modern, dur_use)
            copy_from: Optional method name/label to copy switching probabilities from
        """
        # Assign next available index
        max_idx = max(m.idx for m in self.methods.values())
        method.idx = max_idx + 1

        # Add to methods
        method_list = list(self.methods.values()) + [method]
        self.methods = ss.ndict(method_list, type=Method)

        # Extend method_choice_pars
        if self.pars.method_choice_pars is not None:
            if copy_from:
                copy_from_method = self.get_method(copy_from)
                self.extend_method_choice_pars(method.name, copy_from_method.name)
            else:
                self.extend_method_choice_pars(method.name, None)

        # Extend method_weights
        if self.pars.method_weights is not None:
            self.pars.method_weights = np.append(self.pars.method_weights, 1.0)

        # Re-validate after adding
        if self.pars.method_choice_pars is not None:
            self._validate_methods()

        return

    def remove_method(self, method_label):
        errormsg = ('remove_method is not currently functional. See example in test_parameters.py if you want to run a '
                    'simulation with a subset of the standard set of methods. The remove_method logic needs to be'
                    'replaced with something that can remove a method partway through a simulation.')
        raise ValueError(errormsg)

    # ==================================================================================
    # METHOD_CHOICE_PARS MANAGEMENT
    # ==================================================================================
    def _get_pp_keys(self):
        """Get all postpartum keys from method_choice_pars."""
        if self.pars.method_choice_pars is None:
            return []
        return sorted([k for k in self.pars.method_choice_pars.keys()])

    def _get_age_groups(self):
        """Get all age groups from method_choice_pars."""
        if self.pars.method_choice_pars is None:
            return []
        for pp in self._get_pp_keys():
            return [k for k in self.pars.method_choice_pars[pp].keys() if k != 'method_idx']
        return []

    def _to_list(self, value, all_values):
        """Convert None, single value, or list to list."""
        if value is None:
            return all_values
        elif isinstance(value, (list, tuple)):
            return value
        else:
            return [value]

    def make_empty_method_choice_pars(self):
        """
        Create an empty method_choice_pars structure with zero entries.
        Useful for creating new simulations without location data.
        """
        mc = dict()
        for pp in [0, 1, 6]:
            mc[pp] = sc.objdict()
            mc[pp].method_idx = np.array(self.method_idx)

            for age_grp in fpd.method_age_map.keys():
                if pp == 1:  # Postpartum = 1 has direct array (no from_method level)
                    mc[pp][age_grp] = np.zeros(len(self.method_idx))
                else:  # pp = 0 or 6, has from_method level
                    mc[pp][age_grp] = sc.objdict()
                    for from_method in self.methods.keys():
                        mc[pp][age_grp][from_method] = np.zeros(len(self.method_idx))

        self.pars.method_choice_pars = mc
        return mc

    def extend_method_choice_pars(self, new_method_name, copy_from_method=None):
        """
        Extend method_choice_pars to include a new method.

        Args:
            new_method_name (str): Name of the new method being added
            copy_from_method (str or None): Name of existing method to copy probabilities from.
                                            If None, initialize with zeros.
        """
        mcp = self.pars.method_choice_pars
        if mcp is None:
            return

        # Get the new method's index
        new_method = self.methods[new_method_name]
        new_method_idx = new_method.idx

        # Get source method info if copying
        source_idx_in_array = None
        if copy_from_method:
            source_method = self.methods[copy_from_method]
            # Find position of source method in the arrays
            for i, idx in enumerate(mcp[0].method_idx):
                if idx == source_method.idx:
                    source_idx_in_array = i
                    break

            if source_idx_in_array is None:
                errormsg = f'Source method {copy_from_method} not found in method_choice_pars'
                raise ValueError(errormsg)

        # Process each postpartum state
        for pp in mcp.keys():
            pp_dict = mcp[pp]

            # Add new method to method_idx
            pp_dict.method_idx = np.append(pp_dict.method_idx, new_method_idx)

            # Process each age group
            for age_grp in pp_dict.keys():
                if age_grp == 'method_idx':
                    continue

                if pp == 1:
                    # For pp=1, age groups have direct arrays
                    old_probs = pp_dict[age_grp]

                    if copy_from_method:
                        new_prob = old_probs[source_idx_in_array]
                    else:
                        new_prob = 0.0

                    pp_dict[age_grp] = np.append(old_probs, new_prob)
                else:
                    # For pp=0 and pp=6, age groups have from_method dicts
                    age_dict = pp_dict[age_grp]

                    # For each existing from_method, add probability to switch TO the new method
                    for from_method in list(age_dict.keys()):
                        old_probs = age_dict[from_method]

                        if copy_from_method:
                            new_prob = old_probs[source_idx_in_array]
                        else:
                            new_prob = 0.0

                        age_dict[from_method] = np.append(old_probs, new_prob)

                    # Add the new method as a from_method
                    if copy_from_method and copy_from_method in age_dict:
                        # Copy entire switching behavior from source
                        source_probs = age_dict[copy_from_method].copy()
                        # Extend array to include slot for new method
                        source_probs = np.append(source_probs, 0.0)
                        age_dict[new_method_name] = source_probs
                    else:
                        # Initialize with zeros
                        n_methods = len(pp_dict.method_idx)
                        age_dict[new_method_name] = np.zeros(n_methods)

        # Renormalize after extending
        self.renormalize_method_choice_pars()

        return

    def renormalize_method_choice_pars(self):
        """
        Renormalize all probability arrays in method_choice_pars to sum to 1.
        Replaces Switching.renormalize_all()
        """
        mcp = self.pars.method_choice_pars
        if mcp is None:
            return

        for pp in mcp.keys():
            pp_dict = mcp[pp]

            for age_grp in pp_dict.keys():
                if age_grp == 'method_idx':
                    continue

                if pp == 1:
                    probs = pp_dict[age_grp]
                    if probs.sum() > 0:
                        pp_dict[age_grp] = probs / probs.sum()
                else:
                    for from_method in pp_dict[age_grp].keys():
                        probs = pp_dict[age_grp][from_method]
                        if probs.sum() > 0:
                            pp_dict[age_grp][from_method] = probs / probs.sum()
        return

    def set_switching_prob(self, from_method, to_method, value, postpartum=None,
                          age_grp=None, renormalize=False):
        """
        Set switching probability from one method to another.
        Args:
            from_method (str): Method name to switch from
            to_method (str): Method name to switch to
            value (float): Probability value
            postpartum (int or list or None): Postpartum state(s) (0, 1, or 6). None = all.
            age_grp (str or list or None): Age group(s). None = all.
            renormalize (bool): Whether to renormalize after setting
        """
        mcp = self.pars.method_choice_pars
        if mcp is None:
            errormsg = 'method_choice_pars is None, cannot set switching probabilities'
            raise ValueError(errormsg)

        if to_method not in self.methods:
            raise ValueError(f"to_method '{to_method}' not found")

        # Get index of to_method in the arrays
        to_method_obj = self.methods[to_method]
        to_idx = None
        for i, idx in enumerate(mcp[0].method_idx):
            if idx == to_method_obj.idx:
                to_idx = i
                break

        if to_idx is None:
            raise ValueError(f"to_method '{to_method}' not found in method_choice_pars")

        # Convert to lists for iteration
        pp_list = self._to_list(postpartum, self._get_pp_keys())
        age_list = self._to_list(age_grp, self._get_age_groups())

        for pp in pp_list:
            for age in age_list:
                if pp == 1:
                    # Direct array access for postpartum=1
                    if from_method != 'birth':
                        raise ValueError(f"For postpartum=1, from_method must be 'birth', got '{from_method}'")
                    mcp[pp][age][to_idx] = value
                    if renormalize:
                        self._renormalize_row(pp, age, from_method)
                else:
                    # Need from_method for pp=0 or 6
                    if from_method not in self.methods:
                        raise ValueError(f"from_method '{from_method}' not found")
                    mcp[pp][age][from_method][to_idx] = value
                    if renormalize:
                        self._renormalize_row(pp, age, from_method)

    def get_switching_prob(self, from_method, to_method, postpartum, age_grp):
        """
        Get switching probability from one method to another.
        Args:
            from_method (str): Method name to switch from
            to_method (str): Method name to switch to
            postpartum (int): Postpartum state (0, 1, or 6)
            age_grp (str): Age group

        Returns:
            float: Probability value
        """
        mcp = self.pars.method_choice_pars
        if mcp is None:
            return 0.0

        if to_method not in self.methods:
            raise ValueError(f"to_method '{to_method}' not found")

        # Get index of to_method in the arrays
        to_method_obj = self.methods[to_method]
        to_idx = None
        for i, idx in enumerate(mcp[postpartum].method_idx):
            if idx == to_method_obj.idx:
                to_idx = i
                break

        if to_idx is None:
            raise ValueError(f"to_method '{to_method}' not found in method_choice_pars")

        if postpartum == 1:
            if from_method != 'birth':
                raise ValueError(f"For postpartum=1, from_method must be 'birth', got '{from_method}'")
            return mcp[postpartum][age_grp][to_idx]
        else:
            if from_method not in self.methods:
                raise ValueError(f"from_method '{from_method}' not found")
            return mcp[postpartum][age_grp][from_method][to_idx]

    def copy_switching_probs(self, from_method_source, to_method_source,
                            from_method_dest, to_method_dest,
                            postpartum=None, age_grp=None, renormalize=False):
        """
        Copy switching probability from one method pair to another.
        Example: Copy the probability of switching from pill->iud
                 to the probability of switching from condom->implant
        """
        pp_list = self._to_list(postpartum, self._get_pp_keys())
        age_list = self._to_list(age_grp, self._get_age_groups())

        for pp in pp_list:
            for age in age_list:
                value = self.get_switching_prob(from_method_source, to_method_source, pp, age)
                self.set_switching_prob(from_method_dest, to_method_dest, value, pp, age,
                                       renormalize=renormalize)

    def copy_switching_to_method(self, source_to_method, dest_to_method,
                                 postpartum=None, age_grp=None, renormalize=False):
        """
        Copy all switching probabilities TO a method (entire column).
        Example: Copy all probabilities of switching to IUD
                 to be the probabilities of switching to implant.
        """
        pp_list = self._to_list(postpartum, self._get_pp_keys())
        age_list = self._to_list(age_grp, self._get_age_groups())

        for pp in pp_list:
            for age in age_list:
                if pp == 1:
                    # Copy from birth
                    value = self.get_switching_prob('birth', source_to_method, pp, age)
                    self.set_switching_prob('birth', dest_to_method, value, pp, age, renormalize=renormalize)
                else:
                    # Copy from all methods (excluding permanent methods like btl that can't switch)
                    mcp = self.pars.method_choice_pars
                    for from_method in self.methods.keys():
                        # Check if this method can switch (has a row in the matrix)
                        if from_method in mcp[pp][age]:
                            value = self.get_switching_prob(from_method, source_to_method, pp, age)
                            self.set_switching_prob(from_method, dest_to_method, value, pp, age, renormalize=renormalize)

    def copy_switching_from_method(self, source_from_method, dest_from_method,
                                   postpartum=None, age_grp=None, renormalize=False):
        """
        Copy all switching probabilities FROM a method (entire row).
        Example: Copy all probabilities of switching from pill to other methods
                 to be the probabilities of switching from implant to other methods.
                 This makes implant users switch like pill users.
        """
        pp_list = self._to_list(postpartum, self._get_pp_keys())
        age_list = self._to_list(age_grp, self._get_age_groups())

        for pp in pp_list:
            if pp == 1:
                continue  # Skip postpartum=1 as it has no from_method dimension

            for age in age_list:
                # Copy entire row at once, then renormalize once if needed
                for to_method in self.methods.keys():
                    value = self.get_switching_prob(source_from_method, to_method, pp, age)
                    # Don't renormalize on each set, only after all values copied
                    self.set_switching_prob(dest_from_method, to_method, value, pp, age, renormalize=False)

                # Renormalize once after all values are set
                if renormalize:
                    self._renormalize_row(pp, age, dest_from_method)

    def get_switching_matrix(self, postpartum, from_method=None):
        """
        Get switching matrix for a particular postpartum status across all age groups.
        Args:
            postpartum (int): Postpartum state (0, 1, or 6)
            from_method (str or None): Method name for pp=0 or 6

        Returns:
            dict: Dictionary keyed by age group with probability arrays
        """
        mcp = self.pars.method_choice_pars
        if mcp is None:
            return {}

        if postpartum not in self._get_pp_keys():
            raise ValueError(f"postpartum={postpartum} not found. "
                           f"Available: {self._get_pp_keys()}")

        if postpartum == 1:
            # For postpartum=1, return direct arrays for all age groups
            result = {}
            for age_grp in self._get_age_groups():
                result[age_grp] = mcp[postpartum][age_grp]
            return result
        else:
            # For postpartum=0 or 6, need from_method
            if from_method is None:
                raise ValueError(f"from_method is required when postpartum={postpartum}")

            if from_method not in self.methods:
                raise ValueError(f"from_method '{from_method}' not found. "
                               f"Available: {list(self.methods.keys())}")

            result = {}
            for age_grp in self._get_age_groups():
                if from_method in mcp[postpartum][age_grp]:
                    result[age_grp] = mcp[postpartum][age_grp][from_method]
                else:
                    # Method doesn't have a row (e.g., btl can't switch)
                    result[age_grp] = None

            return result

    def _renormalize_row(self, pp, age_grp, from_method):
        """
        Renormalize a row to sum to 1.
        Helper for switching probability operations.
        """
        mcp = self.pars.method_choice_pars
        if mcp is None:
            return

        if pp == 1:
            row = mcp[pp][age_grp]
        else:
            if from_method not in mcp[pp][age_grp]:
                return  # Method can't switch (e.g., btl)
            row = mcp[pp][age_grp][from_method]

        row_sum = row.sum()
        if row_sum > 0:
            row /= row_sum

    # ==================================================================================
    # CORE STARSIM MODULE METHODS, called directly within the Starsim loop
    # ==================================================================================
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
        self.init_dist = np.array([1/self.n_methods]*self.n_methods)
        self.method_mix = ss.choice(a=np.arange(1, self.n_methods+1))
        return

    def init_method_dist(self, uids):
        return self.choose_method(uids)

    def choose_method(self, uids, event=None):
        choice_arr = self.method_mix.rvs(uids)
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
