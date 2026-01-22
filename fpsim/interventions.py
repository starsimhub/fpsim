'''
Specify the core interventions available in FPsim. Other interventions can be
defined by the user by inheriting from these classes.
'''
import numpy as np
import sciris as sc
import starsim as ss
from . import utils as fpu
from . import methods as fpm

#>> Generic intervention classes

__all__ = ['change_par', 'update_methods', 'add_method', 'change_people_state', 
           'change_initiation_prob', 'change_initiation', 'intervention_from_json']


#%% Helper functions for JSON serialization

def _make_repr(obj, skip_none=True, skip_defaults=None):
    """
    Create a recreatable string representation of an intervention.
    
    Args:
        obj: The intervention object
        skip_none: Whether to skip None values (default True)
        skip_defaults: Dict of param names to default values to skip
        
    Returns:
        str: A string like "fp.change_par(par='exposure_factor', years=[2000], vals=[0.5])"
    """
    skip_defaults = skip_defaults or {}
    class_name = obj.__class__.__name__
    
    # Get parameters from the object
    if hasattr(obj, 'pars') and obj.pars is not None:
        pars = dict(obj.pars)
    else:
        pars = {}
    
    # Also check for direct attributes that aren't in pars
    for attr in ['par', 'years', 'vals', 'verbose', 'year', 'method', 'copy_from']:
        if hasattr(obj, attr) and attr not in pars:
            pars[attr] = getattr(obj, attr)
    
    # Build parameter string
    par_strs = []
    for k, v in pars.items():
        # Skip None values if requested
        if skip_none and v is None:
            continue
        # Skip default values
        if k in skip_defaults and v == skip_defaults[k]:
            continue
        # Format the value
        if isinstance(v, str):
            par_strs.append(f"{k}='{v}'")
        else:
            par_strs.append(f"{k}={v!r}")
    
    return f"fp.{class_name}({', '.join(par_strs)})"


def intervention_from_json(json_data, module=None):
    """
    Recreate an intervention from JSON data.
    
    Args:
        json_data: A dict or objdict from to_json(), or a JSON string
        module: The module containing intervention classes (default: fpsim.interventions)
        
    Returns:
        An intervention object
        
    **Example**::
    
        intv = fp.update_methods(year=2020, eff={'Injectables': 0.99})
        json_data = intv.to_json()
        intv2 = fp.intervention_from_json(json_data)
    """
    import json as json_module
    
    # Parse JSON string if needed
    if isinstance(json_data, str):
        json_data = json_module.loads(json_data)
    
    # Convert to dict if it's an objdict
    if hasattr(json_data, 'to_dict'):
        json_data = dict(json_data)
    
    # Get the intervention type
    intv_type = json_data.get('type') or json_data.get('which')
    if intv_type is None:
        raise ValueError("JSON data must have 'type' or 'which' key specifying the intervention class")
    
    # Get the parameters
    pars = json_data.get('pars', {})
    if hasattr(pars, 'to_dict'):
        pars = dict(pars)
    
    # Filter out None values
    pars = {k: v for k, v in pars.items() if v is not None}
    
    # Get the intervention class
    if module is None:
        import fpsim.interventions as module
    
    if not hasattr(module, intv_type):
        raise ValueError(f"Unknown intervention type: {intv_type}")
    
    intv_class = getattr(module, intv_type)
    
    # Create and return the intervention
    return intv_class(**pars)


class change_par(ss.Intervention):
    '''
    Change a parameter at a specified point in time.

    Args:
        par   (str): the parameter to change
        years (float/arr): the year(s) at which to apply the change
        vals  (any): a value or list of values to change to (if a list, must have the same length as years); or a dict of year:value entries

    If any value is ``'reset'``, reset to the original value.

    **Example**::

        ec0 = fp.change_par(par='exposure_factor', years=[2000, 2010], vals=[0.0, 2.0]) # Reduce exposure factor
        ec0 = fp.change_par(par='exposure_factor', vals={2000:0.0, 2010:2.0}) # Equivalent way of writing
        sim = fp.Sim(interventions=ec0).run()
    '''
    def __init__(self, par, years=None, vals=None, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.par   = par
        self.verbose = verbose
        if isinstance(years, dict): # Swap if years is supplied as a dict, so can be supplied first
            vals = years
        if vals is None:
            errormsg = 'Values must be supplied'
            raise ValueError(errormsg)
        if isinstance(vals, dict):
            years = sc.dcp(list(vals.keys()))
            vals = sc.dcp(list(vals.values()))
        else:
            if years is None:
                errormsg = 'If vals is not supplied as a dict, then year(s) must be supplied'
                raise ValueError(errormsg)
            else:
                years = sc.toarray(sc.dcp(years))
                vals = sc.dcp(vals)
                if sc.isnumber(vals):
                    vals = sc.tolist(vals) # We want to be careful not to take something that might already be an array and interpret different values as years
                n_years = len(years)
                n_vals = len(vals)
                if n_years != n_vals:
                    errormsg = f'Number of years ({n_years}) does not match number of values ({n_vals})'
                    raise ValueError(errormsg)

        self.years = years
        self.vals = vals

        return

    def __repr__(self):
        """Return a recreatable string representation."""
        years_repr = list(self.years) if hasattr(self.years, '__iter__') else self.years
        vals_repr = list(self.vals) if hasattr(self.vals, '__iter__') else self.vals
        parts = [f"par='{self.par}'", f"years={years_repr}", f"vals={vals_repr}"]
        if self.verbose:
            parts.append("verbose=True")
        return f"fp.change_par({', '.join(parts)})"

    def init_pre(self, sim):
        super().init_pre(sim)

        # Validate parameter name
        if self.par not in sim.pars.fp:
            errormsg = f'Parameter "{self.par}" is not a valid sim parameter'
            raise ValueError(errormsg)

        # Validate years and values
        years = self.years
        min_year = min(years)
        max_year = max(years)
        if min_year < sim.pars.start:
            errormsg = f'Intervention start {min_year} is before the start of the simulation'
            raise ValueError(errormsg)
        if max_year > sim.pars.stop:
            errormsg = f'Intervention end {max_year} is after the end of the simulation'
            raise ValueError(errormsg)
        if years != sorted(years):
            errormsg = f'Years {years} should be monotonic increasing'
            raise ValueError(errormsg)

        # Convert intervention years to sim timesteps
        self.counter = 0
        self.inds = sc.autolist()
        for y in years:
            self.inds += sc.findnearest(sim.t.yearvec, y)

        # Store original value
        self.orig_val = sc.dcp(sim.pars.fp[self.par])

        return

    def step(self):
        sim = self.sim
        if len(self.inds) > self.counter:
            ind = self.inds[self.counter] # Find the current index
            if sim.ti == ind: # Check if the current timestep matches
                curr_val = sc.dcp(sim.pars.fp[self.par])
                val = self.vals[self.counter]
                if val == 'reset':
                    val = self.orig_val
                sim.pars.fp[self.par] = val  # Update the parameter value -- that's it!
                if self.verbose:
                    label = f'Sim "{sim.label}": ' if sim.label else ''
                    print(f'{label}On {sim.t.year}, change {self.counter+1}/{len(self.inds)} applied: "{self.par}" from {curr_val} to {sim.pars.fp[self.par]}')
                self.counter += 1
        return

    def finalize(self):
        # Check that all changes were applied
        n_counter = self.counter
        n_vals = len(self.vals)
        if n_counter != n_vals:
            errormsg = f'Not all values were applied ({n_vals} ≠ {n_counter})'
            raise RuntimeError(errormsg)
        super().finalize()
        return


class add_method(ss.Intervention):
    """
    Intervention to add a new contraceptive method to the simulation at a specified time.
   
    Args:
        year (float): The year at which to activate the new method
        method (Method, optional): A Method object defining the new contraceptive method.
            If None, the method will be copied from the source method (specified by ``copy_from``).
        method_pars (dict, optional): Dictionary of parameters to update the method attributes.
            If provided, these values will override corresponding attributes in the method object
            (whether it was provided directly or copied from source). If None, defaults to empty dict.
        copy_from (str): Name of the existing method to copy switching probabilities from.
            Also used as the source method when ``method=None``.
        split_shares (float, optional): If provided, % who would have chosen the 'copy_from' method 
            and now choose the new method
        verbose (bool): Whether to print messages when method is activated (default True)
    
    **Examples**::
    
        # Using a Method object directly
        new_method = fp.Method(name='new_impl', label='New Implant', efficacy=0.999, 
                               dur_use=ss.lognorm_ex(ss.years(3), ss.years(0.5)), modern=True)
        intv = fp.add_method(year=2010, method=new_method, copy_from='impl')
        
        # Copying from source method (method=None, method_pars=None)
        # Creates a copy of 'impl' with name 'impl_copy'
        intv = fp.add_method(year=2010, copy_from='impl')
        
        # Copying from source and overriding properties
        intv = fp.add_method(year=2010, method_pars={'name': 'new_inj', 'efficacy': 0.995}, 
                            copy_from='inj')
        
        # Using method object and overriding properties with method_pars
        base_method = fp.Method(name='new_method', efficacy=0.90)
        intv = fp.add_method(year=2010, method=base_method, 
                            method_pars={'efficacy': 0.998}, copy_from='impl')
    """
    
    def __init__(self, year=None, method=None, method_pars=None, copy_from=None, split_shares=None, verbose=True, **kwargs):
        super().__init__(**kwargs)
        
        # Validate inputs
        if year is None:
            raise ValueError('Year must be specified for add_method intervention')
        if copy_from is None:
            raise ValueError('copy_from must specify an existing method name to copy switching behavior from')
        
        self.year = year
        self.method = method
        # Convert None to empty dict to simplify logic later
        self.method_pars = method_pars if method_pars is not None else {}
        self.copy_from = copy_from
        self.verbose = verbose
        self.split_shares = split_shares
        self.activated = False
        self._method_idx = None  # Will be set after method is added
        
        return

    def __repr__(self):
        """Return a recreatable string representation."""
        parts = [f"year={self.year}", f"copy_from='{self.copy_from}'"]
        
        # Include method if provided
        if self.method is not None:
            method_repr = f"<Method '{self.method.name}'>"
            parts.append(f"method={method_repr}")
        
        # Include method_pars if provided (not empty)
        if self.method_pars:
            parts.append(f"method_pars={self.method_pars!r}")
        
        # Include split_shares if provided
        if self.split_shares is not None:
            parts.append(f"split_shares={self.split_shares}")
        
        # Include verbose only if False (default is True)
        if not self.verbose:
            parts.append("verbose=False")
        
        return f"fp.add_method({', '.join(parts)})"

    def init_pre(self, sim):
        """
        Initialize the intervention before the simulation starts.
        This registers the new method but does not activate it yet.
        """
        super().init_pre(sim)
        
        # Validate year is within simulation range
        if not (sim.pars.start <= self.year <= sim.pars.stop):
            raise ValueError(f'Intervention year {self.year} must be between {sim.pars.start} and {sim.pars.stop}')

        # Resolve copy_from to method name (supports both name and label)
        # Get source_method first so it's available for copying below
        cm = sim.connectors.contraception
        source_method = cm.get_method(self.copy_from)
        self._copy_from_name = source_method.name  # Store resolved name for use in step()
        
        # Handle the 2 cases:
        # Case 1: method=None --> copy source method
        # Case 2: method!=None --> use provided method
        if self.method is None:
            # Case 1: Copy source method to create base method
            self.method = sc.dcp(source_method)
            # If name not provided in method_pars, append '_copy' to make it unique
            if 'name' not in self.method_pars:
                self.method.name += '_copy'
        
        # Case 2: method!=None, use it as-is (no action needed)
        
        # Update method attributes with method_pars
        # This works whether or not method_pars were provided (empty dict if None in __init__)
        # If method_pars contains attributes, they will override corresponding method attributes
        for mp, mpar in self.method_pars.items():
            setattr(self.method, mp, mpar)
        
        # Add the new method to the contraception module, extending the switching probabilities
        # with zeros and resetting init_dist and method_weights
        cm.add_method(self.method)
        self._method_idx = cm.methods[self.method.name].idx
        
        # Resize the method_mix array in fpmod to accommodate the new method
        fp_mod = sim.connectors.fp
        old_mix = fp_mod.method_mix
        new_mix = np.zeros((cm.n_options, sim.t.npts))
        new_mix[:old_mix.shape[0], :] = old_mix
        fp_mod.method_mix = new_mix
        
        if self.verbose:
            print(f'Registered new method "{self.method.name}" (idx={self._method_idx}), will activate in year {self.year}')
        
        return

    def step(self):
        """
        At the specified year, copy switching probabilities to make the method available.
        """
        sim = self.sim
        
        # Check if we've reached the activation year and haven't already activated
        if not self.activated and sim.t.year >= self.year:
            self.activated = True
            
            if self.verbose:
                print(f'Activating new contraceptive method "{self.method.name}" in year {sim.t.year:.1f}')
            
            # Copy switching probabilities
            cm = sim.connectors.contraception
            cm.copy_switching_to_method(
                source_to_method=self._copy_from_name,
                dest_to_method=self.method.name,
                split_shares=self.split_shares
            )
            cm.copy_switching_from_method(
                source_from_method=self._copy_from_name,
                dest_from_method=self.method.name,
            )

            # Renormalize all probabilities
            cm.renormalize_method_choice_pars()
        
        return
    
    def finalize(self):
        """
        Report summary statistics about the new method usage.
        """
        super().finalize()
        
        if self.verbose:
            sim = self.sim
            fp_mod = sim.connectors.fp
            
            # Get final method usage for the new method
            final_usage = fp_mod.method_mix[self._method_idx, -1] if self._method_idx < fp_mod.method_mix.shape[0] else 0
            
            # Count current users
            n_users = np.sum(fp_mod.method == self._method_idx)
            
            print(f'add_method finalized: "{self.method.name}" has {n_users} users ({final_usage*100:.2f}% of method mix)')
        
        return


class change_people_state(ss.Intervention):
    """
    Intervention to modify values of a People's boolean state at one specific
    point in time.

    Args:
        state_name  (string): name of the People's state that will be modified
        new_val     (bool, float): the new state value eligible people will have
        years       (list, float): The year we want to start the intervention.
                     if years is None, uses start and end years of sim as defaults
                     if years is a number or a list with a single element, eg, 2000.5, or [2000.5],
                     this is interpreted as the start year of the intervention, and the
                     end year of intervention will be the end of the simulation
        eligibility (inds/callable): indices OR callable that returns inds
        prop        (float): a value between 0 and 1 indicating the x% of eligible people
                     who will have the new state value
        annual      (bool): whether the increase, prop, represents a "per year" increase, or per time step

    """

    def __init__(self, state_name, new_val, years=None, eligibility=None, prop=1.0, annual=False, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            state_name=state_name,
            new_val=new_val,
            years=years,
            eligibility=eligibility,
            prop=prop,
            annual=annual
        )
        self.module_name = None
        if '.' in state_name:
            self.module_name, self.pars.state_name = state_name.split('.')
        self.annual_perc = None
        return

    def __repr__(self):
        """Return a recreatable string representation."""
        return _make_repr(self, skip_defaults={'prop': 1.0, 'annual': False})

    def init_pre(self, sim):
        super().init_pre(sim)
        self._validate_pars()

        # Lastly, adjust the probability by the sim's timestep, if it's an annual probability
        if self.pars.annual:
            # per timestep/monthly growth rate or perc of eligible women who will be made to choose contraception
            self.annual_perc = self.pars.prop
            self.pars.prop = ((1 + self.annual_perc) ** sim.dt)-1
        # Validate years and values
        if self.pars.years is None:
            # f'Intervention start and end years not provided. Will use sim start an end years'
            self.pars.years = [sim.pars['start'], sim.pars['stop']]
        if sc.isnumber(self.pars.years) or len(self.pars.years) == 1:
            self.pars.years = sc.promotetolist(self.pars.years)
            # Assumes that start year has been specified, append end of the simulation as end year of the intervention
            self.pars.years.append(sim.pars['stop'])

        min_year = min(self.pars.years)
        max_year = max(self.pars.years)
        if min_year < sim.pars['start']:
            errormsg = f'Intervention start {min_year} is before the start of the simulation.'
            raise ValueError(errormsg)
        if max_year > sim.pars['stop']:
            errormsg = f'Intervention end {max_year} is after the end of the simulation.'
            raise ValueError(errormsg)
        if self.pars.years != sorted(self.pars.years):
            errormsg = f'Years {self.pars.years} should be monotonically increasing'
            raise ValueError(errormsg)

        return

    def _validate_pars(self):
        # Validation
        if self.pars.state_name is None:
            errormsg = 'A state name must be supplied.'
            raise ValueError(errormsg)
        if self.pars.new_val is None:
            errormsg = 'A new value must be supplied.'
            raise ValueError(errormsg)
        if self.pars.eligibility is None:
            errormsg = 'Eligibility needs to be provided'
            raise ValueError(errormsg)
        return

    def check_eligibility(self):
        """
        Return an array of uids of agents eligible
        """
        if callable(self.pars.eligibility):
            eligible_uids = self.pars.eligibility(self.sim)
        elif sc.isarray(self.pars.eligibility):
            eligible_uids = self.pars.eligibility
        else:
            errormsg = 'Eligibility must be a function or an array of uids'
            raise ValueError(errormsg)

        return ss.uids(eligible_uids)

    def step(self):
        if self.pars.years[0] <= self.sim.t.year <= self.pars.years[1]:  # Inclusive range
            eligible_uids = self.check_eligibility()
            if self.module_name is not None:
                self.sim.people[self.module_name][self.pars.state_name][eligible_uids] = self.pars.new_val
            else:
                self.sim.people[self.pars.state_name][eligible_uids] = self.pars.new_val
        return


class update_methods(ss.Intervention):
    """
    Intervention to modify method efficacy and/or switching matrix.

    Args:
        year (float): The year we want to change the method.

        eff (dict):
            An optional key for changing efficacy; its value is a dictionary with the following schema:
                {method: efficacy}
                    Where method is the name of the contraceptive method to be changed,
                    and efficacy is a number with the efficacy

        dur_use (dict):
            Optional key for changing the duration of use; its value is a dictionary with the following schema:
                {method: dur_use}
                    Where method is the method to be changed, and dur_use is a dict representing a distribution, e.g.
                    dur_use = {'Injectables: dict(dist='lognormal', par1=a, par2=b)}

        p_use (float): probability of using any form of contraception
        method_mix (list/arr): probabilities of selecting each form of contraception
        
        probs (list): A list of dictionaries for modifying switching probabilities. Each dict can have:
            source (str): the source method (method switching FROM), or 'none' for initiation
            dest   (str): the destination method (method switching TO), or 'none' for discontinuation
            method (str): shorthand - if specified alone with init_*/discont_*, sets source/dest automatically
            factor (float): multiply existing probability by this amount
            value  (float): set probability to this absolute value (mutually exclusive with factor)
            init_factor (float): multiply initiation probability (none→method) by this amount
            init_value  (float): set initiation probability to this value
            discont_factor (float): multiply discontinuation probability (method→none) by this amount
            discont_value  (float): set discontinuation probability to this value
            copy_from (str): copy probabilities from this method
            ages   (str/list): age group(s) to target, e.g. '<18', '18-20', ['20-25', '25-35']
            matrix (int/str): postpartum state - 0/'annual', 1/'pp1', 6/'pp6' (default: 0)

    **Examples**::

        # Double switching probability from Pill to IUDs
        fp.update_methods(year=2020, probs=[{'source': 'Pill', 'dest': 'IUDs', 'factor': 2.0}])

        # Set initiation rate for Injectables to 0.1
        fp.update_methods(year=2020, probs=[{'method': 'Injectables', 'init_value': 0.1}])

        # Increase discontinuation of Implants by 50% for young women
        fp.update_methods(year=2020, probs=[{
            'method': 'Implants', 
            'discont_factor': 1.5, 
            'ages': ['<18', '18-20']
        }])

        # Copy switching behavior from Implants to a new method
        fp.update_methods(year=2020, probs=[{'dest': 'NewMethod', 'copy_from': 'Implants'}])

    """

    def __init__(self, year, eff=None, dur_use=None, p_use=None, method_mix=None, 
                 method_choice_pars=None, probs=None, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            year=year,
            eff=eff,
            dur_use=dur_use,
            p_use=p_use,
            method_mix=method_mix,
            method_choice_pars=method_choice_pars,
            probs=probs,
            verbose=verbose
        )

        self.applied = False
        return

    def __repr__(self):
        """Return a recreatable string representation."""
        return _make_repr(self, skip_defaults={'verbose': False})

    def init_pre(self, sim):
        super().init_pre(sim)
        self._validate()
        par_name = None
        if self.pars.p_use is not None and isinstance(sim.connectors.contraception, fpm.SimpleChoice):
            par_name = 'p_use'
        if self.pars.method_mix is not None and isinstance(sim.connectors.contraception, fpm.SimpleChoice):
            par_name = 'method_mix'

        if par_name is not None:
            errormsg = (
                f"Contraceptive module  {type(sim.connectors.contraception)} does not have `{par_name}` parameter. "
                f"For this type of module, the probability of contraceptive use depends on people attributes and can't be reset using this intervention.")
            print(errormsg)

        return

    def _validate(self):
        # Validation
        if self.pars.year is None:
            errormsg = 'A year must be supplied'
            raise ValueError(errormsg)
        has_input = any([
            self.pars.eff is not None,
            self.pars.dur_use is not None,
            self.pars.p_use is not None,
            self.pars.method_mix is not None,
            self.pars.method_choice_pars is not None,
            self.pars.probs is not None,
        ])
        if not has_input:
            errormsg = 'At least one of eff, dur_use, p_use, method_mix, method_choice_pars, or probs must be supplied'
            raise ValueError(errormsg)
        
        # Validate probs entries if provided
        if self.pars.probs is not None:
            self._validate_probs()
        return

    def _validate_probs(self):
        """Validate the probs parameter entries."""
        probs = sc.tolist(self.pars.probs)
        
        # Valid keys for probs entries
        valid_keys = {
            'source', 'dest', 'method', 'factor', 'value',
            'init_factor', 'init_value', 'discont_factor', 'discont_value',
            'copy_from', 'ages', 'matrix'
        }
        
        # Map string matrix names to numeric values
        matrix_map = {'annual': 0, 'pp1': 1, 'pp6': 6, 0: 0, 1: 1, 6: 6}
        
        for i, entry in enumerate(probs):
            entry = sc.dcp(entry)
            
            # Check for invalid keys
            invalid_keys = set(entry.keys()) - valid_keys
            if invalid_keys:
                errormsg = f'probs[{i}]: Invalid keys {invalid_keys}. Valid keys are: {valid_keys}'
                raise ValueError(errormsg)
            
            # Get modification type counts
            has_factor = entry.get('factor') is not None
            has_value = entry.get('value') is not None
            has_init_factor = entry.get('init_factor') is not None
            has_init_value = entry.get('init_value') is not None
            has_discont_factor = entry.get('discont_factor') is not None
            has_discont_value = entry.get('discont_value') is not None
            has_copy_from = entry.get('copy_from') is not None
            
            # Count how many modification types are specified
            n_mods = sum([
                has_factor or has_value,
                has_init_factor or has_init_value,
                has_discont_factor or has_discont_value,
                has_copy_from
            ])
            
            if n_mods != 1:
                errormsg = (f'probs[{i}]: Must specify one of: factor, value, init_factor, init_value, discont_factor, discont_value, or copy_from' 
                            if n_mods == 0 else f'probs[{i}]: Can only specify one modification type (factor/value, init_*, discont_*, or copy_from)')
                raise ValueError(errormsg)
            
            # Check factor/value mutual exclusivity
            # Example valid entries:
            # probs = [    
            #      {'source': 'None', 'dest': 'Pill', 'factor': 1.5},    # multiply switching prob by 1.5    
            #      {'method': 'IUDs', 'init_value': 0.10},               # set initiation prob to 10%    
            #      {'method': 'Condoms', 'copy_from': 'Pill'},           # copy Pill's probs to Condoms]
            # Example invalid entry:
            # {'method': 'Pill', 'factor': 2.0, 'init_factor': 1.5}  # ✗ can't mix factor AND init_factor

            if has_factor and has_value:
                errormsg = f'probs[{i}]: Cannot specify both factor and value'
                raise ValueError(errormsg)
            if has_init_factor and has_init_value:
                errormsg = f'probs[{i}]: Cannot specify both init_factor and init_value'
                raise ValueError(errormsg)
            if has_discont_factor and has_discont_value:
                errormsg = f'probs[{i}]: Cannot specify both discont_factor and discont_value'
                raise ValueError(errormsg)
            
            # Check source/dest vs method usage
            has_source = entry.get('source') is not None
            has_dest = entry.get('dest') is not None
            has_method = entry.get('method') is not None
            
            if has_method and (has_source or has_dest):
                errormsg = f'probs[{i}]: Cannot specify both "method" and "source"/"dest"'
                raise ValueError(errormsg)
            
            # For factor/value, need source and dest (or method for init/discont)
            # Exception: for matrix='pp1' or matrix=1, only dest is required (source is implicitly 'birth')
            if has_factor or has_value:
                matrix = entry.get('matrix', 0)
                is_pp1 = matrix in ['pp1', 1]
                if is_pp1:
                    # For postpartum=1, only dest is required
                    if not has_dest and not has_method:
                        errormsg = f'probs[{i}]: Must specify dest (or method) when using factor/value with matrix=pp1'
                        raise ValueError(errormsg)
                else:
                    if not (has_source and has_dest) and not has_method:
                        errormsg = f'probs[{i}]: Must specify source and dest (or method) when using factor/value'
                        raise ValueError(errormsg)
            
            # Validate matrix value
            matrix = entry.get('matrix')
            if matrix is not None and matrix not in matrix_map:
                errormsg = f'probs[{i}]: Invalid matrix "{matrix}". Must be one of: {list(matrix_map.keys())}'
                raise ValueError(errormsg)
        
        return

    def _apply_probs(self, cm):
        """
        Apply probability modifications to the switching matrix.
        
        Args:
            cm: The contraception module (sim.connectors.contraception)
        """
        probs = sc.tolist(self.pars.probs)
        sw = cm.switch
        
        # Map string matrix names to numeric postpartum values
        matrix_map = {'annual': 0, 'pp1': 1, 'pp6': 6, 0: 0, 1: 1, 6: 6}
        
        for entry in probs:
            entry = sc.dcp(entry)
            
            # Extract common parameters
            ages = entry.get('ages')
            matrix = entry.get('matrix', 0)  # Default to annual/non-postpartum
            postpartum = matrix_map.get(matrix, 0)
            
            # Extract modification parameters
            source = entry.get('source')
            dest = entry.get('dest')
            method = entry.get('method')
            factor = entry.get('factor')
            value = entry.get('value')
            init_factor = entry.get('init_factor')
            init_value = entry.get('init_value')
            discont_factor = entry.get('discont_factor')
            discont_value = entry.get('discont_value')
            copy_from = entry.get('copy_from')
            
            # Resolve method shorthand for init/discont
            if method is not None:
                if init_factor is not None or init_value is not None:
                    source = 'none'
                    dest = method
                    factor = init_factor
                    value = init_value
                elif discont_factor is not None or discont_value is not None:
                    source = method
                    dest = 'none'
                    factor = discont_factor
                    value = discont_value
                else:
                    # method used with factor/value means source=dest=method (staying on same method)
                    source = method
                    dest = method
            
            # Resolve method names to internal names (support both name and label)
            if source is not None and source.lower() != 'none':
                source = cm.get_method(source).name
            elif source is not None:
                source = 'none'
                
            if dest is not None and dest.lower() != 'none':
                dest = cm.get_method(dest).name
            elif dest is not None:
                dest = 'none'
            
            # Handle copy_from
            if copy_from is not None:
                copy_from_method = cm.get_method(copy_from).name
                if dest is not None:
                    # Copy column (all switches TO dest)
                    sw.copy_from_method_column(
                        source_to_method=copy_from_method,
                        dest_to_method=dest,
                        postpartum=postpartum,
                        age_grp=ages,
                        renormalize=True
                    )
                if source is not None and source != 'none':
                    # Copy row (all switches FROM source)
                    sw.copy_from_method_row(
                        source_from_method=copy_from_method,
                        dest_from_method=source,
                        postpartum=postpartum,
                        age_grp=ages,
                        renormalize=True
                    )
                continue  # Done with this entry
            
            # Handle factor/value modifications
            if factor is not None or value is not None:
                # Determine the source method
                from_method = 'birth' if postpartum == 1 else source
                
                if factor is not None:
                    # Use scale_entry for multiplicative changes
                    sw.scale_entry(from_method, dest, factor, postpartum=postpartum, 
                                   age_grp=ages, renormalize=False)
                else:
                    # Use set_entry for absolute values (handles iteration internally)
                    sw.set_entry(from_method, dest, value, postpartum=postpartum, 
                                 age_grp=ages, renormalize=False)
        
        # Renormalize all probabilities after all modifications
        sw.renormalize_all()
        
        if self.pars.verbose:
            print(f'Applied {len(probs)} probability modification(s) in year {self.sim.t.year}')
        
        return

    def step(self):
        """
        Applies the efficacy or contraceptive uptake changes if it is the specified year
        based on scenario specifications.
        """
        sim = self.sim
        cm = sim.connectors.contraception
        if not self.applied and sim.t.year >= self.pars.year:
            self.applied = True # Ensure we don't apply this more than once

            # Implement efficacy
            if self.pars.eff is not None:
                for k, rawval in self.pars.eff.items():
                    cm.update_efficacy(method_label=k, new_efficacy=rawval)

            # Implement changes in duration of use
            if self.pars.dur_use is not None:
                for k, rawval in self.pars.dur_use.items():
                    cm.update_duration(method_label=k, new_duration=rawval)

            # Change in probability of use
            if self.pars.p_use is not None:
                cm.pars['p_use'].set(self.pars.p_use)

            # Change in method mix
            if self.pars.method_mix is not None:
                this_mix = self.pars.method_mix / np.sum(self.pars.method_mix) # Renormalise in case they are not adding up to 1
                cm.pars['method_mix'] = this_mix
            
            # Change in switching matrix
            if self.pars.method_choice_pars is not None:
                print(f'Changed contraceptive switching matrix in year {sim.t.year}')
                cm.method_choice_pars = self.pars.method_choice_pars
            
            # Apply probability modifications
            if self.pars.probs is not None:
                self._apply_probs(cm)
                
        return


class change_initiation_prob(ss.Intervention):
    """
    Intervention to change the probabilty of contraception use trend parameter in
    contraceptive choice modules that have a logistic regression model.

    Args:
        year (float): The year in which this intervention will be applied
        prob_use_intercept (float): A number that changes the intercept in the logistic regression model
        p_use = 1 / (1 + np.exp(-rhs + p_use_time_trend + p_use_intercept))
    """

    def __init__(self, year=None, prob_use_intercept=0.0, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.year = year
        self.prob_use_intercept = prob_use_intercept
        self.verbose = verbose
        self.applied = False
        self.par_name = None
        return

    def __repr__(self):
        """Return a recreatable string representation."""
        parts = [f"year={self.year}", f"prob_use_intercept={self.prob_use_intercept}"]
        if self.verbose:
            parts.append("verbose=True")
        return f"fp.change_initiation_prob({', '.join(parts)})"

    def init_pre(self, sim=None):
        super().init_pre(sim)
        # self._validate()
        if isinstance(sim.people.contraception_module, (fpm.SimpleChoice)):
            self.par_name = 'prob_use_intercept'

        if self.par_name is None:
            errormsg = (
                f"Contraceptive module  {type(sim.people.contraception_module)} does not have `{self.par_name}` parameter.")
            raise ValueError(errormsg)

        return


    def step(self):
        """
        Applies the changes to efficacy or contraceptive uptake changes if it is the specified year
        based on scenario specifications.
        """
        sim = self.sim
        if not self.applied and sim.t.year >= self.year:
            self.applied = True # Ensure we don't apply this more than once
            sim.people.contraception_module.pars[self.par_name] = self.prob_use_intercept

        return


class change_initiation(ss.Intervention):
    """
    Intervention that modifies the outcomes of whether women are on contraception or not
    Select a proportion of women and sets them on a contraception method.

    Args:
        years (list, float): The year we want to start the intervention.
            if years is None, uses start and end years of sim as defaults
            if years is a number or a list with a single lem,ent, eg, 2000.5, or [2000.5],
            this is interpreted as the start year of the intervention, and the
            end year of intervention will be the eno of the simulation
        eligibility (callable): callable that returns a filtered version of
            people eligible to receive the intervention
        perc (float): a value between 0 and 1 indicating the x% extra of women
            who will be made to select a contraception method .
            The proportion or % is with respect to the number of
            women who were on contraception:
             - the previous year (12 months earlier)?
             - at the beginning of the intervention.
        annual (bool): whether the increase, perc, represents a "per year"
            increase.
    """

    def __init__(self, years=None, eligibility=None, perc=0.0, annual=True, force_theoretical=False, **kwargs):
        super().__init__(**kwargs)
        self.years = years
        self.eligibility = eligibility
        self.perc = perc
        self.annual = annual
        self.annual_perc = None
        self.force_theoretical = force_theoretical
        self.current_women_oncontra = None

        # Initial value of women on contra at the start of the intervention. Tracked for validation.
        self.init_women_oncontra = None
        # Theoretical number of women on contraception we should have by the end of the intervention period, if
        # nothing else affected the dynamics of the contraception. Tracked for validation.
        self.expected_women_oncontra = None
        return

    def __repr__(self):
        """Return a recreatable string representation."""
        parts = []
        if self.years is not None:
            parts.append(f"years={self.years}")
        if self.eligibility is not None:
            parts.append(f"eligibility={self.eligibility!r}")
        if self.perc != 0.0:
            parts.append(f"perc={self.perc}")
        if not self.annual:
            parts.append("annual=False")
        if self.force_theoretical:
            parts.append("force_theoretical=True")
        return f"fp.change_initiation({', '.join(parts)})"

    def init_pre(self, sim=None):
        super().init_pre(sim)

        # Lastly, adjust the probability by the sim's timestep, if it's an annual probability
        if self.annual:
            # per timestep/monthly growth rate or perc of eligible women who will be made to choose contraception
            self.annual_perc = self.perc
            self.perc = ((1 + self.annual_perc) ** sim.dt)-1
        # Validate years and values
        if self.years is None:
            # f'Intervention start and end years not provided. Will use sim start an end years'
            self.years = [sim.pars['start'], sim.pars['stop']]
        if sc.isnumber(self.years) or len(self.years) == 1:
            self.years = sc.promotetolist(self.years)
            # Assumes that start year has been specified, append end of the simulation as end year of the intervention
            self.years.append(sim.pars['stop'])

        min_year = min(self.years)
        max_year = max(self.years)
        if min_year < sim['start']:
            errormsg = f'Intervention start {min_year} is before the start of the simulation.'
            raise ValueError(errormsg)
        if max_year > sim['stop']:
            errormsg = f'Intervention end {max_year} is after the end of the simulation.'
            raise ValueError(errormsg)
        if self.years != sorted(self.years):
            errormsg = f'Years {self.years} should be monotonically increasing'
            raise ValueError(errormsg)

        return

    def check_eligibility(self):
        """
        Select eligible who is eligible
        """
        sim = self.sim
        contra_choosers = []
        if self.eligibility is None:
            contra_choosers = self._default_contra_choosers()
        return contra_choosers

    def _default_contra_choosers(self):
        # TODO: do we care whether women people have ti_contra > 0? For instance postpartum women could be made to choose earlier?
        # Though it is trickier because we need to reset many postpartum-related attributes
        ppl = self.sim.people
        eligible = ((ppl.sex == 0) & (ppl.alive) &                 # living women
                              (ppl.age < self.sim.pars.fp['age_limit_fecundity']) &  # who are fecund
                              (ppl.sexual_debut) &                           # who already had their sexual debut
                              (~ppl.pregnant)    &                           # who are not currently pregnant
                              (~ppl.postpartum)  &                           # who are not in postpartum
                              (~ppl.on_contra)                               # who are not already on contra
                              ).uids

        return eligible

    def step(self):
        sim = self.sim
        ti = sim.ti
        # Save theoretical number based on the value of women on contraception at start of intervention
        if self.years[0] == sim.t.year:
            self.expected_women_oncontra = (sim.people.alive & sim.people.on_contra).sum()
            self.init_women_oncontra = self.expected_women_oncontra

        # Apply intervention within this time range
        if self.years[0] <= sim.t.year <= self.years[1]:  # Inclusive range
            self.current_women_oncontra = (sim.people.alive & sim.people.on_contra).sum()

            # Save theoretical number based on the value of women on contraception at start of intervention
            nnew_on_contra = self.perc * self.expected_women_oncontra

            # NOTE: TEMPORARY: force specified increase
            # how many more women should be added per time step
            # However, if the current number of women on contraception is >> than the expected value, this
            # intervention does nothing. The forcing ocurrs in one particular direction, making it incomplete.
            # If the forcing had to be fully function, when there are more women than the expected value
            # this intervention should additionaly 'reset' the contraceptive state and related attributes (ie, like the duration on the method)
            if self.force_theoretical:
                additional_women_on_contra = self.expected_women_oncontra - self.current_women_oncontra
                if additional_women_on_contra < 0:
                    additional_women_on_contra = 0
                new_on_contra = nnew_on_contra + additional_women_on_contra
            else:
                new_on_contra = self.perc * self.current_women_oncontra

            self.expected_women_oncontra += nnew_on_contra

            if not new_on_contra:
                raise ValueError("For the given parameters (n_agents, and perc increase) we won't see an effect. "
                                 "Consider increasing the number of agents.")

            # Eligible population
            can_choose_contra_uids = self.check_eligibility()
            n_eligible = len(can_choose_contra_uids)

            if n_eligible:
                if n_eligible < new_on_contra:
                    print(f"There are fewer eligible women ({n_eligible}) than "
                          f"the number of women who should be initiated on contraception ({new_on_contra}).")
                    new_on_contra = n_eligible
                # Of eligible women, select who will be asked to choose contraception
                p_selected = new_on_contra * np.ones(n_eligible) / n_eligible
                sim.people.on_contra[can_choose_contra_uids] = fpu.binomial_arr(p_selected)
                new_users_uids = sim.people.on_contra[can_choose_contra_uids].uids
                sim.people.method[new_users_uids] = sim.people.contraception_module.init_method_dist(new_users_uids)
                sim.people.ever_used_contra[new_users_uids] = 1
                method_dur = sim.people.contraception_module.set_dur_method(new_users_uids)
                sim.people.ti_contra[new_users_uids] = ti + method_dur
            else:
                print(f"Ran out of eligible women to initiate")
        return
