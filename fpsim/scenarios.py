'''
Class to define and run scenarios
'''

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from . import parameters as fpp
from . import sim as fps
from . import interventions as fpi

__all__ = ['make_scen', 'Scenario', 'Scenarios']


#%% Validation functions -- for internal use only

def check_not_none(obj, *args):
    ''' Check that all needed inputs are supplied '''
    for key in args:
        if obj[key] is None:
            errormsg = f'Entry "{key}" is not allowed to be None for the following scenario spec:\n{obj}'
            raise ValueError(errormsg)
    return



#%% Scenario classes

class Scenario(sc.prettyobj, sc.dictobj):
    '''
    Store the specification for a single scenario (which may consist of multiple interventions).

    This function is intended to be as flexible as possible; as a result, it may
    be somewhat confusing. There are five different ways to call it -- method efficacy,
    method probability, method initiation/discontinuation, parameter, and custom intervention.

    Args (shared):
        spec   (dict): a pre-made specification of a scenario; see keyword explanations below (optional)
        args   (list): additional specifications (optional)
        label  (str): the sim label to use for this scenario
        pars   (dict): optionally supply additional sim parameters to use with this scenario (that take effect at the beginning of the sim, not at the point of intervention)
        year   (float): the year at which to activate efficacy and probability scenarios
        matrix (str): which set of probabilities to modify for probability scenarios (e.g. annual or postpartum)
        ages   (str/list): the age groups to modify the probabilities for

    Args (efficacy):
        year (float): as above
        eff  (dict): a dictionary of method names and new efficacy values

    Args (probablity):
        year   (float): as above
        matrix (str): as above
        ages   (str): as above
        source (str): the method to switch from
        dest   (str): the method to switch to
        factor (float): if supplied, multiply the [source, dest] probability by this amount
        value  (float): if supplied, instead of factor, replace the [source, dest] probability by this value
        copy_from (str): if supplied, copy probabilities from a different method

    Args (initiation/discontinuation):
        year   (float): as above
        matrix (str): as above
        ages   (str): as above
        method (str): the method for initiation/discontinuation
        init_factor    (float): as with "factor" above, for initiation (None → method)
        discont_factor (float): as with "factor" above, for discontinuation (method → None)
        init_value     (float): as with "value" above, for initiation (None → method)
        discont_value  (float): as with "value" above, for discontinuation (method → None)

    Args (parameter):
        par (str): the parameter to modify
        par_years (float/list): the year(s) at which to apply the modifications
        par_vals (float/list): the value(s) of the parameter for each year

    Args (custom):
        interventions (Intervention/list): any custom intervention(s) to be applied to the scenario

    Congratulations on making it this far.

    **Examples**::

        # Basic efficacy scenario
        s1 = fp.make_scen(eff={'Injectables':0.99}, year=2020)

        # Double rate of injectables initiation -- alternate approach
        s3 = fp.make_scen(method='Injectables', init_factor=2)

        # Parameter scenario: halve exposure
        s5 = fp.make_scen(par='exposure_factor', years=2010, vals=0.5)

        # Custom scenario
        def update_sim(sim): sim.updated = True
        s6 = fp.make_scen(interventions=update_sim)

        # Combining multiple scenarios: change probabilities and exposure factor
        s7 = fp.make_scen(
            dict(method='Injectables', init_value=0.1, discont_value=0.02, create=True),
            dict(par='exposure_factor', years=2010, vals=0.5)
        )

        # Scenarios can be combined
        s8 = s1 + s2
    '''
    def __init__(self, spec=None, label=None, pars=None, year=None, # Basic settings
                 eff=None, dur_use=None, p_use=None, method_mix=None, # Option 1
                 par=None, par_years=None, par_vals=None, # Option 4
                 interventions=None, # Option 5
                 ):

        # Handle input specification
        if isinstance(spec, str) and not isinstance(label, str): # Swap order if types don't match
            label,spec = spec,label
        self.specs = sc.mergelists(*[Scenario(**s).specs for s in sc.tolist(spec)]) # Sorry
        self.label = label
        self.pars  = sc.mergedicts(pars)

        # Handle other keyword inputs
        contraceptive_spec = None
        par_spec   = None
        intv_specs = None

        # It's a scenario related to changing the contraceptive mix parameters
        for scentype, input in {'eff': eff, 'dur_use': dur_use, 'p_use': p_use, 'method_mix': method_mix}.items():
            if input is not None:
                contraceptive_spec = sc.objdict({
                    'which': scentype,
                    scentype: input,
                    'year': year
                })
                check_not_none(contraceptive_spec, 'year')

        # It's a parameter change scenario
        if par is not None:
            par_spec = sc.objdict(
                which = 'par',
                par   = par,
                par_years = par_years,
                par_vals  = par_vals,
            )
            check_not_none(par_spec, 'par_years', 'par_vals')

        # It's a custom scenario(s)
        if interventions is not None:
            intv_specs = []
            for intv in sc.tolist(interventions):
                intv_specs.append(sc.objdict(
                    which        = 'intv',
                    intervention = intv,
                ))

        # Merge these different scenarios into the list, skipping None entries
        self.specs.extend(sc.mergelists(contraceptive_spec, par_spec, intv_specs))

        # Finally, ensure all have a consistent label if supplied
        self.update_label()

        return

    def __add__(self, scen2):
        ''' Combine two scenario spec lists '''
        scen3 = sc.dcp(self)
        scen3.specs.extend(sc.dcp(scen2.specs))
        return scen3

    def __radd__(self, scen2):
        ''' Allows sum() to work correctly '''
        if not scen2: return self
        else:         return self.__add__(scen2)

    def update_label(self, label=None):
        ''' Ensure all specs have the correct label '''

        # Option 1: update the label based on the spec
        if label is None:
            label = self.label
            for spec in self.specs:
                if 'label' in spec and spec['label'] is not None:
                    label = spec['label']

        # Option 2: update the spec based on the label
        else:
            for spec in self.specs:
                spec['label'] = label

        self.label = label
        return

    def run(self, run_args=None, **kwargs):
        '''
        Shortcut for creating and running a Scenarios object based on the current scenario.

        Args:
            run_args (dict): passed to scens.run()
            kwargs (dict): passed to Scenarios()

        '''
        scens = Scenarios(**kwargs)
        scens.add_scen(self)
        scens.run(**sc.mergedicts(run_args))
        return scens


def make_scen(*args, **kwargs):
    ''' Alias for ``fp.Scenario()``. '''
    return Scenario(*args, **kwargs)


# Ensure the function has the same docstring as the class
make_scen.__doc__ +=  '\n\n' + Scenario.__doc__


class Scenarios(sc.prettyobj):
    '''
    Run different intervention scenarios.

    A "scenario" can be thought of as a list of sims, all with the same parameters
    except for the random seed. Usually, scenarios differ from each other only
    in terms of the interventions run (to compare other differences between sims,
    it's preferable to use a MultiSim object).

    Args:
        pars    (dict): parameters to pass to the sim
        repeats (int):  how many repeats of each scenario to run (default: 1)
        scens   (list): the list of scenarios to run; see also ``fp.make_scen()`` and ``Scenarios.add_scen()``
        kwargs  (dict): optional additional parameters to pass to the sim

    **Example**::

        scen1 = fp.make_scen(label='Baseline')
        scen2 = fp.make_scen(year=2002, eff={'Injectables':0.99}) # Basic efficacy scenario
        scens = fp.Scenarios(location='test', repeats=2, scens=[scen1, scen2])
        scens.run()
    '''

    def __init__(self, repeats=None, scens=None):
        self.sim_par_keys = fpp.sim_par_keys
        self.fp_pars_keys = fpp.par_keys

        self.repeats = repeats if repeats is not None else 1
        self.scens = sc.dcp(sc.tolist(scens))
        self.simslist = []
        self.msims = []
        self.msim = None
        self.already_run = False
        return

    def __bool__(self):
        """ Always return True """
        return True

    def __len__(self):
        """ Try to get the length of the scenarios, else return 0 """
        try:
            return len(self.scens)
        except:
            return 0

    def add_scen(self, scen=None, label=None):
        ''' Add a scenario or scenarios to the Scenarios object '''
        if isinstance(scen, Scenario):
            scen.update_label(label)
            self.scens.append(scen)
        else:
            self.scens.append(Scenario(label=label, spec=scen))
        return

    def update_pars(self, **kwargs):
        new_sim_pars = {}
        new_fp_pars = {}

        for k,v in kwargs.items():
            if k in self.sim_par_keys:
                new_sim_pars[k] = v
            elif k in self.fp_pars_keys:
                new_fp_pars[k] = v
            else:
                errormsg = f'Parameter "{k}" not recognized'
                raise ValueError(errormsg)
        return new_sim_pars, new_fp_pars

    def make_sims(self, scenlabel, **kwargs):
        ''' Create a list of sims that are all identical except for the random seed '''
        pars = dict()
        if scenlabel is None:
            errormsg = 'Scenario label must be defined'
            raise ValueError(errormsg)
        sims = sc.autolist()
        for i in range(self.repeats):
            pars['rand_seed'] = i

            sim = fps.Sim(pars=pars, **kwargs)

            sim.scenlabel = scenlabel # Special label for scenarios objects
            if sim.label is None:
                sim.label = scenlabel # Include here if no other label
            sims += sim
        return sims

    def make_scens(self):
        ''' Convert a scenario specification into a list of sims '''
        for i,scen in enumerate(self.scens):
            simlabel = scen.label
            interventions = sc.autolist()
            for spec in sc.tolist(scen.specs):

                # Figure ou what type of spec this is
                spec = sc.dcp(spec)
                if 'which' not in spec:
                    errormsg = f'Invalid scenario spec: key "which" not found among: {sc.strjoin(spec.keys())}'
                    raise ValueError(errormsg)
                else:
                    which = spec.pop('which')

                # Handle interventions
                if which in ['eff', 'dur_use', 'p_use', 'method_mix']:
                    pardict  = spec.pop(which)
                    year = spec.pop('year')
                    interventions += fpi.update_methods(**{which: pardict, 'year': year})
                elif which == 'par':
                    par = spec.pop('par')
                    years = spec.pop('par_years')
                    vals  = spec.pop('par_vals')
                    interventions += fpi.change_par(par=par, years=years, vals=vals)
                elif which == 'intv':
                    intv = spec.pop('intervention')
                    interventions += intv
                else:
                    errormsg = f'Could not understand intervention type "{which}"'
                    raise ValueError(errormsg)

                # Handle label
                label  = spec.pop('label', scen.label)
                assert len(spec)==0, f'Unrecognized scenario key(s) {sc.strjoin(spec.keys())}'
                if simlabel is None:
                    simlabel = label
                else:
                    if label != simlabel:
                        print('Warning, new sim label {label} does not match existing sim label {simlabel}')

            if simlabel is None:
                simlabel = f'Scenario {i}'
            sims = self.make_sims(scenlabel=simlabel, interventions=interventions, **scen.pars)
            self.simslist.append(sims)
        return

    def run(self, **kwargs):
        ''' Actually run a list of sims '''

        # Check that it's set up
        if not self.scens:
            errormsg = 'No scenarios are defined'
            raise ValueError(errormsg)
        if not self.simslist:
            self.make_scens()

        # Create msim
        if self.repeats == 1:
            self.msim = ss.MultiSim(*[s for s in self.simslist])
        else:
            errormsg = 'Scenarios with repeats > 1 are not supported.'
            raise ValueError(errormsg)

        # Run
        self.msim.run(**kwargs)
        self.already_run = True

        # Process
        self.analyze_sims()
        return

    def check_run(self):
        ''' Give a meaningful error message if the scenarios haven't been run '''
        if not self.already_run:
            errormsg = 'Scenarios have not yet been run; please run first via scens.run()'
            raise RuntimeError(errormsg)
        return


    def plot(self, to_plot=None, plot_sims=True, **kwargs):
        ''' Plot the scenarios with bands -- see ``sim.plot()`` for args '''
        self.check_run()
        if to_plot == 'method':
            return self.msim.plot(to_plot=to_plot, plot_sims=plot_sims, **kwargs)
        else:
            return self.msim_merged.plot(to_plot=to_plot, plot_sims=plot_sims, **kwargs)


    def plot_sims(self, to_plot=None, plot_sims=True, **kwargs):
        ''' Plot each sim as a separate line across all senarios -- see ``sim.plot()`` for args '''
        self.check_run()
        return self.msim.plot(to_plot=to_plot, plot_sims=plot_sims, **kwargs)

    def analyze_sims(self, start=None, end=None):
        ''' Take a list of sims that have different labels and extrapolate statistics from each '''
        self.check_run()

        # Pull out first sim and parameters
        sim0 = self.msim.sims[0]
        if start is None: start = sim0.pars['start']
        if end   is None: end   = sim0.pars['stop']

        # Split the sims up by scenario
        results = sc.objdict()
        sims = sc.objdict(defaultdict=sc.autolist)
        for sim in self.msim.sims:
            try:
                label = sim.scenlabel
            except:
                errormsg = f'Warning, could not extract scenlabel from sim {sim.label}; using default...'
                print(errormsg)
                label = sim.label
            sims[label] += sim

        # Count the births across the scenarios
        raw = sc.ddict(list)
        for key,sims in sims.items():
            for sim in sims:
                results_df = sim.to_df()
                raw['scenario'] += [key]      # Append scenario key
                raw['births']   += [results_df['fp_births']] # Append births
                raw['fails']    += [results_df['fp_method_failures']]  # Append failures
                raw['popsize']  += [results_df['n_alive']]    # Append population size
                raw['tfr']      += [results_df['fp_tfr']]    # Append mean tfr rates
                raw['infant_deaths'] += [results_df['fp_infant_deaths']]
                raw['maternal_deaths'] += [results_df['fp_maternal_deaths']]
                raw['mcpr'] += [results_df['contraception_mcpr']]

        # Calculate basic stats
        results.stats = sc.objdict()
        for statkey in ['mean', 'median', 'std', 'min', 'max']:
            results.stats[statkey] = sc.objdict()
            for k,vals in raw.items():
                if k != 'scenario':
                    results.stats[statkey][k] = getattr(np, statkey)(vals)

        # Also save as pandas
        results.df = pd.DataFrame(raw)
        self.results = results

        return
