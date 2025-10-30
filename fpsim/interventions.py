'''
Specify the core interventions available in FPsim. Other interventions can be
defined by the user by inheriting from these classes.
'''
import numpy as np
import sciris as sc
import starsim as ss
from . import utils as fpu
from . import methods as fpm

#%% Generic intervention classes

__all__ = ['change_par', 'update_methods', 'change_people_state', 'change_initiation_prob', 'change_initiation']


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

    """

    def __init__(self, year, eff=None, dur_use=None, p_use=None, method_mix=None, method_choice_pars=None, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            year=year,
            eff=eff,
            dur_use=dur_use,
            p_use=p_use,
            method_mix=method_mix,
            method_choice_pars=method_choice_pars,
            verbose=verbose
        )

        self.applied = False
        return


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
        if self.pars.eff is None and self.pars.dur_use is None and self.pars.p_use is None and self.pars.method_mix is None and self.pars.method_choice_pars is None:
            errormsg = 'Either efficacy, durations of use, probability of use, or method mix must be supplied'
            raise ValueError(errormsg)
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
