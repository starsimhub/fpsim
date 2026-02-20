"""
Defines the FPmod class. This class inherits from the Starsim Pregnancy module, but has its own
logic for how pregnancies are conceived.
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
import fpsim as fp
from . import defaults as fpd
import starsim as ss

# Specify all externally visible things this file defines
__all__ = ['FPmod']


# %% Define classes
class FPmod(ss.Pregnancy):
    """
    Class for storing and updating FP-related events. Inherits from Starsim's Pregnancy module.

    Methods are organized in the order they are called during the simulation lifecycle:
    initialization → pre-step updates → pregnancy progression → delivery → conception → results.
    """

    # %% Initialization

    def __init__(self, pars=None, location=None, data=None, name='fp', **kwargs):
        """
        Initialize FPmod with parameters, data, and distributions.

        Sets up FP-specific parameters, states, and probability distributions for
        conception, mortality, sexual activity, and contraception. Also processes
        exposure calibration data into interpolation functions.

        Args:
            pars (dict): FP parameters to override defaults
            location (str): location name for loading data (e.g. 'senegal')
            data (dict): pre-loaded FP data; if None, loaded from location
            name (str): module name (default 'fp')
            **kwargs: additional parameters passed to update_pars
        """
        super().__init__(name=name)

        # Define parameters
        default_pars = fp.FPPars()
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)
        self.define_states(*fp.fpmod_states)

        # Get data parameters if not provided
        if data is None and location is not None:
            dataloader = fp.get_dataloader(location)
            data = dataloader.load_fp_data(return_data=True)
        self.update_pars(data)

        # Binary distributions specific to FPmod, used for calculating p_conceive
        self._p_inf_mort = ss.bernoulli(p=0)  # Probability of infant mortality
        self._p_abortion = ss.bernoulli(p=0)  # Probability of abortion
        self._p_lam = ss.bernoulli(p=0)  # Probability of LAM
        self._p_active = ss.bernoulli(p=0)
        def age_adjusted_non_pp_active(self, sim, uids):
            return self.pars['sexual_activity'][sim.people.int_age(uids)]
        self._p_non_pp_active = ss.bernoulli(p=age_adjusted_non_pp_active)  # Probability of being sexually active if not postpartum
        self.pars.embryos_per_pregnancy.set(p=[1-self.pars.twins_prob,self.pars.twins_prob])

        # All other distributions
        self._fated_debut = ss.choice(a=self.pars['debut_age']['ages'], p=self.pars['debut_age']['probs'])

        # Deal with exposure parameters
        if isinstance(self.pars['exposure_age'], dict):
            ea = np.array([v for v in self.pars['exposure_age'].values()])
        else:
            ea = sc.toarray(self.pars['exposure_age'])
        if isinstance(self.pars['exposure_parity'], dict):
            ep = np.array([v for v in self.pars['exposure_parity'].values()])
        else:
            ep = sc.toarray(self.pars['exposure_parity'])
        self.pars['exposure_age'] = fp.data2interp(ea, fpd.spline_preg_ages)
        self.pars['exposure_parity'] = fp.data2interp(ep, fpd.spline_parities)

        # Non-standard results - consider moving to the contraception module
        self.method_mix = None

        return

    def init_results(self):
        """
        Initialize result storage.

        Adds FP-specific results on top of the parent Pregnancy results:
        - Event counts (pregnancies, births, stillbirths, etc.) with cumulative versions
        - People counts (n_fecund, n_on_contra, etc.)
        - Rates (IMR, proportion of short-interval births)
        - Method mix array (stored separately as a 2D array, not a Result)
        """
        super().init_results()
        scaling_kw = dict(dtype=int, scale=True)
        nonscaling_kw = dict(dtype=float, scale=False)
        results = sc.autolist()

        # Add event counts - these are all integers, and are scaled by the number of agents
        # We compute new results for each event type, and also cumulative results
        for key in fpd.event_counts:
            results += ss.Result(key, label=key, **scaling_kw, summarize_by='sum')
            results += ss.Result(f'cum_{key}', label=key, dtype=int, scale=False, summarize_by='last')

        # Add people counts - these are all integers, and are scaled by the number of agents
        # However, for these we do not include cumulative totals
        for key in fpd.people_counts:
            results += ss.Result(key, label=key, **scaling_kw, summarize_by='sum')

        # Add infant mortality and short interval births - these are proportions, not scaled
        results += ss.Result('imr', label='Infant mortality rate', **nonscaling_kw, summarize_by='mean')
        results += ss.Result('p_short_interval', label='Proportion of short interval births', **nonscaling_kw, summarize_by='mean')

        # Store results
        self.define_results(*results)

        # Additional results with different formats, stored separately
        # These will not be appended to sim.results, and must be accessed
        # via eg. sim.people.fp.method_mix
        self.method_mix = np.zeros((self.sim.connectors.contraception.n_options, self.t.npts))
        return

    def init_intent_states(self, uids):
        """
        Initialize intent_to_use and fertility_intent states for women aged 15-49.

        Uses age-specific probability distributions from the contraception connector's
        intent_pars data. If no data is available, sets defaults (False for both states).

        Fertility intent categories: 0=cannot-get-pregnant, 1=no, 2=yes.

        Args:
            uids: UIDs of agents to initialize (filtered to eligible 15-49 females)
        """
        ppl = self.sim.people

        # Filter eligible women (15-49, alive, female)
        eligible_mask = ((ppl.age >= fpd.min_age) &
                        (ppl.age < fpd.max_age_preg) &
                        ppl.female &
                        ppl.alive)
        eligible_uids = uids[eligible_mask[uids]]

        if len(eligible_uids) == 0:
            return

        # Load data if available from contraception connector
        contra_connector = self.sim.connectors.contraception
        intent_pars = contra_connector.pars.get('intent_pars', None)

        if intent_pars:
            fertility_data = intent_pars.get('fertility_intent', {})
            contraception_data = intent_pars.get('contra_intent', {})

        # If no data available, use defaults
        if not intent_pars or not fertility_data or not contraception_data:
            # Set default values
            self.fertility_intent[eligible_uids] = False
            self.intent_to_use[eligible_uids] = False
            return

        # Transform ages to integers for indexing
        from . import utils as fpu
        ages = fpu.digitize_ages_1yr(ppl.age[eligible_uids])

        # Mapping from fertility intent category string to integer code
        # 0=cannot-get-pregnant (reference), 1=no, 2=yes
        fi_cat_map = {'cannot-get-pregnant': 0, 'no': 1, 'yes': 2}

        # Initialize fertility_intent
        for i, uid in enumerate(eligible_uids):
            age = int(ages[i])
            if age in fertility_data:
                # Sample from the distribution
                choices = list(fertility_data[age].keys())
                probs = list(fertility_data[age].values())
                choice = np.random.choice(choices, p=probs)
                self.fertility_intent[uid] = (choice == 'yes')
                self.fertility_intent_cat[uid] = fi_cat_map.get(choice, 0)
            else:
                # Default if age not in data
                self.fertility_intent[uid] = False
                self.fertility_intent_cat[uid] = 0

        # Initialize intent_to_use
        for i, uid in enumerate(eligible_uids):
            age = int(ages[i])
            if age in contraception_data:
                # Sample from the distribution
                choices = list(contraception_data[age].keys())
                probs = list(contraception_data[age].values())
                choice = np.random.choice(choices, p=probs)
                self.intent_to_use[uid] = (choice == '1')
            else:
                # Default if age not in data
                self.intent_to_use[uid] = False
        return

    # %% Properties

    @property
    def postpartum(self):
        """
        Boolean mask of women within the postpartum window (defined by dur_postpartum par).

        This does not directly affect any other functionality within this module, but is
        provided for convenience for modules that need to know which women are X timesteps
        postpartum.
        """
        timesteps_since_birth = self.ti - self.ti_delivery
        pp_timesteps = int(self.pars.dur_postpartum/self.t.dt)
        pp_bools = ~self.pregnant & (timesteps_since_birth >= 0) & (timesteps_since_birth <= pp_timesteps)
        return pp_bools

    @property
    def susceptible(self):
        """
        Sexually-active fertile women of childbearing age who are not pregnant.

        Overrides the parent definition to add sexually_active as an additional requirement.
        """
        return self.fertile & self.sexually_active & ~self.pregnant

    @property
    def end_tri1_uids(self):
        """ Return UIDs of those in the final timestep of their first trimester """
        end = (self.dur_gestation <= self.pars.trimesters[0]) & ((self.dur_gestation + self.dt) > self.pars.trimesters[0])
        return self.pregnant.uids[end]

    # %% Utilities

    def _get_uids(self, upper_age=None, female_only=True):
        """
        Get UIDs filtered by age and optionally by sex.

        Args:
            upper_age (float): maximum age to include (default: no limit)
            female_only (bool): if True, only return female UIDs (default: True)

        Returns:
            ss.uids: filtered UIDs
        """
        people = self.sim.people
        if upper_age is None: upper_age = 1000
        within_age = people.age <= upper_age
        if female_only:
            f_uids = (within_age & people.female).uids
            return f_uids
        else:
            uids = within_age.uids
            return uids

    def bfilter(self, dist, uids, probs):
        """
        Shortcut for probability-based filtering: set dist probability and return matching UIDs.

        Args:
            dist: a Bernoulli distribution object
            uids: UIDs to filter
            probs: probability values to set on the distribution

        Returns:
            ss.uids: UIDs that passed the filter
        """
        dist.set(p=probs)
        return dist.filter(uids)

    def bsplit(self, dist, uids, probs):
        """
        Shortcut for probability-based splitting: set dist probability and split UIDs into two groups.

        Args:
            dist: a Bernoulli distribution object
            uids: UIDs to split
            probs: probability values to set on the distribution

        Returns:
            tuple: (true_uids, false_uids) — UIDs that matched and those that didn't
        """
        dist.set(p=probs)
        return dist.split(uids)

    # %% Pre-step updates

    def updates_pre(self, uids=None, upper_age=None):
        """
        Pre-step state updates, run at the beginning of each timestep.

        This runs prior to calculating pregnancy exposure, advancing pregnancies,
        adding new pregnancies, or determining delivery outcomes. Steps performed:
            1. Update fecundity and mortality rates for the current year
            2. Set sexual activity status (postpartum and non-postpartum)
            3. Initialize contraception timing for new agents
            4. Initialize fertility intent for women entering the 15-49 age range
            5. Update contraceptive intent and method choices

        Args:
            uids: UIDs to update (default: all female agents up to upper_age)
            upper_age (float): maximum age for filtering UIDs
        """
        if uids is None: uids = self._get_uids(upper_age=upper_age)
        super().updates_pre(uids=uids)

        # Update fecundity and mortality
        self.personal_fecundity[uids] = self.pars.fecundity.rvs(uids)
        self.update_mortality()

        # Sexual activity
        # Default initialization for fated_debut; subnational debut initialized in subnational.py otherwise
        self.fated_debut[uids] = self._fated_debut.rvs(uids)
        self.check_sexually_active(self.fecund.uids)  # Check for all women of childbearing age
        self.update_time_to_choose(uids)

        # Initialize fertility intent for women who just turned 15 (entered the 15-49 age range this timestep)
        ppl = self.sim.people
        nonpreg = (self.fecund & ~self.pregnant).uids
        just_15 = nonpreg[(ppl.age[nonpreg] > fpd.min_age) & (ppl.age[nonpreg] <= fpd.min_age + self.t.dt_year)]
        if len(just_15) > 0:
            self.init_intent_states(just_15)

        # Update intent to use on birthday for any non-preg or >1m pp
        self.sim.connectors.contraception.update_intent_to_use(nonpreg)

        # Update methods for those who are eligible
        method_updaters = ((self.ti_contra <= self.ti) & self.fecund & ~self.pregnant).uids
        if len(method_updaters):
            self.sim.connectors.contraception.update_contra(method_updaters)
            self.results['switchers'][self.ti] = len(method_updaters)  # How many people switch methods (incl on/off)
        methods_ok = np.array_equal(self.on_contra.nonzero()[-1], self.method.nonzero()[-1])
        if not methods_ok:
            errormsg = 'Agents not using contraception are not the same as agents who are using None method'
            raise ValueError(errormsg)

        return uids

    def update_mortality(self):
        """
        Update infant, maternal, and stillbirth mortality rates for the current year.

        Reads year-specific rates from the parameter data and stores them in
        self.mortality_probs (for infant and stillbirth) and updates the maternal
        death distribution directly.
        """

        mapping = {
            'infant_mortality': 'infant',
            'maternal_mortality': 'maternal',
            'stillbirth_rate': 'stillbirth',
        }
        self.mortality_probs = {}
        for key1, key2 in mapping.items():
            ind = sc.findnearest(self.pars[key1]['year'], self.t.now('year'))
            val = self.pars[key1]['probs'][ind]
            if key2 == 'maternal':
                self.pars.p_maternal_death.set(p=val)
            else:
                self.mortality_probs[key2] = val
        return

    def check_sexually_active(self, uids=None):
        """
        Determine sexual activity status for all women each timestep.

        Partitions women into postpartum and non-postpartum groups, then delegates
        to _set_pp_active and _set_non_pp_active respectively. Finally updates the
        months-inactive counter via _update_inactive.

        Args:
            uids: UIDs to check (default: all alive agents)
        """
        if uids is None:
            uids = self.alive.uids

        # Partition into postpartum and non-postpartum
        pp_uids = uids & self.postpartum.uids
        non_pp_uids = uids - self.postpartum.uids

        # Set sexual activity for each group
        self._set_pp_active(pp_uids)
        self._set_non_pp_active(non_pp_uids)

        # Update inactivity counters
        self._update_inactive(uids)

        return

    def _set_pp_active(self, pp_uids):
        """
        Set sexual activity for postpartum women using birth spacing preferences.

        Uses time-since-birth to look up the baseline probability of sexual activity
        from DHS postpartum data, then adjusts by the agent's birth spacing preference.

        Args:
            pp_uids: UIDs of postpartum women
        """
        if len(pp_uids) == 0:
            return

        pref = self.pars['spacing_pref']
        timesteps_since_birth = self.ti - self.ti_delivery[pp_uids]
        spacing_bins = timesteps_since_birth / pref['interval']
        spacing_bins = np.array(np.minimum(spacing_bins, pref['n_bins']), dtype=int)
        probs_pp = self.pars['sexual_activity_pp']['percent_active'][timesteps_since_birth.astype(int)]
        probs_pp *= pref['preference'][spacing_bins]
        self._p_active.set(p=probs_pp)
        self.sexually_active[pp_uids] = self._p_active.rvs(pp_uids)

    def _set_non_pp_active(self, non_pp_uids):
        """
        Set sexual activity for non-postpartum women using age-specific rates.

        Also tracks sexual debut: the first time a woman becomes sexually active,
        her debut age is recorded.

        Args:
            non_pp_uids: UIDs of non-postpartum women
        """
        if len(non_pp_uids) == 0:
            return

        ppl = self.sim.people
        self.sexually_active[non_pp_uids] = self._p_non_pp_active.rvs(non_pp_uids)

        # Track sexual debut
        never_sex = self.sexual_debut[non_pp_uids] == 0
        now_active = self.sexually_active[non_pp_uids] == 1
        first_debut = non_pp_uids[now_active & never_sex]
        self.sexual_debut[first_debut] = True
        self.sexual_debut_age[first_debut] = ppl.age[first_debut]

    def _update_inactive(self, uids):
        """
        Update the months-inactive counter for women who have debuted.

        Resets to 0 for active women; increments by 1 for inactive women who have
        previously debuted. Women who have not yet debuted are not tracked.

        Args:
            uids: UIDs of women to update
        """
        active_sex = (self.sexually_active[uids] == 1)
        debuted = (self.sexual_debut[uids] == 1)
        active = uids[(active_sex & debuted)]
        inactive = uids[(~active_sex & debuted)]
        self.months_inactive[active] = 0
        self.months_inactive[inactive] += 1

    def update_time_to_choose(self, uids=None):
        """
        Initialize the counter for when women will first choose a contraceptive method.

        Computes the number of timesteps until each woman's fated sexual debut age,
        then sets ti_contra accordingly. Women past their debut age get ti_contra = 0
        (choose immediately).

        Args:
            uids: UIDs to initialize (default: all alive agents)
        """
        ppl = self.sim.people
        if uids is None:
            uids = self.alive.uids

        fecund = uids[(ppl.female[uids] == True) & (ppl.age[uids] < self.pars['age_limit_fecundity'])]
        ti_to_debut = ss.years(self.fated_debut[fecund]-ppl.age[fecund])/self.t.dt

        # If ti_contra is less than one timestep away, we want to also set it to 0 so floor time_to_debut.
        self.ti_contra[fecund] = np.maximum(np.floor(ti_to_debut), 0)

        # Validation
        time_to_set_contra = self.ti_contra[fecund] == 0
        if not np.array_equal(((ppl.age[fecund] - self.fated_debut[fecund]) > - self.t.dt), time_to_set_contra):
            errormsg = 'Should be choosing contraception for everyone past fated debut age.'
            raise ValueError(errormsg)
        return

    # %% Pregnancy progression

    def progress_pregnancies(self):
        """
        Advance ongoing pregnancies and check for miscarriage at end of first trimester.

        Calls the parent to update the gestational clock, then checks if any women
        have reached the end of their first trimester and applies age-specific
        miscarriage probabilities.
        """
        # Call parent to update gestational clock
        super().progress_pregnancies()

        # Check for miscarriage at end of first trimester
        if self.pregnant.any():
            end_tri1 = self.end_tri1_uids
            ppl = self.sim.people
            if len(end_tri1):
                miscarriage_probs = self.pars.miscarriage_rates[ppl.int_age_clip(end_tri1)]
                miscarriage_uids = self.bfilter(self._p_miscarriage, end_tri1, miscarriage_probs)
                if len(miscarriage_uids):
                    self.handle_miscarriage(miscarriage_uids)
        return

    def handle_miscarriage(self, uids):
        """
        Process miscarriage outcomes.

        Resets pregnancy states (pregnant, gestation, ti_delivery), increments
        the miscarriage counter, records the age at miscarriage, and triggers
        handle_loss for contraception/breastfeeding updates.

        Args:
            uids: UIDs of women who miscarried
        """
        ppl = self.sim.people
        n_miscarriages = len(uids)

        # Update states to reflect miscarriage
        self.pregnant[uids] = False
        self.gestation[uids] = np.nan  # No longer pregnant so remove gestational clock
        self.ti_delivery[uids] = np.nan
        self.n_miscarriages[uids] += 1
        self.ti_miscarriage[uids] = self.ti

        # Handle loss of infant and contraceptive update
        self.handle_loss(uids)

        # Track ages
        for uid in uids:
            # put miscarriage age in first nan slot
            age_idx = np.where(np.isnan(self.miscarriage_ages[uid]))[0][0]
            self.miscarriage_ages[uid, age_idx] = ppl.age[uid]

        # Update results
        self.results['miscarriages'][self.ti] = n_miscarriages

        return

    # %% Delivery

    def process_delivery(self, mother_uids, newborn_uids):
        """
        Save twin status before the parent resets it, then proceed with delivery.

        The parent's process_delivery sets carrying_multiple to False before calling
        _post_delivery, so we capture the twin status here for use in _do_stillbirths.

        Args:
            mother_uids: UIDs of mothers delivering
            newborn_uids: UIDs of newborn agents
        """
        self._twins_at_delivery = mother_uids & self.carrying_multiple.uids
        super().process_delivery(mother_uids, newborn_uids)

    def _post_delivery(self, mother_uids, newborn_uids):
        """
        FP-specific post-delivery processing.

        Orchestrates: stillbirths → neonatal mortality → contraception timing.

        Args:
            mother_uids: UIDs of all mothers who delivered
            newborn_uids: UIDs of all newborn agents
        """
        # Handle stillbirths and get mothers of live births
        mothers_of_live = self._do_stillbirths(mother_uids)

        # Handle infant mortality
        mothers_of_nnds, nnds = self.check_infant_mortality(mothers_of_live)
        self.handle_loss(mothers_of_nnds)
        self.results['infant_deaths'][self.ti] += len(nnds)

        # Calculate final mothers of live babies
        mothers_of_live = mother_uids - (mother_uids - mothers_of_live) - mothers_of_nnds

        # Set contraception timing for mothers with live births
        self._set_contra_timing(mothers_of_live)

        return

    def _do_stillbirths(self, mother_uids):
        """
        Handle stillbirth outcomes with age-adjusted probability.

        Splits delivering mothers into stillbirth and live birth groups using
        age-adjusted stillbirth rates. For live births, separates twins from
        singletons and updates parity, birth counts, and age records.

        Args:
            mother_uids: UIDs of all mothers delivering this timestep

        Returns:
            ss.uids: UIDs of mothers with live births
        """
        ppl = self.sim.people
        pars = self.pars

        # Calculate age-adjusted stillbirth probability
        still_prob = self.mortality_probs['stillbirth']
        rate_ages = pars['stillbirth_rate']['ages']
        age_ind = np.searchsorted(rate_ages, ppl.age[mother_uids], side="left")
        prev_idx_is_less = ((age_ind == len(rate_ages)) | (
                np.fabs(ppl.age[mother_uids] - rate_ages[np.maximum(age_ind - 1, 0)]) < np.fabs(
            ppl.age[mother_uids] - rate_ages[np.minimum(age_ind, len(rate_ages) - 1)])))
        age_ind[prev_idx_is_less] -= 1
        still_prob = still_prob * (pars['stillbirth_rate']['age_probs'][age_ind]) if len(self) > 0 else 0

        # Split into stillbirths and live births
        mothers_of_stillborns, mothers_of_live = self.bsplit(self._p_stillbirth, mother_uids, still_prob)

        # Sort by twins and single births
        twins = mothers_of_live & self._twins_at_delivery
        single = mothers_of_live - twins

        # Update counts and states: n_births counts delivery events (1 per mother, regardless of twins)
        self.parity[twins] += 1
        self.n_births[mothers_of_live] += 1
        self.n_twinbirths[twins] += 1
        self.n_stillbirths[mothers_of_stillborns] += 1

        # Record ages and update times
        self.record_ages(mothers_of_stillborns, single, twins)
        self.ti_stillbirth[mothers_of_stillborns] = self.ti
        self.ti_live_birth[mothers_of_live] = self.ti

        # Handle stillborn mothers
        self.handle_loss(mothers_of_stillborns)
        self.results['stillbirths'][self.ti] = len(mothers_of_stillborns)

        return mothers_of_live

    def record_ages(self, stillborn, single, twin):
        """
        Record maternal ages at birth/stillbirth events and compute short birth intervals.

        Args:
            stillborn: UIDs of mothers with stillbirths
            single: UIDs of mothers with singleton live births
            twin: UIDs of mothers with twin live births
        """
        ppl = self.sim.people

        # Record ages of agents when live births / stillbirths occur
        for parity in np.unique(self.n_births[single]):
            single_uids = single[self.n_births[single] == parity]
            self.birth_ages[ss.uids(single_uids), int(parity-1)] = ppl.age[ss.uids(single_uids)]
            if parity == 1: self.first_birth_age[single_uids] = ppl.age[single_uids]
        for parity in np.unique(self.n_births[twin]):
            twin_uids = twin[self.n_births[twin] == parity]
            self.birth_ages[twin_uids, int(parity-1)] = ppl.age[twin_uids]
            self.birth_ages[twin_uids, int(parity)-2] = ppl.age[twin_uids]
            if parity == 2: self.first_birth_age[twin_uids] = ppl.age[twin_uids]
        for parity in np.unique(self.n_stillbirths[stillborn]):
            uids = stillborn[self.n_stillbirths[stillborn] == parity]
            self.stillborn_ages[uids, int(parity)] = ppl.age[uids]

        # Calculate short intervals
        prev_birth_single = single[self.n_births[single] > 1]
        prev_birth_twins = twin[self.n_births[twin] > 2]
        if len(prev_birth_single):
            pidx = (self.n_births[prev_birth_single] - 1).astype(int)
            all_ints = [self.birth_ages[r, pidx] - self.birth_ages[r, pidx-1] for r in prev_birth_single]
            latest_ints = np.array([r[~np.isnan(r)][-1] for r in all_ints])
            short_ints = np.count_nonzero(latest_ints < (self.pars['short_int'].years))
            self.results['short_intervals'][self.ti] += short_ints
        if len(prev_birth_twins):
            pidx = (self.n_births[prev_birth_twins] - 2).astype(int)
            all_ints = [self.birth_ages[r, pidx] - self.birth_ages[r, pidx-1] for r in prev_birth_twins]
            latest_ints = np.array([r[~np.isnan(r)][-1] for r in all_ints])
            short_ints = np.count_nonzero(latest_ints < (self.pars['short_int'].years))
            self.results['short_intervals'][self.ti] += short_ints

        # Compute proportion of short interval births among all live births this timestep
        n_live = len(single) + len(twin)
        if n_live > 0:
            self.results['p_short_interval'][self.ti] = self.results['short_intervals'][self.ti] / n_live

        return

    def check_infant_mortality(self, uids):
        """
        Check for probability of neonatal death (< 1 year of age).

        Uses age-adjusted mortality rates to determine which newborns die.

        Args:
            uids: UIDs of mothers with live births

        Returns:
            tuple: (mothers_of_nnd, nnds) — mother UIDs and newborn UIDs that died
        """
        death_prob = (self.mortality_probs['infant'])
        if len(uids) > 0:
            age_inds = sc.findnearest(self.pars['infant_mortality']['ages'], self.sim.people.age[uids])
            death_prob = death_prob * (self.pars['infant_mortality']['age_probs'][age_inds])
        mothers_of_nnd = self.bfilter(self._p_inf_mort, uids, death_prob)
        nnds = self.find_unborn_children(mothers_of_nnd)  # Technically the children have been born, but age < dt
        return mothers_of_nnd, nnds

    def _set_contra_timing(self, uids):
        """
        Schedule contraception choice for the next timestep after delivery.

        Args:
            uids: UIDs of mothers with live births
        """
        self.ti_contra[uids] = self.ti + 1

    def handle_loss(self, uids):
        """
        Shared handler for pregnancy losses (miscarriage, stillbirth, abortion, neonatal death).

        Triggers three actions:
            1. Schedule contraceptive method update for the next timestep
            2. Schedule breastfeeding to stop next timestep
            3. Request death for unborn/newborn children

        Args:
            uids: UIDs of mothers who experienced a loss
        """
        # Trigger update to contraceptive choices
        self.ti_contra[uids] = self.ti + 1

        # Stop breastfeeding
        self.ti_stop_breastfeed[uids] = self.ti + 1  # Stop breastfeeding next month

        # Track death of unborn child/ren
        child_uids = self.find_unborn_children(uids)
        self.sim.people.request_death(child_uids)

        return

    def update_breastfeeding(self, uids):
        """
        Extend parent breastfeeding update to track cumulative duration.

        When women stop breastfeeding, adds their current breastfeeding duration
        to dur_breastfeed_total (lifetime cumulative).

        Args:
            uids: UIDs of breastfeeding women to check
        """
        stopping = super().update_breastfeeding(uids)
        if len(stopping):
            self.dur_breastfeed_total[stopping] += self.dur_breastfeed[stopping]
        return

    # %% Conception

    def set_rel_sus(self):
        """
        Set relative susceptibility to pregnancy.

        Formula: rel_sus = (1 - method_eff) * (1 - lam_eff) for susceptible women, 0 otherwise.
        This combines contraceptive efficacy and lactational amenorrhea effects.
        """
        lam_eff = self.get_lam_eff()
        method_eff = self.get_contra_eff()
        self.rel_sus[~self.susceptible] = 0  # Reset all to zero
        self.rel_sus[self.susceptible] = 1  # Reset relative susceptibility
        self.rel_sus[:] *= 1 - method_eff
        self.rel_sus[:] *= 1 - lam_eff
        return

    def get_contra_eff(self):
        """
        Build an efficacy array from each agent's current contraceptive method.

        Returns:
            np.ndarray: per-agent contraceptive efficacy (0 to 1)
        """
        cm = self.sim.connectors.contraception
        eff_array = np.array([m.efficacy for m in cm.methods.values()])
        contra_eff = eff_array[self.method]
        return contra_eff

    def get_lam_eff(self):
        """
        Compute lactational amenorrhea (LAM) efficacy adjustment.

        LAM candidates are breastfeeding, susceptible women within the data-defined
        postpartum window. LAM is switched off for anyone not breastfeeding or over
        the maximum LAM duration.

        Returns:
            np.ndarray: per-agent LAM efficacy (0 or LAM_efficacy parameter)
        """
        lam_data_max = int(max(self.pars['lactational_amenorrhea']['month']))
        lam_candidates = self.breastfeeding & self.susceptible & ((self.ti - self.ti_delivery) <= lam_data_max)
        if lam_candidates.any():
            timesteps_since_birth = (self.ti - self.ti_delivery[lam_candidates]).astype(int)
            probs = self.pars['lactational_amenorrhea']['rate'][timesteps_since_birth]
            self._p_lam.set(p=probs)
            self.lam[lam_candidates] = self._p_lam.rvs(lam_candidates)

        # Switch LAM off for anyone not breastfeeding or over max_lam_dur postpartum
        max_lam_dur = self.pars['max_lam_dur']
        over_max = (self.ti - self.ti_delivery) > max_lam_dur
        not_breastfeeding = ~self.breastfeeding
        not_lam = over_max | not_breastfeeding
        self.lam[not_lam] = False

        lam_eff = self.pars['LAM_efficacy'] * self.lam
        return lam_eff

    def make_p_conceive(self, filter_uids=None):
        """
        Calculate per-agent conception probability with FP-specific factors.

        Steps:
            1. Get individual exposure risk via rel_sus (contraception and LAM applied)
            2. Apply individual fecundity variation and age-specific fecundity
            3. Adjust for nulliparity
            4. Apply exposure factors from calibration (overall, by parity, by age)

        Args:
            filter_uids: optional UIDs to restrict to (intersected with susceptible)

        Returns:
            np.ndarray: per-agent conception probability for this timestep
        """
        uids = self.susceptible
        if filter_uids is not None: uids = filter_uids & uids

        # Apply fecundity, nulliparous adjustment, and exposure factors
        raw_probs = self._adj_fecundity(uids, None)
        raw_probs = self._adj_nullip(uids, raw_probs)
        raw_probs = self._adj_exposure(uids, raw_probs)

        # Convert to probability
        raw_probs = np.minimum(raw_probs, 1.0)
        preg_probs = ss.probperyear(raw_probs).to_prob(self.t.dt)

        return preg_probs

    def _adj_fecundity(self, uids, base_probs):
        """
        Apply age-specific fecundity and individual fecundity variation.

        Multiplies age_fecundity by personal_fecundity (individual random variation)
        and by rel_sus (which encodes contraceptive and LAM effects).

        Args:
            uids: UIDs of susceptible women
            base_probs: unused (kept for API consistency with other _adj methods)

        Returns:
            np.ndarray: fecundity-adjusted probabilities
        """
        ppl = self.sim.people
        fecundity = self.pars['age_fecundity'][ppl.int_age_clip(uids)] * self.personal_fecundity[uids]
        return fecundity * self.rel_sus[uids]

    def _adj_nullip(self, uids, probs):
        """
        Adjust conception probability for nulliparous women (parity == 0).

        Nulliparous women have a lower likelihood of conception, scaled by
        the age-specific fecundity_ratio_nullip parameter.

        Args:
            uids: UIDs of susceptible women
            probs: current conception probabilities (modified in place)

        Returns:
            np.ndarray: adjusted probabilities
        """
        ppl = self.sim.people
        nullip = self.parity[uids] == 0
        nullip_uids = uids[nullip]
        probs[nullip] *= self.pars['fecundity_ratio_nullip'][ppl.int_age_clip(nullip_uids)]
        return probs

    def _adj_exposure(self, uids, probs):
        """
        Apply calibration-derived exposure factor adjustments.

        Three multiplicative factors:
            1. Overall exposure_factor
            2. Age-specific exposure_age (interpolated spline)
            3. Parity-specific exposure_parity (interpolated spline)

        Args:
            uids: UIDs of susceptible women
            probs: current conception probabilities (modified in place)

        Returns:
            np.ndarray: adjusted probabilities
        """
        ppl = self.sim.people
        probs *= self.pars['exposure_factor']
        probs *= self.pars['exposure_age'][ppl.int_age_clip(uids)]
        probs *= self.pars['exposure_parity'][np.minimum(self.parity[uids], fpd.max_parity).astype(int)]
        return probs

    def select_conceivers(self, uids=None):
        """
        Select women who conceive, then apply abortion logic.

        Calls the parent to determine initial conceivers based on conception
        probability, then splits them into abortions and continuing pregnancies.
        Tracks method failures (women who conceived while on contraception).

        Args:
            uids: UIDs to consider (default: susceptible women)

        Returns:
            ss.uids: UIDs of women with continuing pregnancies (after abortions removed)
        """
        # Call parent to get initial conceivers
        if uids is None: uids = self.susceptible.uids
        conceived = super().select_conceivers(uids=uids)

        if len(conceived) == 0:
            return ss.uids()

        # Check for abortion
        abort_uids, preg_uids = self.bsplit(self._p_abortion, conceived, self.pars.abortion_prob)
        if len(abort_uids):
            self.handle_abortion(abort_uids)

        # Track method failures for continuing pregnancies
        if hasattr(self.sim.connectors, 'contraception') and len(preg_uids):
            on_method = self.method[preg_uids] != 0
            self.results['method_failures'][self.ti] += on_method.sum()

        return preg_uids

    def handle_abortion(self, uids):
        """
        Process abortion outcomes.

        Called prior to make_embryos/make_pregnancies, so no pregnancy state
        reset is needed. Increments abortion and pregnancy counters, records
        the age at abortion, and updates results.

        Args:
            uids: UIDs of women who aborted
        """
        ppl = self.sim.people
        n_abortions = len(uids)
        self.n_abortions[uids] += 1
        self.n_pregnancies[uids] += 1  # Still count in total pregnancies
        self.ti_abortion[uids] = self.ti

        # Track ages
        for uid in uids:
            age_idx = np.where(np.isnan(self.abortion_ages[uid]))[0]
            if len(age_idx):
                self.abortion_ages[uid, age_idx[0]] = ppl.age[uid]

        # Update results
        self.results['abortions'][self.ti] = n_abortions

        return

    def make_pregnancies(self, uids):
        """
        Create new pregnancies and disable contraception during pregnancy.

        Calls the parent to set pregnancy states, then resets contraception
        (on_contra = False, method = 0) for the newly pregnant women.

        Args:
            uids: UIDs of women who conceived

        Returns:
            np.ndarray: embryo counts per mother (from parent)
        """
        embryo_counts = super().make_pregnancies(uids)
        self.on_contra[uids] = False  # Not using contraception during pregnancy
        self.method[uids] = 0  # Method zero due to non-use
        return embryo_counts

    def process_prenatal_deaths(self, death_uids):
        """
        No-op override: FPsim handles prenatal deaths via handle_loss instead.
        """
        return

    # %% Results and finalization

    def update_results(self):
        """
        Per-step result recording.

        Extends the parent to add:
        - Infant mortality rate (IMR) = infant_deaths / births * 1000
        - Method mix via compute_method_usage
        """
        super().update_results()
        ti = self.ti
        self.results['imr'][ti] = sc.safedivide(self.results['infant_deaths'][ti], self.results['births'][ti]) * 1e3
        self.compute_method_usage()
        return

    def compute_method_usage(self):
        """
        Store the proportion of women using each contraceptive method.

        Filters to women of method-eligible age, then computes the distribution
        across method indices and stores in self.method_mix[:, ti].
        """
        ppl = self.sim.people
        min_age = self.pars.method_age
        max_age = self.pars.max_age
        bool_list_uids = ppl.female & (ppl.age >= min_age) * (ppl.age <= max_age)
        filtered_methods = self.method[bool_list_uids]
        # Use explicit bin edges to ensure each method index gets its own bin
        # bins=n creates n evenly-spaced bins from min to max, which doesn't align with method indices
        n_options = self.sim.connectors.contraception.n_options
        m_counts, _ = np.histogram(filtered_methods, bins=np.arange(n_options + 1))
        self.method_mix[:, self.ti] = m_counts / np.sum(m_counts) if np.sum(m_counts) > 0 else 0
        return

    def finalize_results(self):
        """
        Compute cumulative event counts after the simulation completes.

        Adds cum_{key} results for each event type defined in fpd.event_counts
        (e.g. cum_pregnancies, cum_births, cum_stillbirths, etc.).
        """
        super().finalize_results()
        for res in fpd.event_counts:
            self.results[f'cum_{res}'] = np.cumsum(self.results[res])
        return
