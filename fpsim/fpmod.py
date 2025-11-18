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
    """

    def __init__(self, pars=None, location=None, data=None, name='fp', **kwargs):
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

        # All other distributions
        self._fated_debut = ss.choice(a=self.pars['debut_age']['ages'], p=self.pars['debut_age']['probs'])
        self.choose_slots_twins = None # Initialized in init_pre

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

    def init_pre(self, sim):
        super().init_pre(sim)
        low = sim.pars.n_agents + 1
        high = int(self.pars.slot_scale*sim.pars.n_agents)
        high = np.maximum(high, self.pars.min_slots) # Make sure there are at least min_slots slots to avoid artifacts related to small populations
        self.choose_slots_twins = ss.randint(low=low, high=high, sim=sim, module=self)
        return

    @property
    def susceptible(self):
        """ Defined as sexually-active fertile women of childbearing age who are not pregnant """
        return self.fertile & (~self.pregnant) & self.sexually_active

    @property
    def end_tri1_uids(self):
        """ Return UIDs of those in their first trimester """
        end = (self.dur_gestation <= self.pars.trimesters[0]) & ((self.dur_gestation + self.dt) > self.pars.trimesters[0])
        return self.pregnant.uids[end]

    def init_results(self):
        """
        Initialize result storage. Most default results are either arrays or lists; these are
        all stored in defaults.py. Any other results with different formats can also be added here.
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
        # via eg. sim.connectors.fp.method_mix
        self.method_mix = np.zeros((self.sim.connectors.contraception.n_options, self.t.npts))
        return

    def update_mortality(self):
        """
        Update infant and maternal mortality for the sim's current year.
        Update general mortality trend as this uses a spline interpolation instead of an array.
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
        Decide if agent is sexually active based either time-on-postpartum month
        or their age if not postpartum.

        Agents can revert to active or not active each timestep. Postpartum and
        general age-based data from DHS.
        """
        ppl = self.sim.people

        if uids is None:
            uids = self.alive.uids

        # Set postpartum probabilities
        pp = uids & self.postpartum.uids
        non_pp = uids - self.postpartum.uids
        timesteps_since_birth = self.ti - self.ti_delivery[pp]

        # Adjust for postpartum women's birth spacing preferences
        if len(pp):
            pref = self.pars['spacing_pref']  # Shorten since used a lot
            spacing_bins = timesteps_since_birth / pref['interval']  # Main calculation: divide the duration by the interval
            spacing_bins = np.array(np.minimum(spacing_bins, pref['n_bins']), dtype=int)  # Bound by longest bin
            probs_pp = self.pars['sexual_activity_pp']['percent_active'][timesteps_since_birth.astype(int)]
            # Adjust the probability: check the overall probability with print(pref['preference'][spacing_bins].mean())
            probs_pp *= pref['preference'][spacing_bins]
            self._p_active.set(p=probs_pp)
            self.sexually_active[pp] = self._p_active.rvs(pp)

        # Set non-postpartum probabilities
        if len(non_pp):
            self.sexually_active[non_pp] = self._p_non_pp_active.rvs(non_pp)

            # Set debut to True if sexually active for the first time
            # Record agent age at sexual debut in their memory
            never_sex = self.sexual_debut[non_pp] == 0
            now_active = self.sexually_active[non_pp] == 1
            first_debut = non_pp[now_active & never_sex]
            self.sexual_debut[first_debut] = True
            self.sexual_debut_age[first_debut] = ppl.age[first_debut]

        active_sex = (self.sexually_active[uids] == 1)
        debuted = (self.sexual_debut[uids] == 1)
        active = uids[(active_sex & debuted)]
        inactive = uids[(~active_sex & debuted)]
        self.months_inactive[active] = 0
        self.months_inactive[inactive] += 1

        return

    def get_contra_eff(self):
        """ Get contraception method mix adjustment """
        cm = self.sim.connectors.contraception
        eff_array = np.array([m.efficacy for m in cm.methods.values()])
        contra_eff = eff_array[self.method]
        return contra_eff

    def get_lam_eff(self):
        """ Get LAM efficacy adjustment """
        lam_data_max = max(self.pars['lactational_amenorrhea']['month'])
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
        not_lam = over_max & not_breastfeeding
        self.lam[not_lam] = False

        lam_eff = self.pars['LAM_efficacy'] * self.lam
        return lam_eff

    def set_rel_sus(self):
        """ Set relative susceptibility to pregnancy """
        lam_eff = self.get_lam_eff()
        method_eff = self.get_contra_eff()
        self.rel_sus[~self.susceptible] = 0  # Reset all to zero
        self.rel_sus[self.susceptible] = 1  # Reset relative susceptibility
        self.rel_sus[:] *= 1 - method_eff
        self.rel_sus[:] *= 1 - lam_eff
        return

    def updates_pre(self, uids=None, upper_age=None):
        """
        This runs prior at the beginning of each timestep, prior to calculating pregnancy exposure,
        advancing pregnancies, adding new pregnancies, or determing delivery outcomes. Here we make
        any updates that affect the risk of pregnancy or pre-term birth on this timestep. We also
        set the baseline values for newborn agents.
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

    def update_time_to_choose(self, uids=None):
        """
        Initialise the counter to determine when girls/women will have to first choose a method.
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

    def handle_loss(self, uids):
        """
        Trigger updates for women whose pregnancies do not end in a live birth/baby, due to
        miscarriage, stillbirth, abortion, neonatal death. Currently a lightweight wrapper
        for resetting contraception, but separated out so that it can be called independently.
        """
        # Trigger update to contraceptive choices
        self.ti_contra[uids] = self.ti + 1

        # Stop breastfeeding
        self.ti_stop_breastfeed[uids] = self.ti + 1  # Stop breastfeeding next month

        # Track death of unborn child/ren
        child_uids, mothers_with_twins = self._get_children(uids)
        self.sim.people.request_death(child_uids)
        self.child_uid[uids] = np.nan
        self.twin_uid[mothers_with_twins] = np.nan

        return

    def _get_children(self, uids):
        """ Get UIDs for children born to mothers in uids """
        child_uids = ss.uids(self.child_uid[uids])
        mothers_with_twins = self.twin_uid.notnan.uids & uids
        twin_uids = ss.uids(self.twin_uid[mothers_with_twins])
        return child_uids | twin_uids, mothers_with_twins

    def update_breastfeeding(self, uids):
        stopping = super().update_breastfeeding(uids)
        if len(stopping):
            self.dur_breastfeed_total[stopping] += self.dur_breastfeed[stopping]
        return

    def process_delivery(self, uids, newborn_uids):
        """
        Enhanced delivery processing with stillbirths, neonatal mortality, twins,
        and tracking of ages at events.
        """
        # Call parent to handle standard delivery processing
        uids, newborn_uids = super().process_delivery(uids, newborn_uids)

        ppl = self.sim.people
        fp_pars = self.pars  # Shorten

        # Extract probability of stillbirth adjusted for age
        still_prob = self.mortality_probs['stillbirth']
        rate_ages = fp_pars['stillbirth_rate']['ages']
        age_ind = np.searchsorted(rate_ages, ppl.age[uids], side="left")
        prev_idx_is_less = ((age_ind == len(rate_ages)) | (
                np.fabs(ppl.age[uids] - rate_ages[np.maximum(age_ind - 1, 0)]) < np.fabs(
            ppl.age[uids] - rate_ages[np.minimum(age_ind, len(rate_ages) - 1)])))
        age_ind[prev_idx_is_less] -= 1  # adjusting for quirks of np.searchsorted
        still_prob = still_prob * (fp_pars['stillbirth_rate']['age_probs'][age_ind]) if len(self) > 0 else 0

        # Sort into stillbirths and live births (single and twin) and record times
        self._p_stillbirth.set(p=still_prob)
        stillborn, live = self._p_stillbirth.split(uids)

        # Sort mothers into single and multiple births
        _, twins = self._get_children(live)
        single = live - twins
        self.parity[twins] += 1  # Increment parity for twins
        self.n_births[live] += 1
        self.n_twinbirths[twins] += 1
        self.record_ages(stillborn, single, twins)

        # Update states for mothers of stillborns
        self.n_stillbirths[stillborn] += 1  # Track number of stillbirths for each woman
        self.handle_loss(stillborn)  # State updates for mothers of stillborns
        self.results['stillbirths'][self.ti] = len(stillborn)

        # Update times
        self.ti_stillbirth[stillborn] = self.ti
        self.ti_live_birth[live] = self.ti

        # Handle infant mortality
        mothers_of_nnds, nnds = self.check_infant_mortality(live)
        self.handle_loss(mothers_of_nnds)  # State updates for mothers of NNDs
        self.results['infant_deaths'][self.ti] += len(nnds)

        # Calculate mothers of live babies
        mothers = uids - stillborn - mothers_of_nnds
        live_babies, _ = self._get_children(mothers)

        return live, live_babies

    def record_ages(self, stillborn, single, twin):
        """
        Record ages at stillbirth and live birth and compute short intervals
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

        return

    def check_infant_mortality(self, uids):
        """
        Check for probability of infant mortality (death < 1 year of age)
        """
        death_prob = (self.mortality_probs['infant'])
        if len(uids) > 0:
            age_inds = sc.findnearest(self.pars['infant_mortality']['ages'], self.sim.people.age[uids])
            death_prob = death_prob * (self.pars['infant_mortality']['age_probs'][age_inds])
        self._p_inf_mort.set(p=death_prob)
        mothers_of_nnd = self._p_inf_mort.filter(uids)
        nnds = ss.uids(self.child_uid[mothers_of_nnd])
        return mothers_of_nnd, nnds

    def progress_pregnancies(self):
        """
        Update ongoing pregnancies and check for miscarriage.
        """
        # Call parent to update gestational clock
        super().progress_pregnancies()

        # Check for miscarriage at end of first trimester
        if self.pregnant.any():
            end_tri1 = self.end_tri1_uids
            ppl = self.sim.people
            if len(end_tri1):
                miscarriage_probs = self.pars.miscarriage_rates[ppl.int_age_clip(end_tri1)]
                self._p_miscarriage.set(p=miscarriage_probs)
                miscarriage_uids = self._p_miscarriage.filter(end_tri1)
                if len(miscarriage_uids):
                    self.handle_miscarriage(miscarriage_uids)
        return

    def handle_miscarriage(self, uids):
        """Handle miscarriage outcomes"""
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

    def make_p_conceive(self, filter_uids=None):
        """
        Enhanced conception probability with FP-specific factors. This method:
        1. Gets individual exposure risk, as defined in set_rel_sus (this is where contraception and LAM are applied)
        2. Applies individual fecundity variation
        3. Enhances with age-specific fecundity
        4. Applies adjustments to exposure factors from calibration: overall, by parity, and by age
        """
        ppl = self.sim.people
        uids = self.susceptible
        if filter_uids is not None: uids = filter_uids & uids

        # Find monthly probability of pregnancy based on fecundity and use of contraception including LAM - from data
        pars = self.pars  # Shorten
        fecundity = pars['age_fecundity'][ppl.int_age_clip(uids)] * self.personal_fecundity[uids]
        raw_probs = fecundity * self.rel_sus[uids]

        # Adjust for decreased likelihood of conception if nulliparous vs already gravid - from PRESTO data
        nullip = self.parity[uids] == 0
        nullip_uids = uids[nullip]
        raw_probs[nullip] *= pars['fecundity_ratio_nullip'][ppl.int_age_clip(nullip_uids)]

        # Adjust for probability of exposure to pregnancy episode at this timestep based on age and parity.
        # This encapsulates background factors and is experimental and tunable.
        raw_probs *= pars['exposure_factor']
        raw_probs *= pars['exposure_age'][ppl.int_age_clip(uids)]
        raw_probs *= pars['exposure_parity'][np.minimum(self.parity[uids], fpd.max_parity).astype(int)]

        # Use a single binomial trial to check for conception successes this month
        raw_probs = np.minimum(raw_probs, 1.0)
        preg_probs = ss.probperyear(raw_probs).to_prob(self.t.dt)

        return preg_probs

    def select_conceivers(self, uids=None):
        """
        Select who conceives, with abortion logic.
        """
        # Call parent to get initial conceivers
        if uids is None: uids = self.susceptible.uids
        conceived = super().select_conceivers(uids=uids)

        if len(conceived) == 0:
            return ss.uids()

        # Check for abortion
        self._p_abortion.set(p=self.pars.abortion_prob)
        abort_uids, preg_uids = self._p_abortion.split(conceived)
        if len(abort_uids):
            self.handle_abortion(abort_uids)

        # Track method failures for continuing pregnancies
        if hasattr(self.sim, 'contraception') and len(preg_uids):
            on_method = self.method[preg_uids] != 0
            self.results['method_failures'][self.ti] += on_method.sum()

        return preg_uids

    def handle_abortion(self, uids):
        """
        Handle abortion outcomes. We don't have to reset the pregnancy states because
        this method is called prior to make_embryos or make_pregnancies
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

    def _make_twin_uids(self, conceive_uids):
        """ Helper method to link embryos to mothers """
        # Choose slots for the unborn agents
        new_slots = self.choose_slots_twins.rvs(conceive_uids)
        new_uids = self.sim.people.grow(len(new_slots), new_slots)
        return new_uids, new_slots

    def make_embryos(self, conceive_uids):
        """ Create new embryos """
        # Super call handles most things, but we need to adjust for twins
        new_uids = super().make_embryos(conceive_uids)

        # Determine who is having twins
        self._p_twins.set(p=self.pars.twins_prob)
        twin_uids, single_uids = self._p_twins.split(conceive_uids)

        # Grow the population and assign properties to twins
        new_twin_uids, new_twin_slots = self._make_twin_uids(twin_uids)
        self._set_embryo_states(twin_uids, new_twin_uids, new_twin_slots)
        self.child_uid[conceive_uids] = new_uids  # Stored for the duration of pregnancy then removed
        self.twin_uid[twin_uids] = new_twin_uids  # Link twin embryos to their mothers

        # Handle burn-in (aging embryos to ti=0)
        if self.ti < 0:
            self.sim.people.age[new_twin_uids] += -self.ti * self.sim.t.dt_year

        return new_uids | new_twin_uids

    def make_pregnancies(self, uids):
        """ Create new pregnancies """
        super().make_pregnancies(uids)
        self.on_contra[uids] = False  # Not using contraception during pregnancy
        self.method[uids] = 0  # Method zero due to non-use
        return

    def do_step(self):
        """ Perform all updates except for deaths, which are handled in finish_step """
        super().do_step()
        return

    def step_die(self, uids):
        super().step_die(uids)
        return

    def finish_step(self):
        super().finish_step()
        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        self.results['imr'][ti] = sc.safedivide(self.results['infant_deaths'][ti], self.results['new_births'][ti]) * 1e3
        return

    def finalize(self):
        super().finalize()
        return

    def finalize_results(self):
        super().finalize_results()
        for res in fpd.event_counts:
            self.results[f'cum_{res}'] = np.cumsum(self.results[res])
        return
