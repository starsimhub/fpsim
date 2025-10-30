"""
Defines the FPmod class
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

class FPmod(ss.Module):
    """
    Class for storing and updating FP-related events
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

        # Distributions: binary outcomes
        self._p_fertile = ss.bernoulli(p=1-self.pars['primary_infertility'])  # Probability that a woman is fertile, i.e. 1 - primary infertility
        self._p_miscarriage = ss.bernoulli(p=0)  # Probability of miscarriage
        self._p_mat_mort = ss.bernoulli(p=0)  # Probability of maternal mortality
        self._p_inf_mort = ss.bernoulli(p=0)  # Probability of infant mortality
        self._p_lam = ss.bernoulli(p=0)  # Probability of LAM
        self._p_conceive = ss.bernoulli(p=0)
        self._p_abortion = ss.bernoulli(p=0)
        self._p_active = ss.bernoulli(p=0)
        self._p_stillbirth = ss.bernoulli(p=0)  # Probability of stillbirth
        self._p_twins = ss.bernoulli(p=0)  # Probability of twins
        self._p_breastfeed = ss.bernoulli(p=1)  # Probability of breastfeeding, set to 1 for consistency

        def age_adjusted_non_pp_active(self, sim, uids):
            return self.pars['sexual_activity'][sim.people.int_age(uids)]
        self._p_non_pp_active = ss.bernoulli(p=age_adjusted_non_pp_active)  # Probability of being sexually active if not postpartum

        # All other distributions
        self._fated_debut = ss.choice(a=self.pars['debut_age']['ages'], p=self.pars['debut_age']['probs'])

        # Define ASFR and method mix
        self.asfr_bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100])
        self.asfr_width = self.asfr_bins[1]-self.asfr_bins[0]
        self.asfr = None  # Storing this separately from results as it has a different format
        self.method_mix = None

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

        return

    def _get_uids(self, upper_age=None, female_only=True):
        people = self.sim.people
        if upper_age is None: upper_age = 1000
        within_age = people.age <= upper_age
        if female_only:
            f_uids = (within_age & people.female).uids
            return f_uids
        else:
            uids = within_age.uids
            return uids

    def set_states(self, uids=None, upper_age=None):
        ppl = self.sim.people
        if uids is None: uids = self._get_uids(upper_age=upper_age)

        # Fertility
        self.fertile[uids] = self._p_fertile.rvs(uids)

        # Sexual activity
        # Default initialization for fated_debut; subnational debut initialized in subnational.py otherwise
        self.fated_debut[uids] = self._fated_debut.rvs(uids)
        fecund = ppl.female & (ppl.age < self.pars['age_limit_fecundity'])
        self.check_sexually_active(uids[fecund[uids]])
        self.update_time_to_choose(uids)

        # Fecundity variation
        self.personal_fecundity[uids] = self.pars.fecundity.rvs(uids)
        return

    def init_post(self):
        super().init_post()
        self.set_states()
        return

    def init_results(self):
        """
        Initialize result storage. Most default results are either arrays or lists; these are
        all stored in defaults.py. Any other results with different formats can also be added here.
        """
        super().init_results()  # Initialize the base results

        scaling_kw = dict(shape=self.t.npts, timevec=self.t.timevec, dtype=int, scale=True)
        nonscaling_kw = dict(shape=self.t.npts, timevec=self.t.timevec, dtype=float, scale=False, summarize_by='sum')

        # Add event counts - these are all integers, and are scaled by the number of agents
        # We compute new results for each event type, and also cumulative results
        for key in fpd.event_counts:
            self.results += ss.Result(key, label=key, **scaling_kw)
            self.results += ss.Result(f'cum_{key}', label=key, dtype=int, scale=False)  # TODO, check

        # Add people counts - these are all integers, and are scaled by the number of agents
        # However, for these we do not include cumulative totals
        for key in fpd.people_counts:
            self.results += ss.Result(key, label=key, **scaling_kw)

        for key in fpd.rate_results:
            self.results += ss.Result(key, label=key, **nonscaling_kw)

        # Additional results with different formats, stored separately
        # These will not be appended to sim.results, and must be accessed
        # via eg. sim.connectors.fp.method_mix
        self.method_mix = np.zeros((self.sim.connectors.contraception.n_options, self.t.npts))
        self.asfr = np.zeros((len(self.asfr_bins)-1, self.t.npts))

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
        is_pp = self.postpartum[uids]
        pp = uids[is_pp]
        non_pp = uids[(ppl.age[uids] >= self.fated_debut[uids]) & ~is_pp]
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

    def start_partnership(self, uids):
        """
        Decide if an agent has reached their age at first partnership. Age-based data from DHS.
        """
        ppl = self.sim.people
        is_not_partnered = self.partnered[uids] == 0
        reached_partnership_age = ppl.age[uids] >= self.partnership_age[uids]
        first_timers = uids[is_not_partnered & reached_partnership_age]
        self.partnered[first_timers] = True
        return

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

    def check_conception(self, uids):
        """
        Decide if person (female) becomes pregnant at a timestep.
        """
        ppl = self.sim.people
        if uids is None:
            uids = self.alive.uids

        active_uids = uids[(self.sexually_active[uids] & self.fertile[uids])]

        # Find monthly probability of pregnancy based on fecundity and use of contraception including LAM - from data
        pars = self.pars  # Shorten
        fecundity = pars['age_fecundity'][ppl.int_age_clip(active_uids)] * self.personal_fecundity[active_uids]

        # Get each woman's degree of protection against conception based on her contraception or LAM
        cm = self.sim.connectors.contraception
        eff_array = np.array([m.efficacy for m in cm.methods.values()])
        method_eff = eff_array[self.method]
        lam_eff = pars['LAM_efficacy']
        lam = self.lam[active_uids]
        lam_uids = active_uids[lam]

        # Set baseline susceptibility to pregnancy
        self.rel_sus[active_uids] = 1  # Reset relative susceptibility
        self.rel_sus[:] *= 1 - method_eff
        self.rel_sus[lam_uids] *= 1 - lam_eff
        raw_probs = fecundity * self.rel_sus[active_uids]

        # Adjust for decreased likelihood of conception if nulliparous vs already gravid - from PRESTO data
        nullip = self.parity[active_uids] == 0
        nullip_uids = active_uids[nullip]
        raw_probs[nullip] *= pars['fecundity_ratio_nullip'][ppl.int_age_clip(nullip_uids)]

        # Adjust for probability of exposure to pregnancy episode at this timestep based on age and parity.
        # This encapsulates background factors and is experimental and tunable.
        raw_probs *= pars['exposure_factor']
        raw_probs *= pars['exposure_age'][ppl.int_age_clip(active_uids)]
        raw_probs *= pars['exposure_parity'][np.minimum(self.parity[active_uids], fpd.max_parity).astype(int)]

        # Use a single binomial trial to check for conception successes this month
        raw_probs = np.minimum(raw_probs, 1.0)
        preg_probs = ss.probperyear(raw_probs).to_prob(self.t.dt)
        self._p_conceive.set(p=preg_probs)
        conceived = self._p_conceive.filter(active_uids)
        self.ti_conceived[conceived] = self.ti

        self.results['pregnancies'][self.ti] += len(conceived)  # track all pregnancies
        unintended = conceived[self.method[conceived] != 0]
        self.results['method_failures'][self.ti] += len(unintended)  # unintended pregnancies due to method failure

        # Check for abortion
        self._p_abortion.set(p=pars['abortion_prob'])
        abort, preg = self._p_abortion.split(conceived)

        # Update states
        n_aborts = len(abort)
        self.results['abortions'][self.ti] = n_aborts
        if n_aborts:
            for abort_uid in abort:
                # put abortion age in first nan slot
                abortion_age_index = np.where(np.isnan(self.abortion_ages[abort_uid]))[0][0]
                self.abortion_ages[abort_uid, abortion_age_index] = ppl.age[abort_uid]
            self.n_abortions[abort] += 1  # Add 1 to number of abortions agent has had
            self.ti_abortion[abort] = self.ti

        # Make selected agents pregnant
        self.make_pregnant(preg)

        return

    def make_pregnant(self, uids):
        """
        Update the selected agents to be pregnant. This also sets their method to no contraception
        and determines the length of pregnancy and expected time of delivery.
        """
        self.pregnant[uids] = True
        self.gestation[uids] = 1  # Start the counter at 1
        self.dur_pregnancy[uids] = self.pars.dur_pregnancy.rvs(uids)  # Set pregnancy duration
        self.reset_postpartum(uids)  # Stop lactating and postpartum status if becoming pregnant
        self.on_contra[uids] = False  # Not using contraception during pregnancy
        self.method[uids] = 0  # Method zero due to non-use

        # Set times
        self.ti_delivery[uids] = self.ti + self.dur_pregnancy[uids]  # Set time of delivery
        self.ti_pregnant[uids] = self.ti

        return

    def check_lam(self):
        """
        Check to see if postpartum agent meets criteria for
        Lactation amenorrhea method (LAM) LAM in this time step
        """
        max_lam_dur = self.pars['max_lam_dur']
        lam_candidates = self.postpartum & ((self.ti - self.ti_delivery) <= max_lam_dur)
        if lam_candidates.any():
            timesteps_since_birth = (self.ti - self.ti_delivery[lam_candidates]).astype(int)
            probs = self.pars['lactational_amenorrhea']['rate'][timesteps_since_birth]
            self._p_lam.set(p=probs)
            self.lam[lam_candidates] = self._p_lam.rvs(lam_candidates)

        # Switch LAM off for anyone not postpartum, over 5 months postpartum, or not breastfeeding
        not_postpartum = ~self.postpartum
        over5mo = (self.ti - self.ti_delivery) > max_lam_dur
        not_breastfeeding = ~self.lactating
        not_lam = not_postpartum & over5mo & not_breastfeeding
        self.lam[not_lam] = False

        return

    def update_breastfeeding(self):
        """
        Update breastfeeding status, resetting to False for anyone finished
        """
        bf_done = self.lactating & (self.ti_stop_breastfeeding <= self.ti)  # time to stop
        self.lactating[bf_done] = False
        return

    def update_postpartum(self):
        """
        Update postpartum status, resetting to False for anyone finished
        """
        pp_done = self.postpartum & (self.ti_stop_postpartum <= self.ti)  # time to stop
        self.postpartum[pp_done] = False
        return

    def progress_pregnancy(self, uids):
        """ Advance pregnancy in time and check for miscarriage """
        ppl = self.sim.people
        preg = uids[self.pregnant[uids]]
        self.gestation[preg] += self.t.dt.months

        # Check for miscarriage at the end of the first trimester
        end_first_tri = preg[(self.gestation[preg] == self.pars['end_first_tri'])]
        miscarriage_probs = self.pars['miscarriage_rates'][ppl.int_age_clip(end_first_tri)]
        self._p_miscarriage.set(p=miscarriage_probs)
        miscarriage = self._p_miscarriage.filter(end_first_tri)

        # Reset states and track miscarriages
        n_miscarriages = len(miscarriage)
        self.results['miscarriages'][self.ti] = n_miscarriages

        if n_miscarriages:
            for miscarriage_uid in miscarriage:
                # put miscarriage age in first nan slot
                miscarriage_age_index = np.where(np.isnan(self.miscarriage_ages[miscarriage_uid]))[0][0]
                self.miscarriage_ages[miscarriage_uid, miscarriage_age_index] = ppl.age[miscarriage_uid]
            self.pregnant[miscarriage] = False
            self.n_miscarriages[miscarriage] += 1  # Add 1 to number of miscarriages agent has had
            self.gestation[miscarriage] = 0  # Reset gestation counter
            self.ti_delivery[miscarriage] = np.nan  # Reset time of delivery
            self.ti_contra[miscarriage] = self.ti+1  # Update contraceptive choices
            self.ti_miscarriage[miscarriage] = self.ti  # Record the time of miscarriage

        return

    def reset_postpartum(self, uids):
        """
        Stop breastfeeding and reset durations
        """
        self.lactating[uids] = False
        self.postpartum[uids] = False
        self.dur_breastfeed[uids] = 0
        self.dur_postpartum[uids] = 0
        return

    def check_maternal_mortality(self, uids):
        """
        Check for probability of maternal mortality
        """
        prob = self.mortality_probs['maternal'] * self.pars['maternal_mortality_factor']
        self._p_mat_mort.set(p=prob)
        death = self._p_mat_mort.filter(uids)
        self.sim.people.request_death(death)
        self.results['maternal_deaths'][self.ti] += len(death)
        return

    def check_infant_mortality(self, uids):
        """
        Check for probability of infant mortality (death < 1 year of age)
        TODO: should this be removed if we are using standard death rates, which already include infant mortality?
        """
        death_prob = (self.mortality_probs['infant'])
        if len(uids) > 0:
            age_inds = sc.findnearest(self.pars['infant_mortality']['ages'], self.sim.people.age[uids])
            death_prob = death_prob * (self.pars['infant_mortality']['age_probs'][age_inds])
        self._p_inf_mort.set(p=death_prob)
        death = self._p_inf_mort.filter(uids)

        self.results['infant_deaths'][self.ti] += len(death)
        self.reset_postpartum(death)
        self.ti_contra[death] = self.ti + 1  # Trigger update to contraceptive choices following infant death
        return death

    def process_delivery(self, uids=None):
        """
        Decide if pregnant woman gives birth and explore maternal mortality and child mortality
        Also update states including parity, n_births, n_stillbirths
        """
        if uids is None:
            uids = self.pregnant.uids
        sim = self.sim
        fp_pars = self.pars
        ti = self.ti
        ppl = sim.people

        # Update states
        deliv = uids[self.pregnant[uids] & (self.ti_delivery[uids] <= self.ti)]  # Check for those who are due this timestep
        if len(deliv):  # check for any deliveries

            # Set states
            self.pregnant[deliv] = False
            self.gestation[deliv] = 0  # Reset gestation counter
            self.lactating[deliv] = True
            self.postpartum[deliv] = True  # Start postpartum state at time of birth

            # Set durations
            will_breastfeed, wont_breastfeed = self._p_breastfeed.split(deliv)
            self.dur_breastfeed[will_breastfeed] = self.pars.dur_breastfeeding.rvs(will_breastfeed)  # Draw durations
            self.dur_postpartum[deliv] = self.pars.dur_postpartum  # Set postpartum duration

            self.ti_contra[deliv] = ti + 1  # Trigger a call to re-evaluate whether to use contraception when 1month pp
            self.ti_delivery[deliv] = ti  # Record the time of delivery
            self.ti_stop_breastfeeding[will_breastfeed] = ti + self.dur_breastfeed[will_breastfeed]
            self.ti_stop_breastfeeding[wont_breastfeed] = ti + 1  # If not breastfeeding, stop lactating next timestep
            self.ti_stop_postpartum[deliv] = ti + self.dur_postpartum[deliv]

            # Handle stillbirth
            still_prob = self.mortality_probs['stillbirth']
            rate_ages = fp_pars['stillbirth_rate']['ages']

            age_ind = np.searchsorted(rate_ages, ppl.age[deliv], side="left")
            prev_idx_is_less = ((age_ind == len(rate_ages)) | (
                    np.fabs(ppl.age[deliv] - rate_ages[np.maximum(age_ind - 1, 0)]) < np.fabs(
                ppl.age[deliv] - rate_ages[np.minimum(age_ind, len(rate_ages) - 1)])))
            age_ind[prev_idx_is_less] -= 1  # adjusting for quirks of np.searchsorted
            still_prob = still_prob * (fp_pars['stillbirth_rate']['age_probs'][age_ind]) if len(self) > 0 else 0

            # Sort into stillbirths and live births and record times
            self._p_stillbirth.set(p=still_prob)
            stillborn, live = self._p_stillbirth.split(deliv)
            self.ti_live_birth[live] = ti  # Record the time of live birth
            self.ti_stillbirth[stillborn] = ti  # Record the time of stillbirth

            # Update states for mothers of stillborns
            self.lactating[stillborn] = False  # Set agents of stillbith to not lactate
            self.n_stillbirths[stillborn] += 1  # Track number of stillbirths for each woman
            self.results['stillbirths'][ti] = len(stillborn)

            # Handle twins
            self._p_twins.set(fp_pars['twins_prob'])
            twin, single = self._p_twins.split(live)
            self.results['births'][ti] += 2 * len(twin)  # only add births to population if born alive
            self.results['births'][ti] += len(single)

            # Record ages of agents when live births / stillbirths occur
            for parity in np.unique(self.parity[single]):
                single_uids = single[self.parity[single] == parity]
                # for uid in single_uids:
                self.birth_ages[ss.uids(single_uids), int(parity)] = ppl.age[ss.uids(single_uids)]
                if parity == 0: self.first_birth_age[single_uids] = ppl.age[single_uids]
            for parity in np.unique(self.parity[twin]):
                twin_uids = twin[self.parity[twin] == parity]
                # for uid in twin_uids:
                self.birth_ages[twin_uids, int(parity)] = ppl.age[twin_uids]
                self.birth_ages[twin_uids, int(parity) + 1] = ppl.age[twin_uids]
                if parity == 0: self.first_birth_age[twin_uids] = ppl.age[twin_uids]
            for parity in np.unique(self.parity[stillborn]):
                uids = stillborn[self.parity[stillborn] == parity]
                # for uid in uids:
                self.stillborn_ages[uids, int(parity)] = ppl.age[uids]

            # Update counts
            self.parity[single] += 1
            self.parity[twin] += 2  # Add 2 because matching DHS "total children ever born (alive) v201"
            self.n_births[single] += 1
            self.n_births[twin] += 2

            # Calculate short intervals
            prev_birth_single = single[self.parity[single] > 1]
            prev_birth_twins = twin[self.parity[twin] > 2]
            if len(prev_birth_single):
                pidx = (self.parity[prev_birth_single] - 1).astype(int)
                all_ints = [self.birth_ages[r, pidx] - self.birth_ages[r, pidx-1] for r in prev_birth_single]
                latest_ints = np.array([r[~np.isnan(r)][-1] for r in all_ints])
                short_ints = np.count_nonzero(latest_ints < (fp_pars['short_int'].years))
                self.results['short_intervals'][ti] += short_ints
            if len(prev_birth_twins):
                pidx = (self.parity[prev_birth_twins] - 2).astype(int)
                all_ints = [self.birth_ages[r, pidx] - self.birth_ages[r, pidx-1] for r in prev_birth_twins]
                latest_ints = np.array([r[~np.isnan(r)][-1] for r in all_ints])
                short_ints = np.count_nonzero(latest_ints < (fp_pars['short_int'].years))
                self.results['short_intervals'][ti] += short_ints

            # Calculate total births
            self.results['total_births'][ti] = len(stillborn) + self.results['births'][ti]

            # Check mortality
            self.check_maternal_mortality(live)  # Mothers of only live babies eligible to match definition of maternal mortality ratio
            i_death = self.check_infant_mortality(live)

            # Grow the population with the new live births
            new_uids = ppl.grow(len(live) - len(i_death))
            ppl.age[new_uids] = 0
            self.set_states(uids=new_uids)
            if new_uids is not None:
                return new_uids

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
            self.mortality_probs[key2] = val

        return

    def step(self):
        """
        Perform all updates to people within a single timestep
        """
        ppl = self.sim.people
        self.rel_sus[:] = 0  # Reset relative susceptibility to pregnancy

        # Update infant, maternal, and stillbirth mortality probabilities for the current year
        self.update_mortality()

        # Process delivery, including maternal and infant mortality outcomes
        self.process_delivery()  # Deliver with birth outcomes if reached pregnancy duration

        # Get women eligible to become pregnant
        fecund = (ppl.female & (ppl.age < self.pars['age_limit_fecundity'])).uids
        nonpreg = fecund[~self.pregnant[fecund]]

        # # Check who has reached their age at first partnership and set partnered attribute to True.
        self.start_partnership(ppl.female.uids)

        # Progress pregnancy, advancing gestation and handling miscarriage
        self.progress_pregnancy(self.pregnant.uids)

        # Check if agents are sexually active, and update their intent to use contraception
        self.check_sexually_active(nonpreg)

        # Update methods for those who are eligible
        ready = nonpreg[self.ti_contra[nonpreg] <= self.ti]
        if len(ready):
            self.sim.connectors.contraception.update_contra(ready)
            self.results['switchers'][self.ti] = len(ready)  # Track how many people switch methods (incl on/off)

        methods_ok = np.array_equal(self.on_contra.nonzero()[-1], self.method.nonzero()[-1])
        if not methods_ok:
            errormsg = 'Agents not using contraception are not the same as agents who are using None method'
            raise ValueError(errormsg)

        # Update states
        self.update_postpartum()  # Updates postpartum counter if postpartum
        self.update_breastfeeding()
        self.check_lam()
        self.check_conception(nonpreg)  # Decide if conceives and initialize gestation counter at 0

        # Add check for ti contra
        if (self.ti_contra < 0).any():
            errormsg = f'Invalid values for ti_contra at timestep {self.ti}'
            raise ValueError(errormsg)

        return

    def update_results(self):
        super().update_results()
        ppl = self.sim.people
        ti = self.ti
        age_min = ppl.age >= fp.min_age
        age_max = ppl.age < self.pars['age_limit_fecundity']

        self.results.n_fecund[ti] = np.sum(ppl.female * age_min * age_max)
        self.results.ever_used_contra[ti] = np.sum(self.ever_used_contra * ppl.female) / np.sum(ppl.female) * 100
        self.results.parity0to1[ti] = np.sum((self.parity <= 1) & ppl.female) / np.sum(ppl.female) * 100
        self.results.parity2to3[ti] = np.sum((self.parity >= 2) & (self.parity <= 3) & ppl.female) / np.sum(ppl.female) * 100
        self.results.parity4to5[ti] = np.sum((self.parity >= 4) & (self.parity <= 5) & ppl.female) / np.sum(ppl.female) * 100
        self.results.parity6plus[ti] = np.sum((self.parity >= 6) & ppl.female) / np.sum(ppl.female) * 100

        res = self.results
        percent0to5 = (res.pp0to5[ti] / res.n_fecund[ti]) * 100
        percent6to11 = (res.pp6to11[ti] / res.n_fecund[ti]) * 100
        percent12to23 = (res.pp12to23[ti] / res.n_fecund[ti]) * 100
        nonpostpartum = ((res.n_fecund[ti] - res.pp0to5[ti] - res.pp6to11[ti] - res.pp12to23[ti]) / res.n_fecund[ti]) * 100

        # Store results
        res['pp0to5'][ti] = percent0to5
        res['pp6to11'][ti] = percent6to11
        res['pp12to23'][ti] = percent12to23
        res['nonpostpartum'][ti] = nonpostpartum

        # Update ancillary results: ASFR and method mix
        self.compute_method_usage()
        self.compute_asfr()

        # Use ASFR results to update TFR results
        self.results.tfr[self.ti] = sum(self.asfr[:, ti])*self.asfr_width/1000
        return

    def compute_method_usage(self):
        """ Store number of women using each method """
        ppl = self.sim.people
        min_age = fpd.min_age
        max_age = self.pars['age_limit_fecundity']
        bool_list_uids = ppl.female & (ppl.age >= min_age) * (ppl.age <= max_age)
        filtered_methods = self.method[bool_list_uids]
        m_counts, _ = np.histogram(filtered_methods, bins=self.sim.connectors.contraception.n_options)
        self.method_mix[:, self.ti] = m_counts / np.sum(m_counts) if np.sum(m_counts) > 0 else 0
        return

    def compute_asfr(self):
        """
        Computes age-specific fertility rates (ASFR). Since this is calculated each timestep,
        the annualized results should compute the sum.
        """
        new_mother_uids = (self.ti_live_birth == self.ti).uids
        new_mother_ages = self.sim.people.age[new_mother_uids]
        births_by_age, _ = np.histogram(new_mother_ages, bins=self.asfr_bins)
        women_by_age, _ = np.histogram(self.sim.people.age[self.sim.people.female], bins=self.asfr_bins)
        self.asfr[:, self.ti] = sc.safedivide(births_by_age, women_by_age) * 1000
        return

    def finalize_results(self):
        super().finalize_results()
        for res in fpd.event_counts:
            self.results[f'cum_{res}'] = np.cumsum(self.results[res])

        # Aggregate the ASFR results, taking rolling 12-month sums
        asfr = np.zeros((len(self.asfr_bins)-1, self.t.npts))
        for i in range(len(self.asfr_bins)-1):
            asfr[i, (fpd.mpy-1):] = np.convolve(self.asfr[i, :], np.ones(fpd.mpy), mode='valid')
        self.asfr = asfr
        return
