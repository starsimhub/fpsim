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
    This class is responsible for calculating the probabilities of conceiving at each point
    in time. Those probabilities are then passed to the module responsible for tracking
    pregnancy and birth outcomes.
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
        self._p_lam = ss.bernoulli(p=0)  # Probability of LAM
        self._p_abortion = ss.bernoulli(p=0)
        self._p_active = ss.bernoulli(p=0)
        self._p_breastfeed = ss.bernoulli(p=1)  # Probability of breastfeeding, set to 1 for consistency

        def age_adjusted_non_pp_active(self, sim, uids):
            return self.pars['sexual_activity'][sim.people.int_age(uids)]
        self._p_non_pp_active = ss.bernoulli(p=age_adjusted_non_pp_active)  # Probability of being sexually active if not postpartum

        # All other distributions
        self._fated_debut = ss.choice(a=self.pars['debut_age']['ages'], p=self.pars['debut_age']['probs'])

        # Define method mix
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

        fecund = uids[ppl.female[uids] & (ppl.age[uids] < self.pars['age_limit_fecundity'])]
        ti_to_debut = ss.years(self.fated_debut[fecund]-ppl.age[fecund])/self.t.dt

        # If ti_contra is less than one timestep away, we want to also set it to 0 so floor time_to_debut.
        self.ti_contra[fecund] = np.maximum(np.floor(ti_to_debut), 0)

        # Validation
        time_to_set_contra = self.ti_contra[fecund] == 0
        if not np.array_equal(((ppl.age[fecund] - self.fated_debut[fecund]) > - self.t.dt), time_to_set_contra):
            errormsg = 'Should be choosing contraception for everyone past fated debut age.'
            raise ValueError(errormsg)
        return

    def make_p_fertility(self, uids):
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

        return p_fertility

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

    def step(self):
        """
        Perform all updates to people within a single timestep
        """
        ppl = self.sim.people
        self.rel_sus[:] = 0  # Reset relative susceptibility to pregnancy

        # Get women eligible to become pregnant
        fecund = (ppl.female & (ppl.age < self.pars['age_limit_fecundity'])).uids
        nonpreg = fecund[~self.pregnant[fecund]]

        # # Check who has reached their age at first partnership and set partnered attribute to True.
        self.start_partnership(ppl.female.uids)

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

        # Set the probability of conception
        p_fertility = self.make_p_fertility(nonpreg)  # Decide if conceives and initialize gestation counter at 0
        self.sim.pars.pregnancy.p_fertility.set(p_fertility)  # TODO get this to work

        # Add check for ti contra
        if (self.ti_contra < 0).any():
            errormsg = f'Invalid values for ti_contra at timestep {self.ti}'
            raise ValueError(errormsg)

        return

    def update_results(self):
        super().update_results()
        # TODO figure out what to add
        self.compute_method_usage()
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
