"""
Defines the FPmod class, FPsim's pregnancy: Starsim's declared pregnancy, with FPsim's own conception, losses, infant
mortality, sexual activity, and records.
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
    FPsim's pregnancy module. Starsim's declared pregnancy (`ss.Pregnancy`) does conception, abortion, miscarriage,
    stillbirth, delivery, twins, breastfeeding, maternal deaths, and infant deaths; this module gives the rates of
    conception (`conception_rate()`), of losses (`get_loss_rate()`), and of infant deaths (`get_infant_rate()`), and
    adds sexual activity, the timing of contraceptive choices, lactational amenorrhea (LAM), and FPsim's records of
    each event (with `add_callback()`).
    """

    # %% Initialization

    def __init__(self, pars=None, location=None, data=None, name='fp', **kwargs):
        """
        Initialize FPmod with parameters, data, and distributions.

        Args:
            pars (dict): FP parameters to override defaults (FPsim's, in fp.FPPars, or Starsim's pregnancy's)
            location (str): location name for loading data (e.g. 'senegal')
            data (dict): pre-loaded FP data; if None, loaded from location
            name (str): module name (default 'fp')
            **kwargs: additional parameters
        """
        fp_pars = fp.FPPars()
        fp_pars.update(pars, **kwargs)
        if data is None and location is not None:
            dataloader = fp.get_dataloader(location)
            data = dataloader.load_fp_data(return_data=True)
        fp_pars.update(data)

        # Starsim's pregnancy takes its own parameters; FPsim's are defined after
        preg_keys = ss.Pregnancy().pars.keys()
        preg_pars = {k:v for k,v in fp_pars.items() if k in preg_keys}
        twins = fp_pars.twins_prob or 0
        super().__init__(name=name, p_abortion=fp_pars.abortion_prob or 0, embryos_per_pregnancy=ss.choice(a=[1, 2], p=[1-twins, twins]), **preg_pars)
        self.define_pars(**{k:v for k,v in fp_pars.items() if k not in preg_keys})
        self.pars.update(loss_rate=self.get_loss_rate, infant_rate=self.get_infant_rate)
        self.define_states(*fp.fpmod_states)

        # Distributions specific to FPmod
        self._p_lam = ss.bernoulli(p=0)  # Probability of LAM
        self._p_active = ss.bernoulli(p=0)
        def age_adjusted_non_pp_active(self, sim, uids):
            return self.pars['sexual_activity'][sim.people.int_age(uids)]
        self._p_non_pp_active = ss.bernoulli(p=age_adjusted_non_pp_active)  # Probability of being sexually active if not postpartum
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

        # FPsim's records of the pregnancy's events
        self.add_callback(self.conception, self.on_conception)
        self.add_callback(self.abortion, self.on_abortion)
        self.add_callback(self.miscarriage, self.on_miscarriage)
        self.add_callback(self.stillbirth, self.on_stillbirth)
        self.add_callback(self.birth, self.on_birth)
        self.add_callback(self.infant_death, self.on_infant_death)

        # Non-standard results - consider moving to the contraception module
        self.method_mix = None
        return

    def init_post(self):
        """ Initialize FPsim's values for everyone, before women are made pregnant at the start (burnin) """
        super().init_post()
        self.update_fp(uids=self._female_uids())
        self.init_intent_states((self.fecund & ~self.P).uids)
        return

    def init_results(self):
        """
        Initialize results. Starsim's pregnancy has the flows (conception, birth, stillbirth, miscarriage, abortion,
        infant_mortality) and maternal deaths; FPsim adds its own counts and rates.
        """
        super().init_results()
        scaling_kw = dict(dtype=int, scale=True)
        nonscaling_kw = dict(dtype=float, scale=False)
        results = sc.autolist()

        # Add event counts - these are all integers, and are scaled by the number of agents
        for key in fpd.event_counts:
            results += ss.Result(key, label=key, kind='flow', **scaling_kw, summarize_by='sum')

        # Add people counts - these are all integers, and are scaled by the number of agents
        for key in fpd.people_counts:
            results += ss.Result(key, label=key, **scaling_kw, summarize_by='sum')

        # Add infant mortality and short interval births - these are proportions, not scaled
        results += ss.Result('imr', label='Infant mortality rate', kind='ratio', **nonscaling_kw, summarize_by='mean')
        results += ss.Result('p_short_interval', label='Proportion of short interval births', kind='ratio', **nonscaling_kw, summarize_by='mean')

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
                self.intent_to_use[uid] = (int(choice) == 1)
            else:
                # Default if age not in data
                self.intent_to_use[uid] = False
        return

    # %% Properties

    @property
    def fecund(self):
        """ Women of childbearing age """
        ppl = self.sim.people
        return (ppl.age >= self.pars.min_age) & (ppl.age <= self.pars.max_age) & ppl.female

    @property
    def postpartum(self):
        """
        Women within dur_postpartum of a delivery who are not pregnant. This does not directly affect any other
        functionality within this module, but is provided for modules that need to know who is postpartum.
        """
        steps_since_delivery = self.ti - self.ti_delivered
        pp_steps = int(self.pars.dur_postpartum/self.t.dt)
        return ~self.P & (steps_since_delivery >= 0) & (steps_since_delivery <= pp_steps)

    def _end_tri_uids(self, end):
        """ Pregnant women in the last step of a trimester ending at gestational age `end` """
        uids = self.P.uids
        ga = (self.ti - self.ti_P[uids])*self.t.dt # The gestational age at the start of the step
        return uids[(ga <= end) & (ga + self.t.dt > end)]

    @property
    def end_tri1_uids(self):
        """ Return UIDs of those in the final timestep of their first trimester """
        return self._end_tri_uids(self.pars.trimesters[0])

    @property
    def end_tri2_uids(self):
        """ Return UIDs of those in the final timestep of their second trimester """
        return self._end_tri_uids(self.pars.trimesters[1])

    # %% Utilities

    def _female_uids(self, upper_age=None):
        """ Females, optionally at or below an age """
        people = self.sim.people
        if upper_age is None: upper_age = 1000
        return ((people.age <= upper_age) & people.female).uids

    # %% Updates at the start of each step

    def step(self):
        """ FPsim's updates (sexual activity, contraception), then Starsim's pregnancy, which calls FPsim's records of its events """
        self.update_fp(uids=self._female_uids(upper_age=self.t.dt_year))
        super().step()
        return

    def update_fp(self, uids):
        """
        Updates at the start of each step, before conception. Steps performed:
            1. Set fecundity and fated debut for new agents, and update mortality rates for the current year
            2. Set sexual activity status (postpartum and non-postpartum), and lactational amenorrhea
            3. Initialize contraception timing for new agents
            4. Initialize fertility intent for women entering the 15-49 age range
            5. Update contraceptive intent and method choices

        Args:
            uids: UIDs of the new agents to initialize
        """
        # Update fecundity and mortality
        self.personal_fecundity[uids] = self.pars.fecundity.rvs(uids)
        self.update_mortality()

        # Sexual activity
        # Default initialization for fated_debut; subnational debut initialized in subnational.py otherwise
        self.fated_debut[uids] = self._fated_debut.rvs(uids)
        nonpreg = (self.fecund & ~self.P).uids
        self.check_sexually_active(nonpreg)  # Non-pregnant fecund women only
        self.update_lam()
        self.update_time_to_choose(uids)

        # Initialize fertility intent for women who just turned 15 (entered the 15-49 age range this timestep)
        ppl = self.sim.people
        just_15 = nonpreg[(ppl.age[nonpreg] > fpd.min_age) & (ppl.age[nonpreg] <= fpd.min_age + self.t.dt_year)]
        if len(just_15) > 0:
            self.init_intent_states(just_15)

        # Update intent to use on birthday for any non-preg or >1m pp
        self.sim.connectors.contraception.update_intent_to_use(nonpreg)

        # Update methods for those who are eligible
        method_updaters = ((self.ti_contra <= self.ti) & self.fecund & ~self.P).uids
        if len(method_updaters):
            self.sim.connectors.contraception.update_contra(method_updaters)
            self.results['switchers'][self.ti] = len(method_updaters)  # How many people switch methods (incl on/off)
        methods_ok = np.array_equal(self.on_contra.nonzero()[-1], self.method.nonzero()[-1])
        if not methods_ok:
            errormsg = 'Agents not using contraception are not the same as agents who are using None method'
            raise ValueError(errormsg)
        return

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

        Categorizes women into parous (have had a birth) and non-parous groups.
        Parous women with a recorded live birth get birth spacing preferences
        applied to their sexual activity rate. Non-parous women past debut age
        get age-based rates. Pre-debut women are not touched (stay inactive).

        Args:
            uids: UIDs to check (should be non-pregnant fecund women)
        """
        if uids is None:
            uids = self.alive.uids

        ppl = self.sim.people
        is_pp = self.postpartum[uids]  # Positional bool array (not a BoolArr)
        debuted_age = ppl.age[uids] >= self.fated_debut[uids]  # Positional bool array
        parous = uids[self.parity[uids] > 0]
        non_parous = uids[(self.parity[uids] == 0) & debuted_age & ~is_pp]

        # Parous women: split by whether they have a recorded live birth
        has_live_birth = parous & self.ti_live_birth.notnan
        no_live_birth = parous - has_live_birth

        # Parous without recorded live birth: age-based rate
        if len(no_live_birth):
            self.sexually_active[no_live_birth] = self._p_non_pp_active.rvs(no_live_birth)

        # Parous with live birth: spacing preferences applied
        if len(has_live_birth):
            self._set_parous_active(has_live_birth)

        # Non-parous women past debut: age-based rate + debut tracking
        if len(non_parous):
            self._set_non_parous_active(non_parous)

        # Nulliparous pre-debut: not touched (stay inactive)

        # Update inactivity counters
        self._update_inactive(uids)
        return

    def _set_parous_active(self, uids):
        """
        Set sexual activity for parous women with a recorded live birth.

        Applies birth spacing preferences using time since last live birth.
        Base probability depends on postpartum status: PP women use DHS
        postpartum data, non-PP women use age-based data. Both are then
        multiplied by the spacing preference weight.

        Args:
            uids: UIDs of parous women with a recorded live birth
        """
        ppl = self.sim.people
        pref = self.pars['spacing_pref']
        time_since_birth = self.ti - self.ti_live_birth[uids]

        # Spacing weights from time since last live birth
        spacing_bins = time_since_birth / pref['interval']
        spacing_bins = np.array(np.minimum(spacing_bins, pref['n_bins'] - 1), dtype=int)
        sp_weights = pref['preference'][spacing_bins]

        # Base rate: PP women use PP data, non-PP use age-based data
        is_pp = self.postpartum[uids]
        pp_time = np.minimum(time_since_birth.astype(int),
                             len(self.pars['sexual_activity_pp']['percent_active']) - 1)
        base_probs = np.where(
            is_pp,
            self.pars['sexual_activity_pp']['percent_active'][pp_time],
            self.pars['sexual_activity'][ppl.int_age(uids)]
        )

        adjusted_probs = base_probs * sp_weights
        self._p_active.set(p=adjusted_probs)
        self.sexually_active[uids] = self._p_active.rvs(uids)
        return

    def _set_non_parous_active(self, uids):
        """
        Set sexual activity for non-parous women using age-specific rates.

        Also tracks sexual debut: the first time a woman becomes sexually active,
        her debut age is recorded.

        Args:
            uids: UIDs of non-parous women (nulliparous, past debut age, not PP)
        """
        ppl = self.sim.people
        self.sexually_active[uids] = self._p_non_pp_active.rvs(uids)

        # Track sexual debut
        never_sex = self.sexual_debut[uids] == 0
        now_active = self.sexually_active[uids] == 1
        first_debut = uids[now_active & never_sex]
        self.sexual_debut[first_debut] = True
        self.sexual_debut_age[first_debut] = ppl.age[first_debut]
        return

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
        return

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

    # %% Rates of conception, losses, and infant deaths, which Starsim's pregnancy reads

    def conception_rate(self, sim, uids):
        """
        FPsim's annual probability of conception for each woman, as a rate: her fecundity by age, personal fecundity,
        nulliparity, and exposure, reduced by contraception and LAM; 0 unless fertile, sexually active, and not pregnant

        Args:
            uids: the women Starsim's pregnancy asks about (those not pregnant)
        """
        ppl = sim.people
        pars = self.pars
        probs = np.zeros(len(uids))
        susceptible = self.fecund[uids] & ~self.infertile[uids] & self.sexually_active[uids] & ~self.P[uids]
        sus = uids[susceptible]
        iage = ppl.int_age_clip(sus)
        p = pars['age_fecundity'][iage] * self.personal_fecundity[sus] * (1 - self.contra_eff(sus)) * (1 - pars['LAM_efficacy']*self.lam[sus])
        nullip = self.parity[sus] == 0
        p[nullip] *= pars['fecundity_ratio_nullip'][iage[nullip]]
        p *= pars['exposure_factor'] * pars['exposure_age'][iage]
        p *= pars['exposure_parity'][np.minimum(self.parity[sus], fpd.max_parity).astype(int)]
        probs[susceptible] = np.minimum(p, 1.0) # A probability of 1 is certain within the step
        return ss.probperyear(probs)

    def contra_eff(self, uids):
        """ The efficacy of each woman's contraceptive method """
        cm = self.sim.connectors.contraception
        eff_array = np.array([m.efficacy for m in cm.methods.values()])
        return eff_array[self.method[uids]]

    def update_lam(self):
        """
        Lactational amenorrhea (LAM): drawn each step for susceptible women who are breastfeeding within the
        data-defined postpartum window, and switched off for anyone not breastfeeding or over the maximum LAM duration
        """
        since = self.ti - self.ti_delivered
        lam_data_max = int(max(self.pars['lactational_amenorrhea']['month']))
        candidates = (self.fecund & ~self.infertile & self.sexually_active & ~self.P & self.lactating & (since <= lam_data_max)).uids
        if len(candidates):
            self._p_lam.set(p=self.pars['lactational_amenorrhea']['rate'][since[candidates].astype(int)])
            self.lam[candidates] = self._p_lam.rvs(candidates)
        self.lam[~(since <= self.pars['max_lam_dur']) | ~self.lactating] = False
        return

    def get_loss_rate(self, sim, uids):
        """
        The hazard of a pregnancy loss (Starsim's pregnancy makes it a miscarriage before 20 weeks, a stillbirth after):
        the first-trimester miscarriage risk by age, spread over the first trimester, and the stillbirth risk by year
        and age, spread over the rest of a 40-week pregnancy

        Args:
            uids: the women Starsim's pregnancy asks about (those pregnant)
        """
        tri1 = self.pars.trimesters[0]
        rest = ss.weeks(40) - tri1
        early = self.P.time[uids] < tri1 # The gestational age
        ages = sim.people.int_age_clip(uids)
        rate = np.zeros(len(uids))
        rate[early] = -np.log1p(-self.pars.miscarriage_rates[ages[early]])/tri1.years
        rate[~early] = -np.log1p(-self._get_stillbirth_prob(uids[~early]))/rest.years
        return ss.peryear(rate)

    def _get_stillbirth_prob(self, uids):
        """
        Compute age-adjusted stillbirth probability for the current year.

        Args:
            uids: UIDs of pregnant women

        Returns:
            np.ndarray: per-agent stillbirth probability (total over pregnancy)
        """
        ppl = self.sim.people
        pars = self.pars
        still_prob = self.mortality_probs['stillbirth']  # Year-specific base rate
        rate_ages = pars['stillbirth_rate']['ages']
        age_ind = np.searchsorted(rate_ages, ppl.age[uids], side='left')
        prev_idx_is_less = (
            (age_ind == len(rate_ages)) |
            (np.fabs(ppl.age[uids] - rate_ages[np.maximum(age_ind - 1, 0)]) <
             np.fabs(ppl.age[uids] - rate_ages[np.minimum(age_ind, len(rate_ages) - 1)]))
        )
        age_ind[prev_idx_is_less] -= 1
        return still_prob * pars['stillbirth_rate']['age_probs'][age_ind]

    def get_infant_rate(self, sim, uids):
        """
        The hazard of death before the age of 1 (Starsim's pregnancy applies it to children aged 0 to 1): the infant
        mortality of the year by the mother's age, as a probability over the first year

        Args:
            uids: the children aged 0 to 1 (Starsim's pregnancy calls it only for them)
        """
        ppl = sim.people
        probs = np.zeros(len(uids))
        infant = ppl.parent[uids] != ppl.parent.nan # Those with a mother in the sim
        kids = uids[infant]
        if len(kids):
            moms = ss.uids(ppl.parent[kids])
            ages = self.pars['infant_mortality']['ages']
            mom_age = ppl.age.raw[moms] - ppl.age.raw[kids] # The mother's age at the birth
            nearest = np.argmin(np.abs(np.subtract.outer(mom_age, ages)), axis=1)
            probs[infant] = self.mortality_probs['infant'] * self.pars['infant_mortality']['age_probs'][nearest]
        return ss.probperyear(probs)

    # %% FPsim's records of the pregnancy's events

    def on_conception(self, uids):
        """
        Contraception stops for women who conceive (and don't have an abortion at once); those on a method count as
        method failures. Sexual activity is not updated during a pregnancy (see check_sexually_active()), so it is
        unset until the step after the pregnancy ends, when it is drawn again (postpartum, after a delivery)
        """
        preg = uids[self.P[uids]]
        if len(preg):
            self.results['method_failures'][self.ti] += np.count_nonzero(self.method[preg] != 0)
            self.on_contra[preg] = False  # Not using contraception during pregnancy
            self.method[preg] = 0  # Method zero due to non-use
            self.sexually_active[preg] = False
        return

    def record_loss(self, uids, n_key, ages_key, ti_key):
        """ Record a pregnancy outcome other than a birth: its count, time, and the woman's age """
        getattr(self, ti_key)[uids] = self.ti
        idx = getattr(self, n_key)[uids].astype(int)
        keep = idx < fpd.max_parity
        getattr(self, ages_key).raw[uids[keep], idx[keep]] = self.sim.people.age[uids[keep]]
        getattr(self, n_key)[uids] += 1
        return

    def handle_loss(self, uids):
        """ After a miscarriage, stillbirth, or infant death: contraceptive choices are updated in the next step, and breastfeeding stops """
        self.ti_contra[uids] = self.ti + 1
        self.lactating[uids] = False
        return

    def on_abortion(self, uids):
        self.record_loss(uids, 'n_abortions', 'abortion_ages', 'ti_abortion')
        return

    def on_miscarriage(self, uids):
        self.record_loss(uids, 'n_miscarriages', 'miscarriage_ages', 'ti_miscarriage')
        self.handle_loss(uids)
        return

    def on_stillbirth(self, uids):
        self.record_loss(uids, 'n_stillbirths', 'stillborn_ages', 'ti_stillbirth')
        self.parity[uids] += 1 # A stillbirth counts in parity, as a live birth does
        self.handle_loss(uids)
        return

    def on_birth(self, newborns):
        """ After a live birth: twins, records, short intervals, and the timing of the mother's next contraceptive choice """
        mothers, counts = np.unique(self.sim.people.parent[newborns], return_counts=True)
        mothers = ss.uids(mothers)
        twins = mothers[counts > 1]
        single = mothers[counts == 1]

        # Update counts: parity extra +1 for twins (Starsim's pregnancy gives +1 for each delivery)
        self.parity[twins] += 1
        self.n_births[mothers] += 1
        self.n_twinbirths[twins] += 1
        self.record_ages(single, twins)

        # Contraception timing: the step after the birth
        self.ti_contra[mothers] = self.ti + 1
        return

    def on_infant_death(self, mothers):
        """ After a child dies before the age of 1 (Starsim's pregnancy stops breastfeeding) """
        self.ti_contra[mothers] = self.ti + 1
        return

    def record_ages(self, single, twin):
        """
        Record maternal ages at live births and compute short birth intervals.

        Args:
            single: UIDs of mothers with singleton live births
            twin: UIDs of mothers with twin live births
        """
        age = self.sim.people.age
        birth_ages = self.birth_ages.raw
        for uids, n in [(single, 1), (twin, 2)]: # Twins take two columns
            idx = self.n_births[uids].astype(int) + self.n_twinbirths[uids].astype(int) - 1 # The column of the last birth
            for k in range(n):
                keep = (idx - k >= 0) & (idx - k < fpd.max_parity)
                birth_ages[uids[keep], (idx - k)[keep]] = age[uids[keep]]
        first = self.n_births[single] + self.n_twinbirths[single] == 1
        self.first_birth_age[single[first]] = age[single[first]]
        first = self.n_births[twin] + self.n_twinbirths[twin] == 2
        self.first_birth_age[twin[first]] = age[twin[first]]

        # Short intervals: the time since the woman's previous live birth
        mothers = single | twin
        last = (self.n_births[mothers] + self.n_twinbirths[mothers]).astype(int) - 1
        prev = last - np.where(np.isin(mothers, twin), 2, 1)
        has_prev = (prev >= 0) & (last < fpd.max_parity)
        intervals = age[mothers[has_prev]] - birth_ages[mothers[has_prev], prev[has_prev]]
        short_ints = np.count_nonzero(intervals < self.pars['short_int'].years)
        self.results['short_intervals'][self.ti] += short_ints

        # Compute proportion of short interval births among all live births this timestep
        if len(mothers):
            self.results['p_short_interval'][self.ti] = self.results['short_intervals'][self.ti] / len(mothers)
        return

    # %% Results and finalization

    def update_results(self):
        """ Per-step results, and the method mix """
        super().update_results()
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
        n_options = self.sim.connectors.contraception.n_options
        m_counts, _ = np.histogram(filtered_methods, bins=np.arange(n_options + 1))
        self.method_mix[:, self.ti] = m_counts / np.sum(m_counts) if np.sum(m_counts) > 0 else 0
        return

    def finalize_results(self):
        """ Total births (live births plus stillbirths) and the infant mortality rate (infant deaths per 1,000 live births) """
        super().finalize_results()
        res = self.results
        res['total_births'][:] = res['birth'][:] + res['stillbirth'][:]
        res['imr'][:] = sc.safedivide(res['infant_mortality'][:], res['birth'][:], default=0) * 1e3
        return
