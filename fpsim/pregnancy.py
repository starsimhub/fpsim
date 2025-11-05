"""
Pregnancy
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
import fpsim as fp
from . import defaults as fpd
import starsim as ss

# Specify all externally visible things this file defines
__all__ = ['Pregnancy']


# %% Define classes

class Pregnancy(ss.Demographics):
    """
    Track pregnancy and births
    """

    def __init__(self, pars=None, location=None, data=None, name='pregnancy', **kwargs):
        super().__init__(name=name)

        # Define parameters
        default_pars = fp.PregnancyPars()
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)
        self.define_states(*fp.pregnancy_states)

        # Get data parameters if not provided
        if data is None and location is not None:
            dataloader = fp.get_dataloader(location)
            data = dataloader.load_pregnancy_data(return_data=True)
        self.update_pars(data)

        # Distributions: binary outcomes
        self._p_miscarriage = ss.bernoulli(p=0)  # Probability of miscarriage
        self._p_mat_mort = ss.bernoulli(p=0)  # Probability of maternal mortality
        self._p_inf_mort = ss.bernoulli(p=0)  # Probability of infant mortality
        self._p_conceive = ss.bernoulli(p=0)  # This stays here, and the FP module updates it each timestep, but it should also be possible to run this module without the FP module
        self._p_abortion = ss.bernoulli(p=0)
        self._p_stillbirth = ss.bernoulli(p=0)  # Probability of stillbirth
        self._p_twins = ss.bernoulli(p=0)  # Probability of twins
        self._p_breastfeed = ss.bernoulli(p=1)  # Probability of breastfeeding, set to 1 for consistency

        # Define ASFR and method mix
        self.asfr_bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100])
        self.asfr_width = self.asfr_bins[1]-self.asfr_bins[0]
        self.asfr = None  # Storing this separately from results as it has a different format

        return

    def init_results(self):
        """
        Initialize result storage. Most default results are either arrays or lists; these are
        all stored in defaults.py. Any other results with different formats can also be added here.
        """
        super().init_results()  # Initialize the base results
        self.asfr = np.zeros((len(self.asfr_bins)-1, self.t.npts))
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
        self.on_contra[uids] = False  # Not using contraception during pregnancy  TODO: move this
        self.method[uids] = 0  # Method zero due to non-use  TODO: move this

        # Set times
        self.ti_delivery[uids] = self.ti + self.dur_pregnancy[uids]  # Set time of delivery
        self.ti_pregnant[uids] = self.ti

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
        Perform all updates to pregnancy
        """
        # Update infant, maternal, and stillbirth mortality probabilities for the current year
        self.update_mortality() 

        # Progress pregnancy, advancing gestation and handling miscarriage
        self.progress_pregnancy(self.pregnant.uids)

        # Process delivery, including maternal and infant mortality outcomes
        # TODO integrate this with make_embryos from ss.Pregnancy
        self.process_delivery()  # Deliver with birth outcomes if reached pregnancy duration

        # Update states
        self.update_postpartum()  # Updates postpartum counter if postpartum
        self.update_breastfeeding()

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
        self.compute_asfr()

        # Use ASFR results to update TFR results
        self.results.tfr[self.ti] = sum(self.asfr[:, ti])*self.asfr_width/1000
        return

    def make_p_fertility(self, filter_uids=None):
        """ Store here or in FPmod? """
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
