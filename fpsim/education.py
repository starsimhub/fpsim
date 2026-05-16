"""
Methods and functions related to education
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import fpsim as fp
import starsim as ss
import sciris as sc
import pandas as pd


# %% Class for updating education
class EduPars(ss.Pars):
    """
    Parameters for the education module.
    This class defines the parameters used in the education module.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.age_start = 6  # Age at which education starts
        self.age_stop = 25  # Age at which education stops - assumption
        self.init_dropout = ss.bernoulli(p=0)  # Initial dropout probability

        # Data pars
        self.attainment = None
        self.objective = None
        self.p_dropout = None

        self.update(kwargs)
        return


def make_edu_pars():
    """ Shortcut for making a new instance of ContraPars """
    return EduPars()


class Education(ss.Connector):
    def __init__(self, pars=None, location=None, data=None, **kwargs):
        """
        Initialize the Education module.
        Args:
            pars (dict): parameters for the education module
            location (str): the location to use for education data
        """
        super().__init__(name='edu')

        # Define parameters
        default_pars = EduPars()
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)

        # Probabilities of dropping out - calculated using data inputs
        self._p_dropout = ss.bernoulli(p=0)

        # Get data if not provided
        if data is None and location is not None:
            dataloader = fp.get_dataloader(location)
            data = dataloader.load_edu_data(return_data=True)
        self.update_pars(data)

        # Education states
        self.define_states(
            ss.FloatArr('objective'),  # Education objectives
            ss.FloatArr('attainment', default=0),  # Education attainment - initialized as 0, reset if data provided
            ss.BoolState('started', default=False),  # Whether education has been started
            ss.BoolState('in_school'),  # Currently in school
            ss.BoolState('completed'),  # Whether education is completed
            ss.BoolState('dropped'),  # Whether education was dropped
            ss.BoolState('interrupted', default=False),
        )

        # Store things that will be processed after sim initialization
        self.attainment_data = self.pars['attainment']
        self.dropout_data = self.pars['p_dropout']
        self._objective_dists = None
        self.set_objective_dists(self.pars['objective'])

        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        results = [
            ss.Result('mean_attainment', label='Mean education attainment', scale=False),
            ss.Result('mean_objective', label='Mean education objective', scale=False),
            ss.Result('prop_completed', label='Proportion completed education', scale=False),
            ss.Result('prop_in_school', label='Proportion in school', scale=False),
            ss.Result('prop_dropped', label='Proportion dropped', scale=False),
        ]
        self.define_results(*results)
        return

    def set_objective_dists(self, objective_data):
        """
        Return an educational objective distribution based on provided data.
        The data should be provided in the form of a pandas DataFrame with
         "edu" and "percent" as columns.
        Returns:
            An ``ss.Dist`` instance that returns an educational objective for newly created agents
        """
        if objective_data is None:
            self._objective_dists = ss.uniform(low=0, high=24, name='Educational objective distribution')
        else:
            # Process
            if isinstance(objective_data, pd.DataFrame):
                # Check whether urban is a column
                if 'urban' in objective_data.columns:
                    self._objective_dists = sc.autolist()
                    for is_urb in [0, 1]:
                        df = objective_data.loc[objective_data.urban == is_urb]
                        bins = df['edu'].values
                        props = df['percent'].values
                        self._objective_dists += ss.histogram(values=props, bins=bins, name=f'Edu obj, urban: {is_urb}')
        return

    def init_post(self):
        """
        Initialize with educational attainment based on attainment data, if provided.
        """
        super().init_post()
        self.init_objectives()  # Initialize educational objectives

        # Initialize educational attainment based on provided data
        if self.attainment_data is None:
            return
        else:
            ppl = self.sim.people
            f_uids = self.sim.people.female.uids
            if isinstance(self.attainment_data, pd.DataFrame):
                edu = self.attainment_data['edu']
                f_ages = np.floor(ppl.age[f_uids]).astype(int)
                self.attainment[f_uids] = edu[f_ages].values

        # Initialize states from provided data
        prev_started = self.attainment > 0
        self.started[prev_started] = True

        # Figure out who's completed their education
        completed_uids = self.attainment >= self.objective
        self.completed[completed_uids] = True

        # Figure out who's dropped out
        past_school_age = ((self.attainment < self.objective)
                            & (ppl.age >= self.pars.age_stop)
                            & ppl.female
                            & self.started)
        self.dropped[past_school_age] = True

        # Figure out who's still in school
        incomplete = ((self.attainment < self.objective)
                    & (ppl.age < self.pars.age_stop)
                    & (ppl.age >= self.pars.age_start)
                    & self.sim.people.female
                    & self.started
                    & ~self.dropped)
        dropped, in_school = self.pars.init_dropout.split(incomplete.uids)
        self.dropped[dropped] = True
        self.in_school[in_school] = True

        return

    def _get_uids(self, upper_age=None):
        """ Get uids of females younger than upper_age """
        people = self.sim.people
        if upper_age is None: upper_age = 1000
        within_age = people.age < upper_age
        return (within_age & people.female).uids

    def init_objectives(self, upper_age=None):
        f_uids = self._get_uids(upper_age=upper_age)
        ppl = self.sim.people
        f_uids_urban = f_uids[ppl.urban[f_uids]]
        f_uids_rural = f_uids[~ppl.urban[f_uids]]
        self.objective[f_uids_rural] = self._objective_dists[0].rvs(f_uids_rural)
        self.objective[f_uids_urban] = self._objective_dists[1].rvs(f_uids_urban)
        return

    def step(self):
        self.init_objectives(upper_age=self.t.dt.years)  # set objectives for new agents

        # All updates for education. Note, these are done in a particular order!
        self.start_education()      # Check if anyone needs to start school
        self.interrupt_education()  # Interrupt due to pregnancy
        self.dropout_education()    # Process dropouts
        self.resume_education()     # Determine who goes back to school after an interruption
        self.advance_education()    # Advance attainment
        self.graduate()             # Graduate anyone who achieves their education goal
        return

    def start_education(self):
        """
        Begin education. TODO, this assumes everyone starts, but in reality, some may not start school or start later
        """
        new_students = (~self.in_school & ~self.completed & ~self.dropped
                        & (self.sim.people.age >= self.pars["age_start"])
                        & (self.sim.people.age < self.pars["age_stop"])
                        & self.sim.people.female
                        )
        self.in_school[new_students] = True
        self.started[new_students] = True  # Track who started education
        return

    def interrupt_education(self):
        """
        Interrupt education due to pregnancy. This method hinders education progression if a
        woman is pregnant and towards the end of the first trimester
        """
        ppl = self.sim.people
        # Hinder education progression if a woman is pregnant and towards the end of the first trimester
        pregnant_students = self.in_school & ppl.fp.pregnant & (ppl.fp.gestation == self.sim.pars.fp['end_first_tri'])
        self.interrupted[pregnant_students] = True
        return

    def dropout_education(self):
        ppl = self.sim.people
        parity1 = self.dropout_data['1']
        parity2 = self.dropout_data['2+']

        # Determine who drops out based on the dropout probabilities
        mother_students = (ppl.parity > 0) & self.in_school
        if mother_students.any():
            p_drop = np.full_like(mother_students.uids, fill_value=0.0, dtype=float)
            age_cutoffs = parity1['age']

            # Parity 1
            p1_students = self.in_school & (ppl.fp.parity == 1)
            p1_age_idx = np.searchsorted(age_cutoffs, ppl.age[p1_students], "right") - 1
            p1_idx = mother_students[p1_students].nonzero()[-1]
            p_drop[p1_idx] = parity1['percent'][p1_age_idx]

            # Parity 2+
            p2_students = self.in_school & (ppl.fp.parity > 1)
            p2_age_idx = np.searchsorted(age_cutoffs, ppl.age[p2_students], "right") - 1
            p2_idx = mother_students[p2_students].nonzero()[-1]
            p_drop[p2_idx] = parity2['percent'][p2_age_idx]

            # Scale by dt
            # TODO, this logic seems flawed. What do these dropout probabilities actually represent?
            # We're interpreting them as annual probabilities, but is it actually a one-time filtering
            # operation that should be applied to women when they have a new baby?
            p_drop *= self.t.dt_year
            self._p_dropout.set(0)
            self._p_dropout.set(p_drop)
            drops_out = self._p_dropout.filter(mother_students.uids)
            self.in_school[drops_out] = False
            self.dropped[drops_out] = True

        return

    def resume_education(self):
        """
        # Basic mechanism to resume education post-pregnancy:
        # If education was interrupted due to pregnancy, resume after 9 months pospartum ()
        # TODO: check if there's any evidence supporting this assumption
        """
        ppl = self.sim.people
        # Basic mechanism to resume education post-pregnancy:
        # If education was interrupted due to pregnancy, resume after 9 months pospartum
        postpartum_students = (ppl.fp.postpartum &
                                self.interrupted &
                                ~self.completed &
                                ~self.dropped &
                                ((self.ti - ppl.fp.ti_delivery) > 9)  # 9 months postpartum
                                )
        self.interrupted[postpartum_students] = False
        return

    def advance_education(self):
        self.attainment[self.in_school] += self.t.dt_year
        return

    def graduate(self):
        completed = self.attainment >= self.objective
        self.in_school[completed.uids] = False
        self.completed[completed.uids] = True
        return

    def update_results(self):
        """ Update results for education module """
        super().update_results()
        ppl = self.sim.people
        f = ppl.female & (ppl.age >= 15)
        self.results.mean_attainment[self.ti] = np.mean(self.attainment[f])
        self.results.mean_objective[self.ti] = np.mean(self.objective[f])
        self.results.prop_completed[self.ti] = np.count_nonzero(self.completed[f]) / len(self.completed[f])
        self.results.prop_in_school[self.ti] = np.count_nonzero(self.in_school[f]) / len(self.in_school[f])
        self.results.prop_dropped[self.ti] = np.count_nonzero(self.dropped[f]) / len(self.dropped[f])
        return
