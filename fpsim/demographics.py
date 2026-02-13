"""
Methods and functions related to basic demographics such as urban and
age at first partnership.
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
import starsim as ss


# %% Deaths module
class DeathPars(ss.Pars):
    """
    Parameters for the deaths module.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.age_mortality = None
        self.rate_units = 1
        self.rel_death = 1
        self.update(kwargs)
        return


def make_death_pars():
    """ Shortcut for making a new instance of ContraPars """
    return DeathPars()


class Deaths(ss.Deaths):
    """
    Subclass of ss.Deaths to handle FPsim's specific mortality implementation
    Use of this is optional, and FPsim should run with the base class as well.
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        default_pars = DeathPars()
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)
        return

    def make_p_death(self):

        sim = self.sim
        ppl = sim.people
        death_rate = np.zeros(ppl.uid.raw.shape, dtype=ss.dtypes.float)

        ind = sc.findnearest(self.pars['age_mortality']['year'], self.t.now('year'))
        trend_val = self.pars['age_mortality']['probs'][ind]
        age_mort = self.pars['age_mortality']
        f_spline = age_mort['f_spline'] * trend_val
        m_spline = age_mort['m_spline'] * trend_val
        over_one = ppl.age >= 1
        female = (over_one & ppl.female).uids
        male = (over_one & ppl.male).uids
        f_ages = ppl.int_age(female)
        m_ages = ppl.int_age(male)

        death_rate[male] = m_spline[m_ages]
        death_rate[female] = f_spline[f_ages]
        death_rate *= self.pars.rate_units * self.pars.rel_death
        death_rate = death_rate[ppl.alive.raw]  # Only apply to alive people

        # Scale from rate to probability
        death_rate = ss.peryear(death_rate)
        p_death = death_rate.to_prob(self.t.dt)  # Convert to probability per timestep
        return p_death


# %% Initialization methods
def init_partnership_states(ppl):
    """
    Initialize partnership status and expected age at first partnership by rural/urban status
    """

    # Get init values for these sociodemographic states
    partnered, partnership_age = get_partnership_init_vals(ppl)

    # Populate states
    ppl.partnered = partnered
    ppl.partnership_age = partnership_age


def get_partnership_init_vals(ppl):
    """Get initial distribution of age at first partnership from location-specific data"""
    partnership_data = ppl.pars['age_partnership']
    n = len(ppl)
    partnered = np.zeros(n, dtype=bool)
    partnership_age = np.zeros(n, dtype=float)

    # Get female agents indices and ages
    f_inds = sc.findinds(ppl.is_female)
    f_ages = ppl.age[f_inds]

    # Select age at first partnership
    partnership_age[f_inds] = np.random.choice(partnership_data['age'], size=len(f_inds),
                                               p=partnership_data['partnership_probs'])

    # Check if age at first partnership => than current age to set partnered
    p_inds = sc.findinds((f_ages >= partnership_age[f_inds]))
    partnered[f_inds[p_inds]] = True

    return partnered, partnership_age
