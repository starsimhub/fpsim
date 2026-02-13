"""
Define defaults for use throughout FPsim
"""

# Imports
import numpy as np
import sciris as sc
import starsim as ss
import fpsim
import fpsim.arrays as fpa


#%% Global defaults
useSI          = True
mpy            = 12   # Months per year, to avoid magic numbers
eps            = 1e-9 # To avoid divide-by-zero
min_age        = 15   # Minimum age to be considered eligible to use contraceptive methods
max_age        = 99   # Maximum age (inclusive)
max_age_preg   = 50   # Maximum age to become pregnant
max_parity     = 20   # Maximum number of children to track - also applies to abortions, miscarriages, stillbirths
max_parity_spline = 20   # Used for parity splines
location_registry = {}  # Registry for external custom locations
valid_country_locs = ['senegal', 'kenya', 'ethiopia', 'cotedivoire', 'niger', 'nigeria_kano', 'nigeria_kaduna', 'nigeria_lagos', 'pakistan_sindh']
valid_region_locs = {
    'ethiopia': ['addis_ababa', 'afar', 'amhara', 'benishangul_gumuz', 'dire_dawa', 'gambela', 'harari', 'oromia', 'snnpr', 'somali', 'tigray']
}


# Parse locations
def get_location(location, printmsg=False):

    if not location:
        print("No location specified. Available locations are: ")
        print(", ".join(valid_country_locs))
        print("To use model defaults, set test=True.")
        print(("To use a custom location, you can construct the sim by passing in a dataloader with a path to where you are keeping your data.\n"
              "Example:\n"
              "    import fpsim as fp\n"
              "    # Load your own data\n"
              "    my_data = fp.DataLoader(data_path='path-to-my-data')"
              "    sim = fp.Sim(dataloader=my_data)"))

        raise ValueError('Location must be specified. To use model defaults, set test=True.')
    location = location.lower()  # Ensure it's lowercase

    # External locations override internal ones
    if location in location_registry:
        return location

    # Define valid locations
    if location not in valid_country_locs and not any(location in v for v in valid_region_locs.values()):
        errormsg = f'Location "{location}" is not currently supported'
        raise NotImplementedError(errormsg)
    return location


def get_dataloader(location, printwarn=True):
    """ Return the data loader module """
    from . import locations as fplocs
    location = get_location(location)

    if hasattr(fplocs, location):
        dataloader = getattr(fplocs, location).dataloader()
    else:
        raise NotImplementedError(f'Could not find dataloader for {location}')
    return dataloader


def get_calib_pars(location, verbose=1):
    """ Return the calibration parameters """
    from . import locations as fplocs
    if location is None:
        return
    location = get_location(location)
    if hasattr(fplocs, location):
        calib_pars = getattr(fplocs, location).make_calib_pars()
    else:
        sc.printv(f'No calibration parameters found for {location}', thisverbose=0, verbose=verbose)
        return None
    return calib_pars

def get_test_defaults():
    """ Return the test defaults """
    defaults = {
        'n_agents': 500,
        'start': 2000,
        'stop': 2005,
        'location': 'senegal',
    }
    return defaults


# Defaults states and values of any new(born) agent unless initialized with data or other strategy
# or updated during the course of a simulation.
fpmod_states = [
    # Contraception
    ss.BoolState('on_contra', default=False),  # whether she's on contraception
    ss.IntArr('method', default=0),  # Which method to use. 0 used for those on no method
    ss.FloatArr('ti_contra', default=0),  # time point at which to set method
    ss.FloatArr('barrier', default=0),
    ss.BoolState('ever_used_contra', default=False),  # Ever been on contraception. 0 for never having used
    ss.FloatArr('rel_sus', default=0),  # Relative susceptibility to pregnancy, set to 1 for active fecund women

    # Sexual and reproductive states, all False by default and set during simulation
    ss.BoolState('lam'),
    ss.BoolState('pregnant'),
    ss.BoolState('fertile'),
    ss.BoolState('sexually_active'),
    ss.BoolState('sexual_debut'),
    ss.BoolState('lactating'),
    ss.BoolState('postpartum'),

    # Ages of key events
    ss.FloatArr('sexual_debut_age', default=-1),
    ss.FloatArr('fated_debut', default=-1),
    ss.FloatArr('first_birth_age', default=-1),

    # Counts of events
    ss.FloatArr('parity', default=0),           # Number of births including stillbirths
    ss.FloatArr('n_births', default=0),         # Number of live births
    ss.FloatArr('n_stillbirths', default=0),    # Number of stillbirths
    ss.FloatArr('n_miscarriages', default=0),   # Number of miscarriages
    ss.FloatArr('n_abortions', default=0),      # Number of abortions
    ss.FloatArr('n_pregnancies', default=0),    # Number of pregnancies, including miscarriages, stillbirths, abortions
    ss.FloatArr('months_inactive', default=0),  # TODO, what does this store?
    ss.FloatArr('short_interval', default=0),   # TODO, what does this store?

    # Durations and counters
    ss.FloatArr('gestation', default=0),  # TODO, remove?
    ss.FloatArr('remainder_months', default=0),  # TODO, remove?
    ss.FloatArr('dur_pregnancy', default=0),
    ss.FloatArr('dur_postpartum', default=0),
    ss.FloatArr('dur_breastfeed', default=0),
    ss.FloatArr('dur_breastfeed_total', default=0),

    # Timesteps of significant events
    ss.FloatArr('ti_conceived'),
    ss.FloatArr('ti_pregnant'),
    ss.FloatArr('ti_delivery'),
    ss.FloatArr('ti_last_delivery'),
    ss.FloatArr('ti_live_birth'),
    ss.FloatArr('ti_stillbirth'),
    ss.FloatArr('ti_postpartum'),
    ss.FloatArr('ti_miscarriage'),
    ss.FloatArr('ti_abortion'),
    ss.FloatArr('ti_stop_postpartum'),
    ss.FloatArr('ti_stop_breastfeeding'),
    ss.FloatArr('ti_debut'),
    ss.FloatArr('ti_dead'),

    # Fecundity
    ss.FloatArr('personal_fecundity', default=0),

    # Add these states to the people object. They are not tracked by timestep in the way other states are, so they
    # need to be added manually. Eventually these will become part of a separate module tracking pregnancies and
    # pregnancy outcomes.
    fpa.TwoDimensionalArr('birth_ages', ncols=max_parity),  # Ages at time of live births
    fpa.TwoDimensionalArr('stillborn_ages', ncols=max_parity),  # Ages at time of stillbirths
    fpa.TwoDimensionalArr('miscarriage_ages', ncols=max_parity),  # Ages at time of miscarriages
    fpa.TwoDimensionalArr('abortion_ages', ncols=max_parity),  # Ages at time of abortions

    ss.BoolState('partnered', default=False),  # Will remain at these values if use_partnership is False
    ss.FloatArr('partnership_age', default=-1),  # Will remain at these values if use_partnership is False
    # ss.State('urban', default=True),  # Urban/rural
    # ss.FloatArr('wealthquintile', default=3),  # Wealth quintile
]

# Postpartum keys to months
postpartum_map = {
    'pp0to5':   [0, 6],
    'pp6to11':  [6, 12],
    'pp12to23': [12, 24]
}

# Age bins for tracking age-specific fertility rate
age_bin_map = {
    '10-14':   [10, 15],
    '15-19':   [15, 20],
    '20-24':   [20, 25],
    '25-29':   [25, 30],
    '30-34':   [30, 35],
    '35-39':   [35, 40],
    '40-44':   [40, 45],
    '45-49':   [45, 50]
}

# Age and parity splines
spline_ages      = np.arange(max_age + 1)
spline_preg_ages = np.arange(max_age_preg + 1)
spline_parities  = np.arange(max_parity_spline + 1)

# Define allowable keys to select all (all ages, all methods, etc)
none_all_keys = [None, 'all', ':', [None], ['all'], [':']]


# Age bins for different method switching matrices -- can be overwritten by locations
method_age_map = {
    '<18':   [ 0, 18],
    '18-20': [18, 20],
    '20-25': [20, 25],
    '25-35': [25, 35],
    '>35':   [35, max_age+1], # +1 since we're using < rather than <=
}

immutable_method_age_map = {
    '<18':   [ 0, 18],
    '18-20': [18, 20],
    '20-25': [20, 25],
    '25-30': [25, 30],
    '30-35': [30, 35],
    '>35':   [35, max_age+1], # +1 since we're using < rather than <=
}

method_youth_age_map = {
    '<16': [10, 16],
    '16-17': [16, 18],
    '18-19': [18, 20],
    '20-22': [20, 23],
    '23-25': [23, 26],
    '>25': [26, max_age+1]
}

# Counts - we compute number of new events each timestep, plus number of cumulative events
event_counts = sc.autolist(
    'births',
    'stillbirths',
    'miscarriages',
    'abortions',
    'short_intervals',
    'secondary_births',
    'pregnancies',
    'total_births',
    'maternal_deaths',
    'infant_deaths',
    'method_failures',
)

people_counts = sc.autolist(
    'contra_access',
    'new_users',
    'ever_used_contra',
    'switchers',
    'n_fecund',
    'pp0to5',
    'pp6to11',
    'pp12to23',
    'parity0to1',
    'parity2to3',
    'parity4to5',
    'parity6plus',
    'nonpostpartum',
)

sim_results = sc.autolist(
    'n_urban',
    'n_wq1',
    'n_wq2',
    'n_wq3',
    'n_wq4',
    'n_wq5',
)

# Rates and other results that aren't scaled
rate_results = sc.autolist(
    'tfr',
    'mmr',
    'imr',
    'p_short_interval',
)
