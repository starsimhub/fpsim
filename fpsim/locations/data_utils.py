"""
Class to load datafiles for creating a localized FPsim model
"""
import os
import numpy as np
import pandas as pd
import sciris as sc
import yaml
import fpsim as fp
from fpsim import defaults as fpd
import fpsim.shared_data as sd
from scipy import interpolate as si


sd_dir = os.path.dirname(sd.__file__)  # path to the shared_data directory


# %% Main data loader
class DataLoader:
    """ Class to load and process data files for a given location """
    def __init__(self, data_path=None, location=None):
        # Create a data dictionary keyed by module / purpose
        self.data = sc.objdict(
            fp=sc.objdict(),
            edu=sc.objdict(),
            contra=sc.objdict(),
            deaths=sc.objdict(),
            people=sc.objdict(),
        )

        # For storing data used in calibration
        self.calib_data = sc.objdict()

        # Figure out the data path
        if data_path is None:
            if location is None:
                raise ValueError("Either data_path or location must be provided to read_data")
            loc_mod = getattr(fp.locations, location)
            data_path = sc.thisdir(loc_mod, 'data')
        self.data_path = sc.makepath(data_path)
        self.location = location

        return

    def load(self, contra_mod='mid', return_data=True, load_calib=False):
        """ Load all data """
        self.load_fp_data()
        self.load_edu_data()
        self.load_contra_data(contra_mod)
        self.load_death_data()
        self.load_people_data()
        if load_calib:
            self.load_calib_data()
        if return_data:
            return self.data
        else:
            return

    def load_fp_data(self, return_data=False):
        """
        Load data used within the FP module. All of these are stored directly as parameters
        """
        fp_data = sc.objdict()
        fp_data.dur_breastfeeding = self.bf_stats()
        fp_data.debut_age = self.debut_age()
        fp_data.age_fecundity = self.female_age_fecundity()
        fp_data.fecundity_ratio_nullip = self.fecundity_ratio_nullip()
        fp_data.lactational_amenorrhea = self.lactational_amenorrhea()
        fp_data.sexual_activity = self.sexual_activity()
        sexual_activity_pp = self.sexual_activity_pp()
        fp_data.sexual_activity_pp = sexual_activity_pp
        fp_data.dur_postpartum = sexual_activity_pp['month'][-1].astype(int)
        fp_data.debut_age = self.debut_age()
        fp_data.spacing_pref = self.birth_spacing_pref()
        fp_data.abortion_prob, fp_data.twins_prob = self.scalar_probs()
        fp_data.age_partnership = self.age_partnership()
        fp_data.maternal_mortality = self.maternal_mortality()
        fp_data.infant_mortality = self.infant_mortality()
        fp_data.miscarriage_rates = self.miscarriage()
        fp_data.stillbirth_rate = self.stillbirth()
        if return_data:
            return fp_data
        else:
            self.data.fp = fp_data
            return

    def load_edu_data(self, return_data=False):
        """ Load data used within the Education module """
        edu_data = sc.objdict()
        edu_data.objective = self.education_objective()
        edu_data.attainment = self.education_attainment()
        edu_data.p_dropout = self.education_dropout_probs()
        if return_data:
            return edu_data
        else:
            self.data.edu = edu_data
            return

    def load_contra_data(self, contra_mod='mid', return_data=False):
        """ Load data used within the Contraception module """
        contra_data = sc.objdict()
        contra_data.contra_use_pars = self.process_contra_use(contra_mod)
        mc, init_dist = self.load_method_switching()
        contra_data.method_choice_pars = mc
        contra_data.init_dist = init_dist
        contra_data.dur_use_df = self.load_dur_use()
        if contra_mod == 'mid':
            contra_data.age_spline = self.age_spline('25_40')
        if return_data:
            return contra_data
        else:
            self.data.contra = contra_data
            return

    def load_death_data(self, return_data=False):
        """ Load death data used within the Death module """
        deaths_data = sc.objdict()
        deaths_data.age_mortality = self.age_mortality(data_year=2010)
        if return_data:
            return deaths_data
        else:
            self.data.deaths = deaths_data
            return

    def load_people_data(self, return_data=False):
        """ Load data used for initializing people """
        people_data = sc.objdict()
        people_data.wealth_quintile = self.wealth()
        people_data.urban_prop = self.urban_proportion()
        people_data.age_pyramid = self.age_pyramid()
        if return_data:
            return people_data
        else:
            self.data.people = people_data
            return

    def load_calib_data(self, data_path=None, data_dict=None, key=None, return_data=True):
        """ Load data used for calibration """
        if data_dict is None:
            data_dict = dict(
                basic_wb='basic_wb.yaml',
                popsize='popsize.csv',
                mcpr='cpr.csv',
                tfr='tfr.csv',
                asfr='asfr.csv',
                ageparity='ageparity.csv',
                spacing='birth_spacing_dhs.csv',
                methods='mix.csv',
                afb='afb.table.csv',
                use='use.csv',
            )

        if key is not None:
            if key in data_dict:
                filename = data_dict[key]
                return self.read_data(filename, data_path=data_path)
            else:
                errormsg = f'Key {key} not recognized; must be one of: ' + ', '.join(data_dict.keys())
                raise ValueError(errormsg)

        else:
            for key, filename in data_dict.items():
                self.calib_data[key] = self.read_data(filename, data_path=data_path)
            if return_data:
                return self.calib_data
            else:
                return

    def read_data(self, filename, data_path=None, **kwargs):
        """
        Read in a data file, trying multiple formats and locations as needed
        """
        if data_path is None:
            data_path = self.data_path
        filepath = data_path / filename
        if not os.path.exists(filepath):
            try:
                # Try one level up; likely a regional location so pull from country data
                fallback_path = data_path.parent.parent / 'data'
                filepath = fallback_path / filename
            except Exception:
                errormsg = 'Could not find data file: ' + str(filename)
                raise FileNotFoundError(errormsg)

        # What kind of file is it
        if filepath.suffix == '.obj':
            data = sc.load(filepath, **kwargs)
        elif filepath.suffix == '.json':
            data = sc.loadjson(filepath, **kwargs)
        elif filepath.suffix == '.csv':
            data = pd.read_csv(filepath, **kwargs)
        elif filepath.suffix == '.yaml':
            with open(filepath) as f:
                data = yaml.safe_load(f, **kwargs)
        else:
            errormsg = f'Unrecognized file format for: {filepath}'
            raise ValueError(errormsg)

        # If data for location is region-level data, filter by location and remove 'region' column
        if isinstance(data, pd.DataFrame) and 'region' in data.columns:
            data = data[data['region'] == self.location].reset_index(drop=True).drop('region', axis=1)

        return data

    @staticmethod
    def load_age_adjustments():
        with open(os.path.join(sd_dir, 'age_adjustments.yaml'), 'r') as f:
            adjustments = yaml.safe_load(f)
        return adjustments

    # %% Scalar pars
    def bf_stats(self):
        """ Load breastfeeding stats """
        bf_data = self.read_data('bf_stats.csv')
        bf_mean = bf_data.loc[0]['value']  # Location parameter of truncated norm distribution. Requires children's recode DHS file, see data_processing/breastfeeding_stats.R
        bf_sd = bf_data.loc[1]['value']     # Location parameter of truncated norm distribution. Requires children's recode DHS file, see data_processing/breastfeeding_stats.R
        return [bf_mean, bf_sd]

    def scalar_probs(self):
        """ Load abortion and twins probabilities """
        data = self.read_data('scalar_probs.csv')
        abortion_prob = data.loc[data['param']=='abortion_prob', 'prob'].values[0]   # From https://bmcpregnancychildbirth.biomedcentral.com/articles/10.1186/s12884-015-0621-1, % of all pregnancies calculated
        twins_prob = data.loc[data['param']=='twins_prob', 'prob'].values[0]         # From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239

        return abortion_prob, twins_prob

    # %% Demographics
    def age_partnership(self):
        """ Probabilities of being partnered at age X"""
        age_partnership_data = self.read_data('age_partnership.csv')
        partnership_dict = {
            "age": age_partnership_data["age_partner"].to_numpy(),
            "partnership_probs": age_partnership_data["percent"].to_numpy(),
        }
        return  partnership_dict

    def wealth(self):
        """ Process percent distribution of people in each wealth quintile"""
        cols = ["quintile", "percent"]
        wealth_data = self.read_data('wealth.csv', header=0, names=cols)
        return wealth_data

    def urban_proportion(self):
        """Load information about the proportion of people who live in an urban setting"""
        urban_data = self.read_data('urban.csv')
        return urban_data["mean"][0]  # Return this value as a float

    def age_pyramid(self):
        """Load age pyramid data"""
        df = self.read_data('age_pyramid.csv')
        pyramid = df.to_numpy()
        return pyramid

    # %% Mortality
    def age_mortality(self, data_year=None):
        """
        Age-dependent mortality rates taken from UN World Population Prospects 2022.  From probability of dying each year.
        https://population.un.org/wpp/
        Used CSV WPP2022_Life_Table_Complete_Medium_Female_1950-2021, Kenya, 2010
        Used CSV WPP2022_Life_Table_Complete_Medium_Male_1950-2021, Kenya, 2010
        Mortality rate trend from crude death rate per 1000 people, also from UN Data Portal, 1950-2030:
        https://population.un.org/dataportal/data/indicators/59/locations/404/start/1950/end/2030/table/pivotbylocation
        Projections go out until 2030, but the csv file can be manually adjusted to remove any projections and stop at your desired year
        """
        mortality_data = self.read_data('mortality_prob.csv')
        mortality_trend = self.read_data('mortality_trend.csv')

        if data_year is None:
            error_msg = "Please provide a data year to calculate mortality rates"
            raise ValueError(error_msg)

        mortality = {
            'ages': mortality_data['age'].to_numpy(),
            'm': mortality_data['male'].to_numpy(),
            'f': mortality_data['female'].to_numpy(),
            'year': mortality_trend['year'].to_numpy(),
            'probs': mortality_trend['crude_death_rate'].to_numpy(),
        }

        trend_ind = np.where(mortality['year'] == data_year)
        trend_val = mortality['probs'][trend_ind]

        mortality['probs'] /= trend_val  # Normalize around data year for trending
        m_mortality_spline_model = si.splrep(x=mortality['ages'],
                                             y=mortality['m'])  # Create a spline of mortality along known age bins
        f_mortality_spline_model = si.splrep(x=mortality['ages'], y=mortality['f'])
        m_mortality_spline = si.splev(fpd.spline_ages,
                                      m_mortality_spline_model)  # Evaluate the spline along the range of ages in the model with resolution
        f_mortality_spline = si.splev(fpd.spline_ages, f_mortality_spline_model)
        m_mortality_spline = np.minimum(1, np.maximum(0, m_mortality_spline))  # Normalize
        f_mortality_spline = np.minimum(1, np.maximum(0, f_mortality_spline))

        mortality['m_spline'] = m_mortality_spline
        mortality['f_spline'] = f_mortality_spline

        return mortality

    # %% Pregnancy and birth outcomes
    def miscarriage(self):
        """
        Returns a linear interpolation of the likelihood of a miscarriage
        by age, taken from data from Magnus et al BMJ 2019: https://pubmed.ncbi.nlm.nih.gov/30894356/
        Data to be fed into likelihood of continuing a pregnancy once initialized in model
        Age 0 and 5 set at 100% likelihood.  Age 10 imputed to be symmetrical with probability at age 45 for a parabolic curve
        """
        df = pd.read_csv(os.path.join(sd_dir, 'miscarriage.csv'))
        miscarriage_rates = np.array([df['age'].values, df['prob'].values])
        miscarriage_interp = fp.data2interp(miscarriage_rates, fpd.spline_preg_ages)
        return miscarriage_interp

    def stillbirth(self):
        """
        From Report of the UN Inter-agency Group for Child Mortality Estimation, 2020
        https://childmortality.org/wp-content/uploads/2020/10/UN-IGME-2020-Stillbirth-Report.pdf
        """
        df = self.read_data('stillbirths.csv')
        stillbirth_rate = {
            'year': df['year'].values,
            'probs': df['probs'].values / 1000,
        }

        # Try to load age adjustments from YAML
        adjustments = self.load_age_adjustments()
        if 'stillbirth' in adjustments:
            stillbirth_rate['ages'] = np.array(adjustments['stillbirth']['ages'])
            stillbirth_rate['age_probs'] = np.array(adjustments['stillbirth']['odds_ratios'])

        return stillbirth_rate

    def maternal_mortality(self):
        """
        From World Bank indicators for maternal mortality ratio (modeled estimate) per 100,000 live births:
        https://data.worldbank.org/indicator/SH.STA.MMRT?locations=ET

        For Senegal, data is risk of maternal death assessed at each pregnancy. Data from Huchon et al. (2013)
        prospective study on risk of maternal death in Senegal and Mali.
        Maternal deaths: The annual number of female deaths from any cause related to or aggravated by pregnancy
        or its management (excluding accidental or incidental causes) during pregnancy and childbirth or within
        42 days of termination of pregnancy, irrespective of the duration and site of the pregnancy,
        expressed per 100,000 live births, for a specified time period.
        """
        df = self.read_data('maternal_mortality.csv')
        maternal_mortality = {
            'year': df['year'].values,
            'probs': df['probs'].values / 100000,
        }
        return maternal_mortality

    def infant_mortality(self):
        """
        From World Bank indicators for infant mortality (< 1 year) for Kenya, per 1000 live births
        From API_SP.DYN.IMRT.IN_DS2_en_excel_v2_1495452.numbers
        """
        df = self.read_data('infant_mortality.csv')

        infant_mortality = {
            'year': df['year'].values,
            'probs': df['probs'].values / 1000,
        }

        # Try to load age adjustments from YAML
        adjustments = self.load_age_adjustments()
        if 'infant_mortality' in adjustments:
            infant_mortality['ages'] = np.array(adjustments['infant_mortality']['ages'])
            infant_mortality['age_probs'] = np.array(adjustments['infant_mortality']['odds_ratios'])

        return infant_mortality

    # %% Fecundity and conception
    @staticmethod
    def female_age_fecundity():
        '''
        Use fecundity rates from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
        Fecundity rate assumed to be approximately linear from onset of fecundity around age 10 (average age of menses 12.5) to first data point at age 20
        45-50 age bin estimated at 0.10 of fecundity of 25-27 yr olds
        '''
        df = pd.read_csv(os.path.join(sd_dir, 'age_fecundity.csv'))

        # Extract bins and fecundity values
        bins = df['bin'].values
        f = df['f'].values / 100    # Convert from per 100 to proportion

        fecundity_interp_model = si.interp1d(x=bins, y=f)
        fecundity_interp = fecundity_interp_model(fpd.spline_preg_ages)
        fecundity_interp = np.minimum(1, np.maximum(0, fecundity_interp))  # Normalize to avoid negative or >1 values

        return fecundity_interp

    def fecundity_ratio_nullip(self):
        """
        Returns an array of fecundity ratios for a nulliparous woman vs a gravid woman
        from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
        Approximates primary infertility and its increasing likelihood if a woman has never conceived by age
        """
        df = pd.read_csv(os.path.join(sd_dir, 'fecundity_ratio_nullip.csv'))

        # Extract data and interpolate
        fecundity_ratio_nullip = np.array([df['age'].values, df['prob'].values])
        fecundity_nullip_interp = fp.data2interp(fecundity_ratio_nullip, fpd.spline_preg_ages)

        return fecundity_nullip_interp

    def lactational_amenorrhea(self):
        """
        Returns an array of the percent of breastfeeding women by month postpartum 0-11 months who meet criteria for LAM:
        Exclusively breastfeeding (bf + water alone), menses have not returned.  Extended out 5-11 months to better match data
        as those women continue to be postpartum insusceptible.
        Kenya: From DHS Kenya 2014 calendar data
        Ethiopia: From DHS Ethiopia 2016 calendar data
        Senegal: From DHS Senegal calendar data
        """
        df = self.read_data('lam.csv')
        lactational_amenorrhea = {
            'month': df['month'].values,
            'rate': df['rate'].values,
        }
        return lactational_amenorrhea

    def sexual_activity(self):
        """
        Returns a linear interpolation of rates of female sexual activity, defined as
        percentage women who have had sex within the last four weeks.
        From STAT Compiler DHS https://www.statcompiler.com/en/
        Using indicator "Timing of sexual intercourse"
        Includes women who have had sex "within the last four weeks"
        Excludes women who answer "never had sex", probabilities are only applied to agents who have sexually debuted
        Data taken from DHS, no trend over years for now
        Onset of sexual activity probabilities assumed to be linear from age 10 to first data point at age 15
        """
        df = self.read_data('sexually_active.csv')

        sexually_active = df['probs'].values / 100  # Convert from percent to rate per woman
        activity_ages = df['age'].values
        activity_interp_model = si.interp1d(x=activity_ages, y=sexually_active)
        activity_interp = activity_interp_model(fpd.spline_preg_ages)  # Evaluate interpolation along resolution of ages

        return activity_interp

    def sexual_activity_pp(self):
        """
        Returns an array of monthly likelihood of having resumed sexual activity within 0-35 months postpartum
        Uses 2014 Kenya DHS individual recode (postpartum (v222), months since last birth, and sexual activity within 30 days.
        Data is weighted.
        Limited to 23 months postpartum (can use any limit you want 0-23 max)
        Postpartum month 0 refers to the first month after delivery
        TODO-- Add code for processing this for other countries to data_processing
        """
        df = self.read_data('sexually_active_pp.csv')
        postpartum_activity = {
            'month': df['month'].values,
            'percent_active': df['probs'].values,
        }
        return postpartum_activity

    def debut_age(self):
        """
        Returns an array of weighted probabilities of sexual debut by a certain age 10-45.
        Data taken from DHS variable v531 (imputed age of sexual debut, imputed with data from age at first union)
        Use sexual_debut_age_probs.py under locations/data_processing to output for other DHS countries
        """
        df = self.read_data('debut_age.csv')
        debut_age = {
            'ages': df['age'].values,
            'probs': df['probs'].values,
        }
        return debut_age

    def birth_spacing_pref(self):
        """
        Returns an array of birth spacing preferences by closest postpartum month.
        If the CSV file is missing, a default table with equal weights is used.
        """
        # Try to read the CSV, fallback to dummy df if not found
        try:
            df = self.read_data('birth_spacing_pref.csv')
        except FileNotFoundError:
            print(f"birth_spacing_pref.csv not found for {self.location}, using default weights of 1.")
            months = np.arange(0, 39, 3)  # 0 to 36 months in 3-month intervals
            weights = np.ones_like(months, dtype=float)
            df = pd.DataFrame({'month': months, 'weights': weights})

        # Check uniform intervals
        intervals = np.diff(df['month'].values)
        interval = intervals[0]
        assert np.all(intervals == interval), (
            f"In order to be computed in an array, birth spacing preference bins must be uniform. Got: {intervals}"
        )

        pref_spacing = {
            'interval': interval,
            'n_bins': len(intervals),
            'months': df['month'].values,
            'preference': df['weights'].values
        }

        return pref_spacing

    # %% Education
    def education_objective(self):
        """
        Read in education objective data to a DataFrame representing the proportion of women
        with different education objectives, stratified by urban/rural residence.

        The 'percent' column represents the proportion of women aiming for 'edu' years of education. The data
        is based on education completed by women over 20 with no children, stratified by urban/rural residence
        from the Demographic and Health Surveys (DHS).
        """
        return self.read_data("edu_objective.csv")

    def education_attainment(self):
        """
        Read education attainment data (from education_initialization.csv) into a DataFrame.
        The DataFrame represents the average (mean) number of years of education 'edu', a woman
        aged 'age' has attained/completed.
        NOTE: The data in education_initialization.csv have been extrapolated to cover the age range
        [0, 99], inclusive range. Here we only interpolate data for the group 15-49 (inclusive range).
        """
        df = self.read_data("edu_initialization.csv")
        return df.set_index('age')

    def education_dropout_probs(self):
        """
        Transforms education dropout probabilities (from edu_stop.csv) from a DataFrame
        into a dictionary. The dataframe will be used as the probability that a woman aged 'age'
        and with 'parity' number of children, would drop out of school if she is enrolled in
        education and becomes pregnant.

        The data comes from PMA Datalab and 'percent' represents the probability of stopping/droppping
        out of education within 1 year of the birth represented in 'parity'.
            - Of women with a first birth before age 18, 12.6% stopped education within 1 year of that birth.
            - Of women who had a subsequent (not first) birth before age 18, 14.1% stopped school within 1 year of that birth.

        Args:
            df (pd.DataFrame): Contains 'age', 'parity' and 'percent' columns.

        Returns:
           data (dictionary): Dictionary with keys '1' and '2+' indicating a woman's parity. The values are dictionaries,
               each with keys 'age' and 'percent'.
        """
        data = {}
        df = self.read_data("edu_stop.csv")
        for k in df["parity"].unique():
            data[k] = {"age": None, "percent": None}
            data[k]["age"] = df["age"].unique()
            data[k]["percent"] = df["percent"][df["parity"] == k].to_numpy()
        return data

    #%% Contraception
    def process_contra_use(self, which):
        """
        Process contraceptive use parameters.
        Args:
            which: either 'simple' or 'mid'
        """

        # Read in data
        alldfs = [
            self.read_data(f'contra_coef_{which}.csv'),
            self.read_data(f'contra_coef_{which}_pp1.csv'),
            self.read_data(f'contra_coef_{which}_pp6.csv'),
        ]

        contra_use_pars = dict()

        for di, df in enumerate(alldfs):
            if which == 'mid':
                contra_use_pars[di] = sc.objdict(
                    intercept=df[df['rhs'].str.contains('Intercept')].Estimate.values[0],
                    age_factors=df[df['rhs'].str.contains('age') & ~df['rhs'].str.contains('prior_userTRUE')].Estimate.values,
                    ever_used_contra=df[df['rhs'].str.contains('prior_userTRUE') & ~df['rhs'].str.contains('age')].Estimate.values[0],
                    edu_factors=df[df['rhs'].str.contains('edu')].Estimate.values,
                    parity=df[df['rhs'].str.contains('parity')].Estimate.values[0],
                    urban=df[df['rhs'].str.contains('urban')].Estimate.values[0],
                    wealthquintile=df[df['rhs'].str.contains('wealthquintile')].Estimate.values[0],
                    age_ever_user_factors=df[df['rhs'].str.contains('age') & df['rhs'].str.contains('prior_userTRUE')].Estimate.values,
                )

            elif which == 'simple':
                age_factors = df[df['rhs'].str.match('age') & ~df['rhs'].str.contains('prior_userTRUE')].Estimate.values
                age_factors = np.insert(age_factors, 0, 0)  # Add a zero for the 0-18 group
                contra_use_pars[di] = sc.objdict(
                    intercept=df[df['rhs'].str.contains('Intercept')].Estimate.values[0],
                    age_factors=age_factors,
                    fp_ever_user=df[df['rhs'].str.contains('prior_userTRUE') & ~df['rhs'].str.contains('age')].Estimate.values[0],
                    age_ever_user_factors=df[df['rhs'].str.match('age') & df['rhs'].str.contains('prior_userTRUE')].Estimate.values,
                )

        return contra_use_pars

    def load_method_switching(self, methods=None):
        """ Choice of method is age and previous method """

        df = self.read_data('method_mix_matrix_switch.csv', keep_default_na=False, na_values=['NaN'])

        # Get default methods - TODO, think of something better
        if methods is None:
            methods = fp.make_methods()

        csv_map = {method.csv_name: method.name for method in methods.values()}
        idx_map = {method.csv_name: method.idx for method in methods.values()}
        idx_df = {}
        for col in df.columns:
            if col in csv_map.keys():
                idx_df[col] = idx_map[col]

        mc = dict()  # This one is a dict because it will be keyed with numbers
        init_dist = sc.objdict()  # Initial distribution of method choice

        # Get end index
        ei = 4+len(methods)-1

        for pp in df.postpartum.unique():
            mc[pp] = sc.objdict()
            mc[pp].method_idx = np.array(list(idx_df.values()))
            for akey in df.age_grp.unique():
                mc[pp][akey] = sc.objdict()
                thisdf = df.loc[(df.age_grp == akey) & (df.postpartum == pp)]
                if pp == 1:  # Different logic for immediately postpartum
                    mc[pp][akey] = thisdf.values[0][4:ei].astype(float)  # If this is going to be standard practice, should make it more robust
                else:
                    from_methods = thisdf.From.unique()
                    for from_method in from_methods:
                        from_mname = csv_map[from_method]
                        row = thisdf.loc[thisdf.From == from_method]
                        mc[pp][akey][from_mname] = row.values[0][4:ei].astype(float)

                # Set initial distributions by age
                if pp == 0:
                    init_dist[akey] = thisdf.loc[thisdf.From == 'None'].values[0][4:ei].astype(float)
                    init_dist.method_idx = np.array(list(idx_df.values()))

        return mc, init_dist

    def load_dur_use(self):
        """ Process duration of use parameters"""
        df = self.read_data('method_time_coefficients.csv', keep_default_na=False, na_values=['NaN'])
        return df

    @staticmethod
    def age_spline(which):
        d = pd.read_csv(os.path.join(sd_dir, f'splines_{which}.csv'))
        d.index = d.age  # Set the age as the index
        return d



