"""
Defines the Sim class, the core class of the FP model (FPsim).
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import pylab as pl
import seaborn as sns
import sciris as sc
import pandas as pd
import starsim as ss
import fpsim as fp
from .settings import options as fpo
from . import utils as fpu
from . import defaults as fpd

# Specify all externally visible things this file defines
__all__ = ['Sim']


#%% Plotting helper functions

def fixaxis(useSI=True, set_lim=True, legend=True):
    """ Format the axis using SI units and limits """
    if legend:
        pl.legend()  # Add legend
    if set_lim:
        sc.setylim()
    if useSI:
        sc.SIticks()
    return


def tidy_up(fig, do_show=None, do_save=None, filename=None):
    """ Helper function to handle the slightly complex logic of showing, saving, returing -- not for users """

    # Handle inputs
    if do_show is None: do_show = fpo.show
    if do_save is None: do_save = fpo.save
    backend = pl.get_backend()

    # Handle show
    if backend == 'agg':  # Cannot show plots for a non-interactive backend
        do_show = False
    if do_show:  # Now check whether to show, and atually do it
        pl.show()

    # Handle saving
    if do_save:
        if isinstance(do_save, str):  # No figpath provided - see whether do_save is a figpath
            filename = sc.makefilepath(filename)  # Ensure it's valid, including creating the folder
        sc.savefig(fig=fig, filename=filename)  # Save the figure

    # Handle close
    if fpo.close and not do_show:
        pl.close(fig)

    # Return the figure or figures unless we're in Jupyter
    if not fpo.returnfig:
        return
    else:
        return fig


# %% Sim class

class Sim(ss.Sim):
    """
    The Sim class handles the running of the simulation. It extends the Starim Sim class, so all Starsim Sim methods
    are available to FPsims.

    When a Sim is initialized, it triggers the creation of the population. Methods related
    to creating, initializing, and updating people can be found in the People class.

    Args:
        pars     (dict):   parameters; accepts parameters for all constituent modules of FPsim
        sim_pars (dict):   simulation-specific parameters; see SimPars within parameters.py for details
        fp_pars  (dict):   FP-specific parameters; FPPars within parameters.py for details
        contra_pars (dict): parameters for the contraception module; see make_contra_pars() in parameters.py for details
        edu_pars (dict):   parameters for the education module; see make_edu_pars() in parameters.py for details
        fp_module (FPmod): the family planning module; defaults to ``fp.FPmod()``
        contraception_module (ContraceptiveChoice): the contraception choice module; defaults to ``fp.StandardChoice()``
        education_module (Education): the education module; defaults to ``fp.Education()``
        people   (People):  the people object; defaults to ``fp.People()``
        demographics (list): list of demographics modules; defaults to ``[ss.Deaths()]``
        diseases (list):    list of disease modules (default None)
        networks (list):   list of network modules (default None)
        interventions (list): list of intervention modules (default None)
        analyzers (list):  list of analyzer modules (default None)
        connectors (list): list of connector modules (default None, can add empowerment or other examples)
        dataloader (Dataloader): data loader to use; defaults to the one associated with the location; see data_utils.py
        copy_inputs (bool): whether to copy input dicts (default True, recommended)
        kwargs   (dict):   additional parameters belonging to any constituent module

    **Examples**::

        sim = fp.Sim()
        sim = fp.Sim(n_agents=10e3, location='senegal', label='My small Senegal sim')
    """

    def __init__(self, pars=None, sim_pars=None, people_pars=None, fp_pars=None, contra_pars=None, edu_pars=None, death_pars=None,
                 fp_module=None, contraception_module=None, education_module=None,
                 label=None, people=None, demographics=None, diseases=None, networks=None,
                 interventions=None, analyzers=None, connectors=None, dataloader=None, copy_inputs=True, **kwargs):

        # Inputs and defaults
        self.contra_pars = None    # Parameters for the contraception module - processed later
        self.edu_pars = None     # Parameters for the education module - processed later
        self.fp_pars = None       # Parameters for the family planning module - processed later
        self.deaths_pars = None
        self.people_pars = None
        self.pars = None        # Parameters for the simulation - processed later
        self.data = None         # Data dictionary, loaded later
        self.dataloader = dataloader  # Data loader, if provided

        # Call the constructor of the parent class WITHOUT pars or module args, the make defaults
        super().__init__(pars=None, label=label)
        self.pars = fp.make_sim_pars()  # Make default parameters using values from parameters.py

        # Separate the parameters
        # Four sources of par values in decreasing order of priority:
        # 1-2. kwargs == args (if multiple definitions, raise exception)
        # 3. pars
        # 4. default pars
        # combine copies of them in this order if copy_inputs
        # remap any as necessary
        # separate into sim and fp-specific pars
        sim_kwargs = dict(label=label, people=people, demographics=demographics, diseases=diseases, networks=networks,
                    interventions=interventions, analyzers=analyzers, connectors=connectors)
        sim_kwargs = {key: val for key, val in sim_kwargs.items() if val is not None}
        all_sim_pars = self.process_pars(pars, sim_pars, people_pars, fp_pars, contra_pars, edu_pars, death_pars, sim_kwargs, **kwargs)
        self.pars.update(all_sim_pars)

        # Process modules by adding them as Starsim connectors
        default_contra = fp.StandardChoice(pars=self.contra_pars)
        default_edu = fp.Education(pars=self.edu_pars)
        default_fp = fp.FPmod(pars=self.fp_pars)
        contraception_module = contraception_module or sc.dcp(default_contra)
        education_module = education_module or sc.dcp(default_edu)
        fp_module = fp_module or sc.dcp(default_fp)
        connectors = sc.tolist(connectors) + [contraception_module, education_module, fp_module]
        self.pars['connectors'] = connectors

        # Process demographics
        if demographics is None and not len(self.pars['demographics']):
            deaths = fp.Deaths(pars=self.deaths_pars)
            self.pars['demographics'] = deaths

        # Metadata and settings
        fpu.set_metadata(self)  # Set version, date, and git info
        self.summary = None

        return

    @staticmethod
    def remap_pars(pars):
        """
        Remap the parameters to the new names. This is useful for backwards compatibility.
        """
        if 'start_year' in pars:
            pars['start'] = pars.pop('start_year')
        if 'end_year' in pars:
            pars['stop'] = pars.pop('end_year')
        if 'seed' in pars:
            pars['rand_seed'] = pars.pop('seed')
        if 'location' in pars and pars['location'] == 'test':
            pars['location'] = 'senegal'
            pars['test'] = True
        return pars

    def process_pars(self, pars=None, sim_pars=None, people_pars=None, fp_pars=None, contra_pars=None, edu_pars=None, death_pars=None, sim_kwargs=None, **kwargs):
        """
        Separate the parameters into simulation and fp-specific parameters.
        """
        # Marge in pars and kwargs
        all_pars = fp.mergepars(pars, sim_pars, people_pars, fp_pars, contra_pars, edu_pars, death_pars, sim_kwargs, kwargs)
        all_pars = self.remap_pars(all_pars)  # Remap any v2 parameters to v3 names

        # Deal with sim pars
        user_sim_pars = {k: v for k, v in all_pars.items() if k in self.pars.keys()}
        for k in user_sim_pars: all_pars.pop(k)
        sim_pars = sc.mergedicts(user_sim_pars, sim_pars, _copy=True)

        # Pull out test
        if kwargs.get('test') or (pars is not None and pars.get('test')) or (sim_pars is not None and sim_pars.get('test')):
            defaults = fpd.get_test_defaults()

            # Only apply test defaults for parameters not explicitly provided by user
            user_provided_keys = set()
            if pars: user_provided_keys.update(pars.keys())
            if sim_pars: user_provided_keys.update(sim_pars.keys())
            if kwargs: user_provided_keys.update(kwargs.keys())

            # Remove user-provided keys from defaults to avoid overriding them
            filtered_defaults = {k: v for k, v in defaults.items() if k not in user_provided_keys}

            sim_pars = sc.mergedicts(sim_pars, filtered_defaults, _copy=True)

        # Get location
        verbose = sim_pars.get('verbose', self.pars.verbose)
        veps = 0
        location = sim_pars.get('location', self.pars.location)
        is_test = sim_pars.get('test', self.pars.test)
        if location is None:
            sc.printv(f'No location specified, checking for dataloader... ', veps)

        # Load data
        if self.dataloader is None:
            self.dataloader = fpd.get_dataloader(location, printwarn=not is_test)
        if self.pars.verbose > 0:
            print(f'Loading data from files in {self.dataloader.data_path}... ')
        data_dict = self.dataloader.load()  # Load all data and sort by module

        # Load calibration parameters
        calib_pars = fpd.get_calib_pars(location, verbose=verbose)
        if calib_pars is not None:
            sc.printv(f'Applying calibration parameters for {location}...', veps)
            all_pars = fp.mergepars(all_pars, calib_pars)  # Use smart merging for calibration parameters

        # Deal with all module pars in a loop
        module_par_map = {
            'fp': (fp.make_fp_pars(), fp_pars),
            'contra': (fp.make_contra_pars(), contra_pars),
            'edu': (fp.make_edu_pars(), edu_pars),
            'deaths': (fp.make_death_pars(), death_pars),
            'people': (fp.make_people_pars(), people_pars)
        }

        for module, (module_default_pars, direct_user_pars) in module_par_map.items():
            indirect_module_pars = {k: v for k, v in all_pars.items() if k in module_default_pars.keys()}  # From pars or kwargs
            for k in indirect_module_pars: all_pars.pop(k)
            data_module_pars = data_dict.get(module, {})
            merged_pars = fp.mergepars(data_module_pars, indirect_module_pars, direct_user_pars, _copy=True)
            setattr(self, f'{module}_pars', merged_pars)

        # Raise an exception if there are any leftover pars
        if all_pars:
            raise ValueError(f'Unrecognized parameters: {all_pars.keys()}. Refer to parameters.py for parameters.')

        return sim_pars

    def init(self, force=False):
        """ Fully initialize the Sim with modules, people and result storage"""

        # Load age data and create people
        people = fp.People(self.pars.n_agents, pars=self.people_pars)
        self.pars['people'] = people

        if force or not self.initialized:
            super().init(force=force)

        return self

    def init_results(self):
        """
        Initialize the results dictionary. This is called at the start of the simulation.
        """
        super().init_results()
        scaling_kw = dict(shape=self.t.npts, timevec=self.t.timevec, dtype=int, scale=True)
        for key in fpd.sim_results:
            self.results += ss.Result(key, label=key, **scaling_kw)
        return

    # Function to scale all y-axes in fig based on input channel
    @staticmethod
    def conform_y_axes(figure, bottom=0, top=100):
        for axes in figure.axes:
            axes.set_ylim([bottom, top])
        return figure

    def plot(self, key=None, to_plot=None, xlims=None, ylims=None, do_save=None, do_show=True, filename='fpsim.png', style=None,
             fig=None, fig_args=None, fig_kw=None, plot_kw=None, alpha=None, is_jupyter=False,
             plot_args=None, axis_args=None, fill_args=None, label=None, new_fig=True, colors=None):
        """
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
            key   (str/dict): Alias for to_plot
            to_plot   (str/dict): What to plot (e.g. 'default' or 'cpr'), or a dictionary of result:label pairs
            xlims     (list/dict): passed to pl.xlim() (use ``[None, None]`` for default)
            ylims     (list/dict): passed to pl.ylim()
            do_save   (bool): Whether or not to save the figure. If a string, save to that filename.
            do_show   (bool): Whether to show the plots at the end
            filename  (str):  If a figure is saved, use this filename
            style     (bool): Custom style arguments
            fig_args  (dict): Passed to pl.figure() (plus ``nrows`` and ``ncols`` for overriding defaults)
            plot_args (dict): Passed to pl.plot()
            axis_args (dict): Passed to pl.subplots_adjust()
            fill_args (dict): Passed to pl.fill_between())
            label     (str):  Label to override default
            new_fig   (bool): Whether to create a new figure (true unless part of a multisim)
            colors    (list/dict): Colors for plots with multiple lines
        """
        if to_plot is None:
            if key is not None: to_plot = key
            else: to_plot = 'default'
        fig_args = sc.mergedicts(dict(figsize=(16, 10), nrows=None, ncols=None), fig_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.7), plot_args)
        axis_args = sc.mergedicts(dict(left=0.1, bottom=0.05, right=0.9, top=0.97, wspace=0.2, hspace=0.25), axis_args)
        fill_args = sc.mergedicts(dict(alpha=0.2), fill_args)

        with fpo.with_style(style):

            nrows, ncols = fig_args.pop('nrows'), fig_args.pop('ncols')
            fig = pl.figure(**fig_args) if new_fig else pl.gcf()
            pl.subplots_adjust(**axis_args)

            if to_plot is not None and 'as_' in to_plot:
                nrows, ncols = 2, 3

            res = self.results  # Shorten since heavily used

            if isinstance(to_plot, dict):
                pass
            elif isinstance(to_plot, str):
                if to_plot == 'default':
                    to_plot = {
                        'contraception.mcpr':   'Modern contraceptive prevalence rate (%)',
                        'fp.cum_births':           'Live births',
                        'fp.cum_stillbirths':      'Stillbirths',
                        'fp.cum_maternal_deaths':  'Maternal deaths',
                        'fp.cum_infant_deaths':    'Infant deaths',
                        'fp.imr':                  'Infant mortality rate',
                    }
                elif to_plot == 'cpr':
                    to_plot = {
                        'contraception.mcpr': 'MCPR (modern contraceptive prevalence rate)',
                        'contraception.cpr':  'CPR (contraceptive prevalence rate)',
                        'contraception.acpr': 'ACPR (alternative contraceptive prevalence rate)',
                    }
                elif to_plot == 'mortality':
                    to_plot = {
                        'contraception.mmr':                 'Maternal mortality ratio',
                        'fp.cum_maternal_deaths': 'Maternal deaths',
                        'fp.cum_infant_deaths':   'Infant deaths',
                        'fp.imr':                 'Infant mortality rate',
                        }
                elif to_plot == 'apo':  #adverse pregnancy outcomes
                    to_plot = {
                        'fp.cum_pregnancies':     'Pregnancies',
                        'fp.cum_stillbirths':     'Stillbirths',
                        'fp.cum_miscarriages':    'Miscarriages',
                        'fp.cum_abortions':       'Abortions',
                        }
                elif to_plot == 'method':
                    to_plot = {
                        'method_mix':                 'Method mix'
                    }
                elif to_plot is not None:
                    errormsg = f"Your to_plot value: {to_plot} is not a valid option"
                    raise ValueError(errormsg)
            else:
                errmsg = f"to_plot can be a dictionary or a string. A {type(to_plot)} is not a valid option."
                raise TypeError(errmsg)

            rows, cols = sc.getrowscols(len(to_plot), nrows=nrows, ncols=ncols)
            if to_plot == 'cpr':
                rows, cols = 1, 3
            for p, key, reslabel in sc.odict(to_plot).enumitems():
                ax = pl.subplot(rows, cols, p + 1)
                if '.' in key:
                    mod, rkey = key.split('.')
                    this_res = res[mod][rkey]
                else:
                    this_res = res[key]
                is_dist = hasattr(this_res, 'best')
                if is_dist:
                    y, low, high = this_res.best, this_res.low, this_res.high
                else:
                    y, low, high = this_res, None, None

                # Figure out x axis
                x = res.timevec

                percent_keys = ['mcpr', 'cpr', 'acpr', 'method_mix',
                                'proportion_short_interval']
                if key in percent_keys and key != 'method_mix':
                    y = y * 100 # why doesn't *= syntax work here? Is it overloaded on Result objects?
                    if is_dist:
                        low *= 100
                        high *= 100

                # Handle label
                if label is not None:
                    plotlabel = label
                else:
                    if new_fig:  # It's a new figure, use the result label
                        plotlabel = reslabel
                    else:  # Replace with sim label to avoid duplicate labels
                        plotlabel = self.label

                # Actually plot
                if key == "method_mix":
                    errormsg = 'Plotting method mix currently not working'
                    raise ValueError(errormsg)
                    # data = self.format_method_df(timeseries=True)
                    # method_names = data['Method'].unique()
                    # flipped_data = {method: [percentage for percentage in data[data['Method'] == method]['Percentage']]
                    #                 for method in method_names}
                    # colors = [colors[method] for method in method_names] if isinstance(colors, dict) else colors
                    # ax.stackplot(data["Year"].unique(), list(flipped_data.values()), labels=method_names, colors=colors)
                else:
                    ax.plot(x, y, label=plotlabel, **plot_args)

                if is_dist:
                    if 'c' in plot_args:
                        fill_args['facecolor'] = plot_args['c']
                    ax.fill_between(x, low, high, **fill_args)

                # Plot interventions, if present
                if hasattr(self, 'interventions'):
                    for intv in sc.tolist(self['interventions']):
                        if hasattr(intv, 'plot_intervention'): # Don't plot e.g. functions
                            intv.plot_intervention(self, ax)

                # Handle annotations
                fixaxis(useSI=fpd.useSI, set_lim=new_fig)  # If it's not a new fig, don't set the lim
                if key in percent_keys:
                    pl.ylabel('Percentage')
                elif 'mmr' in key:
                    pl.ylabel('Deaths per 100,000 live births')
                elif 'imr' in key:
                    pl.ylabel('Deaths per 1,000 live births')
                elif 'tfr' in key:
                    pl.ylabel('Fertility rate per 1,000 women')
                elif 'stillbirths' in key:
                    pl.ylabel('Number of stillbirths')
                else:
                    pl.ylabel('Count')
                pl.xlabel('Year')
                pl.title(reslabel, fontweight='bold')
                if xlims is not None:
                    pl.xlim(xlims)
                if ylims is not None:
                    pl.ylim(ylims)
                if (key == "method_usage"):  # need to overwrite legend for some plots
                    ax.legend(loc='upper left', frameon=True)
                if 'cpr' in to_plot:
                    if is_dist:
                        top = int(np.ceil(max(self.results['acpr'].high) / 10.0)) * 10  # rounding up to nearest 10
                    else:
                        top = int(np.ceil(max(self.results['acpr']) * 10.0)) * 10
                    self.conform_y_axes(figure=fig, top=top)
        return tidy_up(fig=fig, do_show=do_show, do_save=do_save, filename=filename)

    def plot_age_first_birth(self, do_show=None, do_save=None, fig_args=None, filename="first_birth_age.png"):
        """
        Plot age at first birth

        Args:
            fig_args (dict): arguments to pass to ``pl.figure()``
            do_show (bool): whether the user wants to show the output plot (default: true)
            do_save (bool): whether the user wants to save the plot to filepath (default: false)
            filename (str): the name of the path to output the plot
        """
        birth_age = self.people.first_birth_age
        data = birth_age[birth_age > 0]
        fig = pl.figure(**sc.mergedicts(dict(figsize=(7, 5)), fig_args))
        pl.title("Age at first birth")
        sns.boxplot(x=data, orient='v', notch=True)
        pl.xlabel('Age (years')
        return tidy_up(fig=fig, do_show=do_show, do_save=do_save, filename=filename)

    def format_method_df(self, method_list=None, timeseries=False):
        """
        Outputs a dataframe for method mix plotting for either a single year or a timeseries

        Args:
            method_list (list):
                list of proportions where each index is equal to the integer value of the corresponding method
            timeseries (boolean):
                if true, provides a dataframe with data from every year, otherwise a method_list is required for the year

        Returns:
            pandas.DataFrame with columns ["Percentage", "Method", "Sim", "Seed"] and optionally "Year" if timeseries
        """
        inv_method_map = {index: name for name, index in self.pars.fp['methods']['map'].items()}

        def get_df_from_result(method_list):
            df_dict = {"Percentage": [], "Method": [], "Sim": [], "Seed": []}
            for method_index, prop in enumerate(method_list):
                if method_index != fpd.method_map['None']:
                    df_dict["Percentage"].append(100 * prop)
                    df_dict['Method'].append(inv_method_map[method_index])
                    df_dict['Sim'].append(self.label)
                    df_dict['Seed'].append(self.pars['rand_seed'])

            return pd.DataFrame(df_dict)

        if not timeseries:
            return get_df_from_result(method_list)

        else:
            initial_year = self.pars['start_year']
            total_df = pd.DataFrame()
            for year_offset, method_list in enumerate(self.results['method_usage']):
                year_df = self.format_method_df(method_list)
                year_df['Year'] = [initial_year + year_offset] * len(year_df)
                total_df = pd.concat([total_df, year_df], ignore_index=True)
            return total_df

    def list_available_results(self):
        """Pretty print availbale results keys, sorted alphabetically"""
        output = 'Result keys:\n'
        keylen = 35  # Maximum key length  -- "interactive"
        all_keys = self.results.flatten(sep='.').keys()
        for k in sorted(all_keys):
            keystr = sc.colorize(f'  {k:<{keylen}s} ', fg='blue', output=True)
            reprstr = sc.indent(n=0, text=keystr, width=None)
            output += f'{reprstr}'
        print(output)
