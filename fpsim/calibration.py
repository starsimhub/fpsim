'''
Define the Calibration class
'''


import os
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
import seaborn as sns
import starsim as ss
import optuna as op
from . import parameters as fpp
from . import experiment as fpe
from . import locations as fplocs


__all__ = ['Calibration']

# Exposure age knots: ages at which we set the exposure multiplier
EXPOSURE_KNOT_AGES = [0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50]

# Default bounds for exposure_age knots
EXPOSURE_AGE_BOUNDS = {
    0: (1.0, 1.0), 5: (0.01, 0.5), 10: (0.01, 0.5), 12.5: (0.05, 1.0),
    15: (0.3, 2.0), 18: (0.3, 2.5), 20: (0.5, 3.0), 25: (0.5, 3.0),
    30: (0.5, 3.0), 35: (0.2, 2.0), 40: (0.1, 1.5), 45: (0.05, 1.0), 50: (0.01, 0.5),
}


class Calibration(sc.prettyobj):
    '''
    A class to handle calibration of FPsim objects. Uses the Optuna hyperparameter
    optimization library (optuna.org).

    Supports calibration of scalar parameters (e.g. exposure_factor, prob_use_intercept,
    prob_use_trend_par, fecundity bounds) as well as the exposure_age curve — a 13-knot
    age-specific exposure multiplier fitted with a smoothness penalty to prevent sharp
    jumps between adjacent knots.

    During calibration, the location's make_calib_pars() is temporarily neutralized so
    that the optimizer's trial values take effect instead of being overridden by the
    location's existing calibrated parameters.

    Note: running a calibration does not guarantee a good fit! You must ensure that
    you run for a sufficient number of iterations, have enough free parameters, and
    that the parameters have wide enough bounds. Please see the tutorial on calibration
    for more information.

    Args:
        pars                (dict)  : simulation parameters, should include 'location' and 'n_agents'
        calib_pars          (dict)  : scalar parameters to calibrate, format dict(key=[best, low, high])
        weights             (dict)  : a custom dictionary of weights for each calibration target
        fit_exposure_age    (bool)  : whether to calibrate the exposure_age curve (default: True)
        smoothness_weight   (float) : weight of the smoothness penalty for exposure_age (default: 0.5)
        exposure_age_bounds (dict)  : custom bounds per knot age, format {age: (low, high)} (default: EXPOSURE_AGE_BOUNDS)
        n_trials            (int)   : the number of trials per worker (default: 100)
        n_workers           (int)   : the number of parallel workers (default: maximum CPUs)
        total_trials        (int)   : if n_trials is not supplied, calculate by dividing this by n_workers
        name                (str)   : the name of the database (default: 'fpsim_calibration')
        db_name             (str)   : the name of the database file (default: 'fpsim_calibration.db')
        keep_db             (bool)  : whether to keep the database after calibration (default: false)
        storage             (str)   : the location of the database (default: sqlite)
        verbose             (bool)  : whether to print details of the calibration
        kwargs              (dict)  : additional Optuna configuration

    Returns:
        A Calibration object

    **Examples**::

        # Calibrate scalar params + exposure_age curve
        calib_pars = dict(
            exposure_factor    = [1.0, 0.5, 3.0],
            prob_use_intercept = [-1.0, -3.0, 0.0],
            prob_use_trend_par = [0.0, -0.15, 0.05],
            fecundity_low      = [0.7, 0.5, 0.9],
            fecundity_high     = [1.5, 1.0, 2.5],
        )
        weights = dict(mcpr=3, asfr=3, total_fertility_rate=3, age_first_stats=4)
        calib = fp.Calibration(
            pars=dict(location='senegal', n_agents=5000),
            calib_pars=calib_pars,
            weights=weights,
        )
        calib.calibrate()

        # Scalar-only calibration (no exposure_age curve)
        calib = fp.Calibration(pars=pars, calib_pars=calib_pars,
                               weights=weights, fit_exposure_age=False)
        calib.calibrate()
    '''

    def __init__(self, pars, calib_pars=None, weights=None, fit_exposure_age=True,
                 smoothness_weight=0.5, exposure_age_bounds=None,
                 verbose=True, keep_db=False, **kwargs):
        self.pars       = pars
        self.calib_pars = calib_pars
        self.weights    = weights
        self.fit_exposure_age    = fit_exposure_age
        self.smoothness_weight   = smoothness_weight
        self.exposure_age_bounds = exposure_age_bounds if exposure_age_bounds is not None else EXPOSURE_AGE_BOUNDS
        self.verbose    = verbose
        self.keep_db    = keep_db
        self._original_calib = None

        # Configure Optuna
        self.set_optuna_defaults()
        self.configure_optuna(**kwargs)
        return

    def set_optuna_defaults(self):
        ''' Create a (mutable) dictionary with default global settings '''
        ''' Set defaults for Optuna '''
        g = sc.objdict()
        g.name      = 'fpsim_calibration'
        g.db_name   = f'{g.name}.db'
        g.storage   = f'sqlite:///{g.db_name}'
        g.n_trials  = 20  # Define the number of trials, i.e. sim runs, per worker
        g.n_workers = sc.cpu_count()
        self.g = g
        return

    def configure_optuna(self, **kwargs):
        ''' Update Optuna configuration, if required '''

        total_trials = kwargs.pop('total_trials', None)
        for key in self.g.keys():
            self.g[key] = kwargs.pop(key, self.g[key])

        if total_trials is not None:
            self.g.n_trials = int(np.ceil(total_trials/self.g.n_workers))

        if len(kwargs):
            errormsg = f'Did not recognize key(s) "{sc.strjoin(kwargs.keys())}", valid arguments are: {sc.strjoin(self.g.keys())}'
            raise ValueError(errormsg)

        return

    def validate_pars(self):
        '''
        Ensure parameters are in the correct format. Two formats are permitted:
        either a dict of arrays or lists in order best-low-high, e.g.::

            calib_pars = dict(
                exposure_factor           = [1.0, 0.5,  1.5],
                maternal_mortality_factor = [1,   0.75, 3.0],
            )

        Or the same thing, as a dict of dicts::

            calib_pars = dict(
                exposure_factor           = dict(best=1.0, low=0.5,  high=1.5),
                maternal_mortality_factor = dict(best=1,   low=0.75, high=3.0),
            )
        '''

        # First, we check that it's a dict
        if not isinstance(self.calib_pars, dict):
            errormsg = f'Calibration parameters must be supplied as a dict, not {type(self.calib_pars)}'
            raise TypeError(errormsg)

        # It's a dict! Now we iterate
        for key,val in self.calib_pars.items():

            # Check that the key is a valid parameter
            all_par_keys = fpp.all_pars().keys()
            # if key not in par_keys:
            if key not in all_par_keys:
                errormsg = f'Key "{key}" is not present the available parameter keys: {sc.newlinejoin(all_par_keys)}'
                raise sc.KeyNotFoundError(errormsg)

            # If each entry of calib_pars is a dict, convert to array
            if isinstance(val, dict):
                val_keys = set(val.keys())
                expected = set(['best', 'high', 'low'])
                if val_keys != expected: # Check that keys match
                    errormsg = f'If supply parameter limits as a dict, keys must be {expected}, not {val_keys}'
                    raise sc.KeyNotFoundError(errormsg)
                else: # If they do, convert to list/array form
                    self.calib_pars[key] = [val['best'], val['low'], val['high']]

            # Check that values are numeric
            try:
                self.calib_pars[key] = np.array(self.calib_pars[key])
            except Exception as E:
                errormsg = f'Calibration parameters must be supplied as an array of values [best, low, high], not "{self.calib_pars[key]}"'
                raise ValueError(errormsg) from E

            # Check that the values are in the right order
            try:
                v = self.calib_pars[key]
                assert v[1] <= v[0] <= v[2]
            except AssertionError as E:
                errormsg = f'Values must be in monotonic increasing order [best, low, high], but you have: {v}'
                raise ValueError(errormsg) from E
        return

    def _neutralize_calib_pars(self):
        ''' Temporarily replace the location's make_calib_pars() so trial values take effect '''
        location = self.pars.get('location')
        if location is None:
            return
        loc_module = getattr(fplocs, location, None)
        if loc_module is None or not hasattr(loc_module, 'make_calib_pars'):
            return
        self._original_calib = loc_module.make_calib_pars
        def _empty_calib_pars():
            return {'exposure_parity': np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,20],
                                                  [1,1,1,1,1,1,1,0.8,0.5,0.3,0.15,0.10,0.05,0.01]])}
        loc_module.make_calib_pars = _empty_calib_pars

    def _restore_calib_pars(self):
        ''' Restore the original make_calib_pars function '''
        location = self.pars.get('location')
        if location and self._original_calib is not None:
            loc_module = getattr(fplocs, location, None)
            if loc_module is not None:
                loc_module.make_calib_pars = self._original_calib
            self._original_calib = None

    def run_exp(self, calib_pars, return_exp=False, **kwargs):
        """ Create and run an experiment """
        pars = sc.mergedicts(sc.dcp(self.pars), calib_pars)
        exp = fpe.Experiment(pars=pars, **kwargs)
        exp.run(weights=self.weights)
        if return_exp:
            return exp
        else:
            return exp.fit.mismatch

    def run_trial(self, trial):
        ''' Define the objective for Optuna. Suggests scalar parameters and optionally
            exposure_age knots, runs the experiment, and returns mismatch + smoothness penalty. '''
        pars = {}
        for key, (best,low,high) in self.calib_pars.items():
            pars[key] = trial.suggest_float(key, low, high)

        # Optionally fit the exposure_age curve
        smoothness_penalty = 0.0
        if self.fit_exposure_age:
            exp_age_vals = []
            for age in EXPOSURE_KNOT_AGES:
                low, high = self.exposure_age_bounds[age]
                if low == high:
                    exp_age_vals.append(low)
                else:
                    exp_age_vals.append(trial.suggest_float(f'exp_age_{age}', low, high))
            pars['exposure_age'] = np.array([EXPOSURE_KNOT_AGES, exp_age_vals])
            diffs = np.diff(exp_age_vals)
            smoothness_penalty = self.smoothness_weight * np.mean(diffs**2)

        try:
            mismatch = self.run_exp(pars)
            return mismatch + smoothness_penalty
        except Exception as e:
            if self.verbose:
                print(f'  Trial failed: {e}')
            return float('inf')


    def worker(self):
        ''' Run a single worker '''
        study = op.load_study(storage=self.g.storage, study_name=self.g.name)
        output = study.optimize(self.run_trial, n_trials=self.g.n_trials)
        return output


    def run_workers(self):
        ''' Run multiple workers in parallel '''
        output = sc.parallelize(self.worker, self.g.n_workers)
        return output

    def remove_db(self):
        '''
        Remove the database file if keep_db is false and the path exists.
        '''
        sc.rmpath(self.g.db_name, die=False)
        if self.verbose:
            print(f'Removed existing calibration {self.g.db_name}')
        return

    def make_study(self):
        ''' Make a study, deleting one if it already exists '''
        if not self.keep_db:
            self.remove_db()
        output = op.create_study(storage=self.g.storage, study_name=self.g.name)
        return output

    def calibrate(self, calib_pars=None, weights=None, verbose=None, **kwargs):
        ''' Actually perform calibration. Neutralizes the location's make_calib_pars(),
            runs Optuna optimization, reconstructs the exposure_age array if fitted,
            and restores the original make_calib_pars() when done. '''

        # Load and validate calibration parameters
        if calib_pars is not None: self.calib_pars = calib_pars
        if weights    is not None: self.weights    = weights
        if verbose    is not None: self.verbose    = verbose
        if self.calib_pars is None:
            self.calib_pars = {}
        self.validate_pars()
        self.configure_optuna(**kwargs) # Update optuna settings

        # Neutralize location's make_calib_pars so trial values take effect
        self._neutralize_calib_pars()

        # Run the optimization
        if self.verbose:
            sc.heading('Starting calibration')
            print('Settings:')
            print(self.g)
        t0 = sc.tic()
        self.make_study()
        self.run_workers()
        self.study = op.load_study(storage=self.g.storage, study_name=self.g.name)
        self.best_pars = sc.dcp(self.study.best_params)
        T = sc.toc(t0, output=True)

        # Reconstruct exposure_age from individual knot params
        if self.fit_exposure_age:
            exp_age_vals = []
            for age in EXPOSURE_KNOT_AGES:
                low, high = self.exposure_age_bounds[age]
                if low == high:
                    exp_age_vals.append(low)
                else:
                    exp_age_vals.append(self.best_pars.pop(f'exp_age_{age}'))
            self.best_exposure_age = np.array([EXPOSURE_KNOT_AGES, exp_age_vals])
            self.best_pars['exposure_age'] = self.best_exposure_age

        print(f'Best trial: mismatch={self.study.best_value:.2f}, time: {T}')

        # Process the results
        self.initial_pars = {k:v[0] for k,v in self.calib_pars.items()}
        self.par_bounds   = {k:np.array([v[1], v[2]]) for k,v in self.calib_pars.items()}
        self.before = self.run_exp(calib_pars=self.initial_pars, label='Before calibration', return_exp=True)
        self.after  = self.run_exp(calib_pars=self.best_pars,    label='After calibration',  return_exp=True)
        self.parse_study()

        # Restore location's original make_calib_pars
        self._restore_calib_pars()

        # Tidy up
        if not self.keep_db:
            self.remove_db()
        if verbose:
            self.summarize()

        return

    def summarize(self):
        try:
            before = self.before.fit.mismatch
            after = self.after.fit.mismatch
            print('Initial parameter values:')
            print(self.initial_pars)
            print('Best parameter values:')
            print(self.best_pars)
            print(f'Mismatch before calibration: {before:n}')
            print(f'Mismatch after calibration:  {after:n}')
            print(f'Percent improvement:         {(before-after)/before*100:0.1f}%')
            return before, after
        except Exception as E:
            errormsg = 'Could not get summary, have you run the calibration?'
            raise RuntimeError(errormsg) from E


    def parse_study(self):
        ''' Parse the study into a data frame '''
        best = self.study.best_params  # Use raw study params (not reconstructed)

        print('Making results structure...')
        results = []
        n_trials = len(self.study.trials)
        failed_trials = []
        for trial in self.study.trials:
            data = {'index':trial.number, 'mismatch': trial.value}
            for key,val in trial.params.items():
                data[key] = val
            if data['mismatch'] is None:
                failed_trials.append(data['index'])
            else:
                results.append(data)
        print(f'Processed {n_trials} trials; {len(failed_trials)} failed')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i,r in enumerate(results):
            for key in keys:
                if key not in r:
                    print(f'Warning! Key {key} is missing from trial {i}, replacing with default')
                    r[key] = best[key]
                data[key].append(r[key])
        self.data = data
        self.df = pd.DataFrame.from_dict(data)

        return


    def to_json(self, filename=None):
        ''' Convert the data to JSON '''
        order = np.argsort(self.df['mismatch'])
        json = []
        for o in order:
            row = self.df.iloc[o,:].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
            for key,val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        if filename:
            sc.savejson(filename, json, indent=2)
        else:
            return json


    def plot_trend(self, best_thresh=2):
        ''' Plot the trend in best mismatch over time '''
        mismatch = sc.dcp(self.df['mismatch'].values)
        best_mismatch = np.zeros(len(mismatch))
        for i in range(len(mismatch)):
            best_mismatch[i] = mismatch[:i+1].min()
        smoothed_mismatch = sc.smooth(mismatch)
        fig = pl.figure(figsize=(16,12), dpi=120)

        ax1 = pl.subplot(2,1,1)
        pl.plot(mismatch, alpha=0.2, label='Original')
        pl.plot(smoothed_mismatch, lw=3, label='Smoothed')
        pl.plot(best_mismatch, lw=3, label='Best')

        ax2 = pl.subplot(2,1,2)
        max_mismatch = mismatch.min()*best_thresh
        inds = sc.findinds(mismatch<=max_mismatch)
        pl.plot(best_mismatch, lw=3, label='Best')
        pl.scatter(inds, mismatch[inds], c=mismatch[inds], label='Usable indices')
        for ax in [ax1, ax2]:
            pl.sca(ax)
            pl.grid(True)
            pl.legend()
            sc.setylim()
            sc.setxlim()
            pl.xlabel('Trial number')
            pl.ylabel('Mismatch')
        return fig


    def plot_all(self):
        ''' Plot every point: warning, very slow! '''
        g = pairplotpars(self.data, color_column='mismatch', bounds=self.par_bounds)
        return g


    def plot_best(self, best_thresh=2):
        ''' Plot only the points with lowest mismatch '''
        max_mismatch = self.df['mismatch'].min()*best_thresh
        inds = sc.findinds(self.df['mismatch'].values <= max_mismatch)
        g = pairplotpars(self.data, inds=inds, color_column='mismatch', bounds=self.par_bounds)
        return g


    def plot_stride(self, npts=200):
        '''Plot a fixed number of points in order across the results '''
        inds = np.linspace(0, len(self.df)-1, npts).round()
        g = pairplotpars(self.data, inds=inds, color_column='mismatch', bounds=self.par_bounds)
        return g



def pairplotpars(data, inds=None, color_column=None, bounds=None, cmap='parula', bins=None, edgecolor='w', facecolor='#F8A493', figsize=(20,16)):
    ''' Plot scatterplots, histograms, and kernel densities '''

    data = sc.odict(sc.dcp(data))

    # Create the dataframe
    df = pd.DataFrame.from_dict(data)
    if inds is not None:
        df = df.iloc[inds,:].copy()

    # Choose the colors
    if color_column:
        colors = sc.vectocolor(df[color_column].values, cmap=cmap)
    else:
        colors = [facecolor for i in range(len(df))]
    df['color_column'] = [sc.rgb2hex(rgba[:-1]) for rgba in colors]

    # Make the plot
    grid = sns.PairGrid(df)
    grid = grid.map_lower(pl.scatter, **{'facecolors':df['color_column']})
    grid = grid.map_diag(pl.hist, bins=bins, edgecolor=edgecolor, facecolor=facecolor)
    grid = grid.map_upper(sns.kdeplot)
    grid.fig.set_size_inches(figsize)
    grid.fig.tight_layout()

    # Set bounds
    if bounds:
        for ax in grid.axes.flatten():
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if xlabel in bounds:
                ax.set_xlim(bounds[xlabel])
            if ylabel in bounds:
                ax.set_ylim(bounds[ylabel])

    return grid
