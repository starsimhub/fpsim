'''
Optimize for birth spacing preference parameters
These parameters correct up or down for postpartum sexual activity
Uses calibration class and sets up a special SpacingCalib class
'''

import sciris as sc
import numpy as np
import pylab as pl
import fpsim as fp
import pandas as pd

# Define best, low, high limits for the parameters
calib_pars = dict(
    space0_6=[0.5, 0.2, 2.0],
    space9_15=[0.5, 0.2, 2.0],
    space18_24=[0.5, 0.2, 2.0],
    space27_36=[0.5, 0.2, 2.0],
)

location = 'senegal'
spacing_file = (f'../../fpsim/locations/{location}/data/birth_spacing_dhs.csv')  # Kenya DHS birth spacing data

do_save_figs = 1

spacing_bins = sc.odict(
    {'0-12': 0, '12-24': 1, '24-48': 2, '>48': 4})  # Spacing bins in years; CK: need to check that this is right
min_age = 15
max_age = 50
mpy = 12

# Load birth spacing data
data = pd.read_csv(spacing_file)


class SpacingCalib(fp.Calibration):
    ''' Custom version of calibration, just for birth spacing '''

    def __init__(self, calib_pars, verbose=True, keep_db=False, **kwargs):

        # Settings
        self.calib_pars = calib_pars
        self.verbose = verbose
        self.keep_db = keep_db

        # Configure Optuna
        self.set_optuna_defaults()
        self.configure_optuna(**kwargs)
        return

    def validate_pars(self):
        ''' Redefine this to skip it '''
        pass

    def make_sim(self, spacing_pars):
        base_pars = dict(location=location, n_agents=1000, verbose=0)
        sim = fp.Sim(base_pars)
        sim.pars['spacing_pref']['preference'][:3] = spacing_pars['space0_6']
        sim.pars['spacing_pref']['preference'][3:6] = spacing_pars['space9_15']
        sim.pars['spacing_pref']['preference'][6:9] = spacing_pars['space18_24']
        sim.pars['spacing_pref']['preference'][9:] = spacing_pars['space27_36']
        return sim

    def get_data_spaces(self, data):
        # Extract birth spacing data from  data
        data_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)

        # Extract birth spaces from data
        for i, j in data.iterrows():
            space = j['space_mo'] / mpy
            ind = sc.findinds(space > spacing_bins[:])[-1]
            data_spacing_counts[ind] += j['Freq']

        data_spacing_counts[:] /= data_spacing_counts[:].sum()
        spaces = np.array(data_spacing_counts.values())
        return spaces

    def get_spaces(self, sim):
        model_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
        ppl = sim.people
        for i in range(len(ppl)):
                if ppl.alive[i] and not ppl.sex[i] and min_age <= ppl.age[i] < max_age:
                        if ppl.parity[i] > 1:
                                cleaned_birth_ages = ppl.birth_ages[i][~np.isnan(ppl.birth_ages[i])]
                                for d in range(len(cleaned_birth_ages) - 1):
                                        space = cleaned_birth_ages[d + 1] - cleaned_birth_ages[d]
                                        if space > 0:
                                                ind = sc.findinds(space > spacing_bins[:])[-1]
                                                model_spacing_counts[ind] += 1

        model_spacing_counts[:] /= model_spacing_counts[:].sum()
        spaces = np.array(model_spacing_counts.values())
        return spaces

    def compute_mismatch(self, data_spaces, sim_spaces):
        diffs = np.array(data_spaces) - np.array(sim_spaces)
        mismatch = sum(diffs ** 2)
        return mismatch

    def run_exp(self, pars, label=None, return_exp=False):
        sim = self.make_sim(pars)
        sim.label = label
        sim.run()
        sim.sim_spaces = self.get_spaces(sim)
        sim.data_spaces = self.get_data_spaces(data)
        sim.mismatch = self.compute_mismatch(sim.data_spaces, sim.sim_spaces)
        if return_exp:
            return sim
        else:
            return sim.mismatch

    def run_trial(self, trial):
        spacing_pars = dict()
        for key, (vbest, vmin, vmax) in self.calib_pars.items():
            spacing_pars[key] = trial.suggest_float(key, vmin, vmax)
        mismatch = self.run_exp(spacing_pars)
        return mismatch

    def plot_spacing(self):
        pl.figure(figsize=(12, 6))

        for i, sim in enumerate([self.before, self.after]):
            data = sim.data_spaces
            model = sim.sim_spaces
            pl.subplot(1, 2, i + 1)
            nbars = len(data)
            y = np.arange(nbars)
            dy = 0.2
            kw = dict(height=0.2)
            pl.barh(y + dy, data, label='Data', facecolor='k', **kw)
            pl.barh(y, model, label='Model', facecolor='cornflowerblue', **kw)
            pl.barh(y - dy, data - model, label='Diff', facecolor='red', **kw)
            pl.yticks(y, spacing_bins.keys())
            pl.title(f'{sim.label}\n(mismatch: {sim.mismatch * 1000:n})')
            pl.legend()

        if do_save_figs:
            pl.savefig("birth_space_optimized.png", bbox_inches='tight', dpi=100)

        return


if __name__ == '__main__':
    trials = 5

    with sc.timer():
        calib = SpacingCalib(calib_pars, total_trials=trials)
        calib.calibrate()
        calib.plot_spacing()
        fig = calib.plot_trend()
        pl.savefig("calib_trend.png", bbox_inches='tight', dpi=100)

        sc.pp(calib.to_json()[:10])
