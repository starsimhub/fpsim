==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1


Version 3.4.1 (2025-09-22)
---------------------------

  This release adds several priority regions with preliminary calibrations to FPsim Version 3.3.0. These countries/regions are: Cote d'Ivore, Niger, Pakistan Sindh region, Nigeria with regions Kano, Kaduna, and Lagos.

* **DHS data processing scripts**
  * Changed DHS and PMA data processing R scripts to filter to region level is specified in config.R file inputs region_variable (either “v024” or for Nigeria specifically, “sstate”), region name (e.g. “kaduna”), region_code (integer, e.g. 110 for Nigeria Kaduna), and country_region folder name (e.g. nigeria_kaduna). Script path:  fpsim/fpsim/data_processing/DHS_PMA_scripts/*.R.
  * Created DHS/PMA config_files sub folder within fpsim/data_processing/DHS_PMA_scripts/ to store previously run config files for future reference. Config files by country/region and saved as config_{country_region}.R
  * Updated fpsim/fpsim/data_processing/DHS_PMA_scripts/breastfeeding_stats.R script to accept the “still breastfeeding” status as either the value code (95) or the string (“still breastfeeding”)


* **New  locations**
  * Added additional locations (countries or subregions) to fpsim/fpsim/locations/{country_region} folders to store DHS, PMA, UN, and World Bank data.
  * Created location specific fpsim/fpsim/locations/{country_region}/run_{country}.py files to store parameters of calibrated models (calibrated to FPsim v3.3.0), run and save the calibrated model, and generate and save figures of the calibration (fpsim/plotting.py script, plot_calib function) in a “calib_results” directory. Also generated a README.md file for each country{_region} describing any manual edits to data files after generation by the DHS_PMA_scripts.
  * Added the following countries and country/regions to defaults.py valid_country_locs = ['senegal', 'kenya', 'ethiopia', 'cotedivoire', 'niger', 'nigeria_kano', 'nigeria_kaduna', ‘nigeria_lagos’, 'pakistan_sindh']
  * Added country/region imports to the fpsim/locations/__init__.py


Version 3.4.0 (2025-09-11)
---------------------------
Refactors data storage within FPsim, so that new locations can reside in external directories. The existing syntax `sim = fp.Sim(location='kenya')` still works, but now it is additionally possible to add a new setting from any folder via:
```
import fpsim as fp
my_data = fp.DataLoader(data_path='path-to-my-data')
sim = fp.Sim(dataloader=my_data)
```
In addition to allowing extra flexibility for users creating new locations, this change also supported the ongoing modularization of FPsim into distinct modules. The `DataLoader` class processes all data in the folder provided and sorts it into the module that requires it: death rates for the `Deaths` module, contraceptive choice parameters for the `Contraception` module, etc.


Version 3.3.2 (2025-09-05)
---------------------------
Minor bugfix in plotting.plot_calib(). Fixes x axis units for MCPR plot and reenables ASFR plot.


Version 3.3.1 (2025-08-28)
---------------------------
Minor release to add missing changelog.


Version 3.3.0 (2025-08-27)
---------------------------
Updates the FPsim code to be compatible with the most recent version of Starsim (v3.0.2). The most recent Starsim release included many improvements to the logic of time, probability, and rates. FPsim now benefits from these improvements; in particular:
- fp.defaults.mpy has been removed
- All of the logic of FPsim is now agnostic to the timestep. This means that everything will still work if you want to use a timestep of 3 months instead of 1, for instance.
- There were previously many places in the code where random draws occured. These have all been replaced with Starsim distributions, so it is now possible to take advantage of Starsim's common random number generation.
- The `archive` folder has been removed.


Version 3.2.0 (2025-08-15)
---------------------------

* **FPmod Class**
  This release moves the majority of FP logic from the `People` class to a new `FPmod` class. This is intended to make it more straightforward in the future to combine FP analyses with other analyses. This has several downstream implications for how to use the model and access results:
  * Most FP results are now stored in `sim.results.fp`, e.g. `sim.results.fp.pregnancies`. This will lead to eventual consistency with combined Starsim models, e.g. `sim.results.hiv.infections`.
  * Similarly, results related to contraception are stored in `sim.results.contraception`, e.g. `sim.result.contraception.mcpr`.
  * Introduces specific distributions for each random event; these distributions are CRN-safe and will eventually move FPsim to a CRN-safe model.
  * `sim.fp_pars` is now `sim.pars.fp`, which will make it consistent with other parameters like `sim.pars.hiv`.
  * Moves the `location` parameter from the `FPPars` class to the `SimPars` class.


* **Result tidying**
  * All the `tfr_{by_age}` results have been moved to an age-specific result structure. This means, for instance, that you would access the fertility rate for 20-24yos via `sim.connectors.fp.asfr[4, :]` instead of `sim.results.tfr_20-24`. 
  * All the `{result}_over_year` results have been removed. Annualized results can be easily calculated after a sim has been run, using `sim.results.to_df(resample='year')`. 


* **Misc changes**
  * `longitudinal_history` analyzer moved from the fpsim repo to the kenya_empowerment repo
  * Added attributes to people for storing the timesteps at which significant events occur, e.g. `ti_live_birth`, to make it easier to create analyzers related to these events
  * Removes `total_births_{by_age}` and `total_women_{by_age}` - these were previously needed to calculate the total fertility rate, but this is no longer needed, and the addition of these results increases the size of the results object considerably. If these results are needed, they can be computed with an analyzer or people snapshot.
  * Renames `urban_women` to `n_urban` for consistency with other results, e.g. `n_wq1`, `n_fecund`. 
  * Removes the `extract_employment` method from the `Experiment` class, as this is no longer functional.
  * Removes the `MultiSim` class.
  * Numerous bugfixes; see PR for details.


Version 3.1.0 (2025-07-31)
---------------------------
- Refactors Contraception and Education to Starsim modules (connectors)
- Exposes all parameters for different modules in the `pars` dictionary passed to the Sim. This means that creating a sim is far more flexible, e.g. all of these will work:
```
import fpsim as fp
sim1 = fp.Sim(prob_use_trend_par=0.3, start=2000)
sim2 = fp.Sim(pars = dict(prob_use_trend_par=0.3, start=2000))
sim3 = fp.Sim(contra_pars = dict(prob_use_trend_par=0.3), sim_pars=dict(start=2000))
sim4 = fp.Sim(start=2000, contraception_module = fp.SimpleChoice(prob_use_trend_par=0.3))
```
- Refactors parameters.py, so that all FP parameters inherit from Starsim parameter classes. We now have a `SimPars` class, as well as `FPPars`, `ContraPars`, and `EduPars`. When a Sim is created, any parameters passed in by the user will automatically get assigned to the right modules. This also allowed for a notable simplifaction in the parameter validation logic, since Starsim handles that automatically.
- *GitHub info*: PR `575 <https://github.com/fpsim/fpsim/pull/575>`_


Version 3.0.4 (2025-07-30)
---------------------------
- Updates ethiopia/regions model files according to latest template used in country files
- Updates data_utils.py and plotting.py to be usable with both country and region model locations
- In conjunction with two changes above, modifies ethiopia region data filenames to be consistent with country data filenames
- Shifts valid_country_locs and valid_region_locs to top of defaults.py for easier modification
- Updates locations/README to document the steps to add region model(s) according to changes and expected format above
- Slight change to filenames() function format in template and model files to more robustly capture filepaths with correct base_dir
- Removes deprecated region logic in experiment.py and sim.py
- Tiny updates to sim and multisim examples
- *GitHub info*: PR `581 <https://github.com/fpsim/fpsim/pull/592>`_


Version 3.0.3 (2025-07-09)
---------------------------
Resolves issues:
#580 - removes requirement headers for Tutorials 2-5 (redundant because in header of Tutorial 1)
#578 - modifies sigmoid function in methods.py to use scipy's expit function to avoid runtime warnings (due to overflow with very large/small rhs values)
- *GitHub info*: PR `581 <https://github.com/fpsim/fpsim/pull/581>`_


Version 3.0.2 (2025-07-02)
---------------------------
- Fixes issue 567 (bug in method_mix_by_age_analyzer)
- Fixes issue 568 (reduces default verbosity of Sim)
- *GitHub info*: PR `574 <https://github.com/fpsim/fpsim/pull/574>`_


Version 3.0.1 (2025-07-01)
---------------------------
Adds a plotting class that can be used to create plots typically used for calibration and analyses. Both manual and automatic
calibration example scripts were cleaned and consolidated. Tutorial explaining overall calibration process created.

 *GitHub info*: PR `https://github.com/fpsim/fpsim/pull/547>`_


Version 3.0.0 (2025-06-26)
---------------------------

This version of FPsim is now built on `Starsim <https://starsim.org>`_. Several major changes were introduced as a result.

Overview of changes
~~~~~~~~~~~~~~~~~~~

* **Analyzer Refactor**

  * Replaced ``fpsim.analyzers.Analyzer`` with ``starsim.Analyzer`` as the base class.
  * Updated all analyzer classes in ``fpsim/analyzers.py`` to inherit from ``starsim.Analyzer``.
  * Renamed and refactored key analyzer methods:

    * Replaced ``apply`` with ``step``.
    * Refactored initialization/finalization methods to align with StarSim conventions (``init_results``, ``init_pre``, ``init_post``, etc.).
    * Updated logic to use ``.female`` instead of ``.is_female`` and ``.uids`` for indexing.

  * Replaced direct boolean/array filtering with StarSim's approach using ``.uids`` and attribute access.

* **Array Handling**

  * Introduced new module: ``fpsim/arrays.py`` with a ``TwoDimensionalArr`` array class for handling multi-valued attributes (e.g., birth ages).
  * Replaced previous handling of multi-valued person states (like ``birth_ages``, ``stillborn_ages``, etc.) with ``TwoDimensionalArr`` arrays in ``fpsim/defaults.py``.

* **Parameter System Overhaul**

  * Simplified and separated FPsim-specific and simulation-level parameters in ``fpsim/parameters.py``.

    * Added ``default_sim_pars`` for high-level simulation settings.
    * Refactored ``pars()`` function to create parameter sets and validate keys.
    * Removed old ``Pars`` class.
    * Updated parameter validation logic.

  * Adjusted parameter passing and initialization throughout the codebase for compatibility (notably in ``fpsim/experiment.py``, ``fpsim/scenarios.py``, etc.).

* **Method & Module API Changes**

  * All module methods that handle people (contraception, education, etc.) now require explicit ``uids`` for subsetting and updating.
  * Updated contraception and education modules to take and use ``uids`` for all relevant operations.
  * Refactored method selection, probability assignment, and duration logic in ``fpsim/methods.py`` for array compatibility and efficiency.

Detailed changes
~~~~~~~~~~~~~~~~

* **fpsim/analyzers.py**

  * Refactored all analyzers for new array and indexing conventions.
  * Updated key logic for CPR, method mix, education, age pyramids, and tracking analyzers.

* **fpsim/defaults.py**

  * Changed all person state definitions to use StarSim array classes.
  * Removed direct use of numpy arrays, replaced with typed StarSim arrays and new ``TwoDimensionalArr``.
  * Updated results arrays and lists for new structure.

* **fpsim/education.py**

  * Refactored all methods to use explicit ``uids`` for subsetting.
  * Improved efficiency and clarity of education progression, interruption, resumption, and graduation logic.

* **fpsim/experiment.py**

  * Updated experiment parameter passing to use new ``pars`` structure.
  * Adjusted data extraction and model comparison routines for compatibility.

* **fpsim/methods.py**

  * Refactored all module methods to use ``uids`` for indexing and updating.
  * Improved probabilistic selection, method choice, and duration assignment logic.

* **fpsim/parameters.py**

  * Removed class-based parameter system; now uses plain dictionaries and utility functions.
  * Introduced validation and JSON (de)serialization helpers.

* **fpsim/scenarios.py**

  * Refactored scenario parameter handling and simulation creation for new parameter API.

* **fpsim/utils.py**

  * Updated Numba-accelerated utility functions to use ``float32`` for compatibility.

* **Other changes**

  * Removed obsolete ``fpsim/base.py``.
  * Various bug fixes and code style improvements for array handling, type consistency, and API clarity.

---

**Note:**
This PR introduces several breaking changes to the FPsim API, notably in how arrays and parameters are handled, and how modules interact with people objects. Downstream code and scripts will likely require updates to align with the new conventions.

*GitHub info*: PR `490 <https://github.com/fpsim/fpsim/pull/490>`_


Version 2.0.2 (2025-05-28)
---------------------------
- Introduces a flexible location_registry system to support custom location modules defined outside the core fpsim package.
- External users can now create an analysis repo and register their own locations in their respective repo via fpsim.defaults.register_location(), eliminating the need for a user to modify the core code in order to test creating their own model in location other than Ethiopia, Senegal, or Kenya.
- *GitHub info*: PR `547 <https://github.com/fpsim/fpsim/pull/547>`_

Version 2.0.1 (2025-05-09)
---------------------------
- Cleanup of data processing scripts in the data_processing directory
- Creation of shared_data dir for data shared across model locations and corresponding data_utils cleanup
- Addition of <location>.py template for new users
- Documentation overhaul in locations/README, data_processing/README, and data processing scripts
- *GitHub info*: PR `531 <https://github.com/fpsim/fpsim/pull/531>`_

Version 2.0.0 (2025-03-31)
---------------------------
FPsim 2.0 represents a major overhaul of how the model represents contraceptive choices. Previously, women updated their contraceptive choices each year in their birth month.
This version introduces a new method for setting contraceptive choices, whereby women choose a method of contraception, and are then assigned a duration of use (`dur_use`). They update their contraceptive choice at the end of this duration, or after certain other events (e.g. postpartum, or after a change in one of their empowerment metrics).

To support this new feature, this PR also introduces several new items:
 - `ContraceptiveChoice` modules can be defined (eg, `SimpleChoice`) and flexibly added to sims depending on how one wishes to model the mechanism for choosing contraception

 There are also several other minor changes:
 - `sim.i` has been renamed `sim.ti` for consistency across starsim modules and for clarity that it refers to an integer representation of time (time step)
 - `track_switching` has been moved to an analyzer
 - the fpsim `ndict` class has been removed and replaced with the starsim `ndict` class
 - the lists of lists in the results object have been replaced with arrays
 - `Education` modules have been added
 - new parameters and people's attributes have been added: `fertility_intent`, `intent_to_use` and `wealth_quintile`
 - new analyzer "Life of" has been added to visualise the events during the lifecourse of a woman
 - new intervention that allows for changes in the probability of initiation (of contraceptive use) in contraception modules that use a logistic regression module.
 - Fixes a bug that prevented simulations from running if pars['timestep'] !=1
 - Adds a circular buffer to track the last 12 months of data, and enable model updates that depend on the previous state

 *GitHub info*: PR `411 <https://github.com/fpsim/fpsim/pull/411>`_

Version 1.0.4 (2024-08-19)
---------------------------
- Fixes issue 310 (Removes legacy use_subnational logic)
- *GitHub info*: PR `392 <https://github.com/fpsim/fpsim/pull/392>`_

Version 1.0.3 (2024-07-26)
---------------------------
- Adds .devcontainer configuration for a Codespaces dev container
- *GitHub info*: PR `369 <https://github.com/fpsim/fpsim/pull/369>`_

Version 1.0.2 (2024-07-25)
---------------------------
- Fixes issue 347, correcting variable in defaults storing age-specific results
- *GitHub info*: PR `388 <https://github.com/fpsim/fpsim/pull/388>`_

Version 1.0.1 (2024-06-17)
---------------------------
- Adds empowerment metrics (paid work and education attainment) to calibration targets
- Creates script for empowerment calibration
- *GitHub info*: PR `240 <https://github.com/fpsim/fpsim/pull/240>`_

Version 1.0.0 (2024-06-07)
--------------------------
- Releases FPsim1.0 to pypi
- Adds FPsim New User Instructions to tutorials
- *GitHub info*: PR `349 <https://github.com/fpsim/fpsim/pull/349>`_

Version 0.28.3 (2024-04-30)
--------------------------
- Creates subnational tutorial for Ethiopia
- Updates regional scripts to improve subnationals calibrations for Ethiopia
- *GitHub info*: PR `334 <https://github.com/fpsim/fpsim/pull/334>`_

Version 0.28.2 (2024-04-20)
---------------------------
- Refactoring of location directory, such that there is a country folder under locations/ dir containing the data, model script(s), and figs pertaining to that location only
- Modifies these new paths accordingly in all relevant files
- *GitHub info*: PR `316 <https://github.com/fpsim/fpsim/pull/316>`_

Version 0.28.1 (2024-04-11)
---------------------------
- Updates regional parameters to calibrate subnationally in Ethiopia
- Revises subnational calibration script to run for all regions and includes a new multisim script for subnational comparisons 
- *GitHub info*: PR `319 <https://github.com/fpsim/fpsim/pull/319>`_

Version 0.27.0 (2024-02-09)
---------------------------
- Builds out new regional attribute and parameters for subnational dynamics in Ethiopia
- Creates new subnational calibration script and subfolder for regional datasets and figures 
- *GitHub info*: PR `241 <https://github.com/fpsim/fpsim/pull/241>`_

Version 0.26.8 (2024-02-08)
---------------------------
- Removed unused ``timeseries_recorder`` and ``verbose_sim`` analyzers.
- *GitHub info*: PR `257 <https://github.com/fpsim/fpsim/pull/257>`_

Version 0.26.7 (2023-10-24)
---------------------------
- Created two scripts in data_processing dir to scrape UN and World Bank country calibration data
- Updated location_template.txt with logic to pull this scraped data for future contexts
- *GitHub info*: PR `213 <https://github.com/fpsim/fpsim/pull/213>`_

Version 0.26.6 (2023-9-27)
--------------------------
- Renamed the country data filenames to be standardized across contexts
- Updated references to these updated country data files
- *GitHub info*: PR `196 <https://github.com/fpsim/fpsim/pull/196>`_

Version 0.26.5 (2023-9-27)
--------------------------
- Adds Tutorial for manual calibration to docs/tutorials/ directory
- *GitHub info*: PR `166 <https://github.com/fpsim/fpsim/pull/166>`_

Version 0.26.4 (2023-9-27)
--------------------------
- Refactoring of Experiment class to run with the latest country data formats
- Country files have updated references to these updated country data files
- Senegal country data files updated to latest standard format (based on Kenya)
- *GitHub info*: PR `193 <https://github.com/fpsim/fpsim/pull/193>`_

Version 0.26.3 (2023-9-27)
--------------------------
- Updates Senegal files in locations/senegal to be in the same format as Kenya for standardization purposes
- Minor updates to calibrate_manual.py to be able to run with Senegal data (in addition to Kenya and other contexts)
- *GitHub info*: PR `165 <https://github.com/fpsim/fpsim/pull/165>`_

Version 0.26.2 (2023-7-25)
--------------------------
- Updates code for manual calibration
- Adds folder to create figures for manual calibration runs
- *GitHub info*: PR `164 <https://github.com/fpsim/fpsim/pull/164>`_

Version 0.26.1 (2023-6-29)
--------------------------
- Updates code for contraceptive matrices
- Adds contraceptive matrices for Ethiopia to run manual calibration
- *GitHub info*: PR `161 <https://github.com/fpsim/fpsim/pull/161>`_

Version 0.26.0 (2023-5-31)
--------------------------
- Builds out new parameters file for Ethiopia
- Adds camparison data for Ethiopia calibration
- *GitHub info*: PR `156 <https://github.com/fpsim/fpsim/pull/156>`_

Version 0.25.0 (2023-5-13)
--------------------------
- Adds to sim.py to track sexual inactivity in agents
- Restricts method use to only those women sexually active over the last 12 months and debuted
- *GitHub info*: PR `157 <https://github.com/fpsim/fpsim/pull/157>`_

Version 0.24.1 (2023-4-14)
--------------------------
- Fixes example_calib.py and example_exp.py to run with fpsim library
- Updates manual_calibration.py to be flexible to take any location with identically structured data as Kenya
- *GitHub info*: PR `133 <https://github.com/fpsim/fpsim/pull/133>`_

Version 0.24.0 (2023-3-17)
--------------------------
- Finish a manual calibration script for plotting model outcomes vs data for Kenya
- Goal is to make this script more flexible for other locations, still needs some tweaks for data import
- Compares ASFR, TFR, age/parity mix, contraceptive use and mix, CPR, pop growth rate, age at first birth, and birth spacing bins
- *GitHub info*: PR `133 <https://github.com/fpsim/fpsim/pull/133>`_

Version 0.23.2 (2023-3-6)
--------------------------
- Add duration of a short birth interval between live births (short_int) to senegal.py
- Track number of short interval births an agent has had over their life and number are happening at each time step in sim.py 
- Add age limit parameters (age_low, age_high) to senegal.py to track age-specific short interval births.
- Create a time series plot of all short births interval during a sim as well as the age-specific time series.
- *GitHub info*: PR `107 <https://github.com/fpsim/fpsim/pull/107>`_

Version 0.23.1 (2023-2-28)
--------------------------
- Add location folder for Ethiopia calibration
- Add comparison data to Ethiopia folder
- *GitHub info*: PR `118 <https://github.com/fpsim/fpsim/pull/118>`_

Version 0.23.0 (2023-2-10)
--------------------------
- Add optimize-space-prefs.py using Calibration class to algorithmically find best birth space params
- *GitHub info*: PR `119 <https://github.com/fpsim/fpsim/pull/119>`_

Version 0.22.1 (2023-2-03)
--------------------------
- Update contraceptive matrices to be weighted
- *GitHub info*: PR `113 <https://github.com/fpsim/fpsim/pull/113>`_


Version 0.22.0 (2023-1-24)
--------------------------
- Add calibrate_manual.py to compare sim runs to data with new data structures
- Add plot_birth_spacing.py under senegal location to fine tune this calibration
- *GitHub info*: PR `109 <https://github.com/fpsim/fpsim/pull/109>`_

Version 0.21.2 (2022-12-16)
---------------------------
- Updates Kenya, 2nd pass, completed 1st draft
- Starts calibrate_manual.py for Kenya with ASFR plot
- *GitHub info*: PR `76 <https://github.com/fpsim/fpsim/pull/76>`_

Version 0.21.1 (2022-12-09)
---------------------------
- Updates calibrated data to compare for Kenya, 1st pass
- Adds raw data to kenya folder
- *GitHub info*: PR `70 <https://github.com/fpsim/fpsim/pull/70>`_

Version 0.21.0 (2022-12-06)
---------------------------
- Updates contraceptive matrices in kenya.py to be from Kenya PMA 2019-2020
- Adds raw data to kenya folder and processing code to data_processing folder
- *GitHub info*: PR `51 <https://github.com/fpsim/fpsim/pull/51>`_


Version 0.20.0 (2022-11-30)
---------------------------
- Builds out new parameters file for Kenya
- Adds and reorganizes directories for external data files and data processing scripts
- *GitHub info*: PR `37 <https://github.com/fpsim/fpsim/pull/37>`_


Version 0.19.2 (2022-10-28)
---------------------------
- Added user guide
- *GitHub info*: PR `4 <https://github.com/fpsim/fpsim/pull/4>`_


Version 0.19.1 (2022-10-26)
---------------------------
- Moved to new repository location (http://github.com/fpsim/fpsim)
- Updated documentation in README
- Created new tutorials in tutorials folder
- Ordered tutorials by complexity through T1, T2, T3... Tn numbering system
- *GitHub info*: PR `1 <https://github.com/fpsim/fpsim/pull/1>`_


Version 0.19.0 (2022-09-01)
---------------------------
- Added age-specific plotting for tfr, pregnancies, imr, mmr, stillbirths, and births to Sim, MultiSim, and Scenarios
- Added ability to plot channels by age over the course of an interval of time (one year, for example)
- Added yearly age-specific plotting for pregnancies, imr and mmr
- *GitHub info*: PR `590 <https://github.com/amath-idm/fpsim/pull/590>`_


Version 0.18.2 (2022-08-12)
---------------------------
- Added age specific plotting for cpr, mcpr, and acpr to Sim, MultiSim, and Scenarios
- *GitHub info*: PR `584 <https://github.com/amath-idm/fpsim/pull/584>`_


Version 0.18.1 (2022-08-08)
---------------------------
- Added y-axis scaling to Sim.plot and MultiSim.plot()
- *GitHub info*: PR `583 <https://github.com/amath-idm/fpsim/pull/583>`_


Version 0.18.0 (2022-08-01)
---------------------------
- Adjusted stillbirth rates from Nori et al., which was conducted June 2022
- *GitHub info*: PR `560 <https://github.com/amath-idm/fpsim/pull/560>`_


Version 0.17.5 (2022-07-28)
---------------------------
- Refactored ExperimentVerbose and verbose_sim and related parts of test suite
- *GitHub info*: PR `471 <https://github.com/amath-idm/fpsim/pull/471>`_


Version 0.17.4 (2022-07-27)
---------------------------
- Added new test suite for the Scenarios API
- *GitHub info*: PR `527 <https://github.com/amath-idm/fpsim/pull/527>`_


Version 0.17.3 (2022-07-18)
---------------------------
- Added tutorial jupyter notebook to showcase Scenarios features
- *GitHub info*: PR `484 <https://github.com/amath-idm/fpsim/pull/484>`_


Version 0.17.2 (2022-07-15)
---------------------------
- Switched method mix plotting from line chart to stacked area chart for all classes
- *GitHub info*: PR `568 <https://github.com/amath-idm/fpsim/pull/568>`_


Version 0.17.1 (2022-07-14)
---------------------------
- Added example_scens.py for a quick debug of adding a novel method when developing new features
- Updated README with new debugging guidance
- GitHub info*: PR `570 <https://github.com/amath-idm/fpsim/pull/570>`_


Version 0.17.0 (2022-07-08)
---------------------------
- Added method mix timeseries plotting to Sim, MultiSim, and Scenarios through plot(to_plot='method')
- Added some test coverage for method mix plotting
- *GitHub info*: PR `554 <https://github.com/amath-idm/fpsim/pull/554>`_


Version 0.16.2 (2022-07-01)
---------------------------
- Refactors channel aggregation in Scenarios.analyze_sims()
- *GitHub info*: PR `561 <https://github.com/amath-idm/fpsim/pull/561>`_


Version 0.16.1 (2022-06-30)
---------------------------
- Add tracking of pregnancies
- Add cumulative sum of pregnancies to plotting functionality (see plot('apo'))
- *GitHub info*: PR `555 <https://github.com/amath-idm/fpsim/pull/555>`_


Version 0.16.0 (2022-06-28)
---------------------------
- Split matrix age category >25 into 26-35 and >35 
- Baseline contraceptive behavior remains the same, but interventions can differentiate now
- *GitHub info*: PR `551 <https://github.com/amath-idm/fpsim/pull/551>`_


Version 0.15.0 (2022-06-13)
---------------------------
- Added new plotting functionality ``Scenarios.plot('mortality')``
- Added new plotting functionality ``Scenarios.plot('apo')`` for adverse pregnancy outcomes
- Added ``stillbirths_over_year`` to keys, tracking, and plotting
- Added tracking of miscarriage, abortion, corresponding keys and plotting
- Temporarily commented out plot_interventions in ``sim.py`` to fix x-axis and vline issues in plotting
- *GitHub info*: PR `549 <https://github.com/amath-idm/fpsim/pull/549>`_


Version 0.14.2 (2022-06-06)
---------------------------
- Adding 3 new columns to the results dataframe in Scenarios


Version 0.14.2 (2022-05-31)
---------------------------
- Fixed bug in ``fp.snapshot()`` missing non-exact timesteps.
- Fixed bug with ``fp.timeseries_recorder()`` not being capable of being added as a kwarg.
- Tidied output of ``SimVerbose.story()``.
- Added ``sim.get_analyzer()`` and ``sim.get_intervention()`` methods (along with the plural versions).
- Renamed ``Experiment.dhs_data`` to ``Experiment.data``; likewise for ``model_to_calib`` → ``model``.
- Fixed bug with MCPR year plotting in ``Experiment``.
- Fixed bug with analyzers being applied only at the end of the sim instead of at every timestep.
- Fixed bug with interventions not plotting with simulations.
- Fixed bug with ``finalize()`` not being called for interventions.
- Increased code coverage of tests from 67% to 80%.
- *GitHub info*: PR `533 <https://github.com/amath-idm/fp_analyses/pull/533>`_


Version 0.14.1 (2022-05-27)
---------------------------
- Fixed bugs in how ``copy_from`` is implemented in scenarios.
- *GitHub info*: PR `526 <https://github.com/amath-idm/fp_analyses/pull/526>`_


Version 0.14.0 (2022-05-26)
---------------------------
- Adds an options module, allowing things like DPI to be set via ``fp.options(dpi=150)``.
- Updates plotting options and allows more control over style.
- Adds more control to plots, including ``start_year`` and ``end_year``.
- Adds a ``copy_from`` keyword to method probability update scenarios.
- Renames ``years`` to ``par_years`` in scenarios.
- Changes the logic of the ``People`` update step so that lactational amenorrhea is calculated after breastfeeding is updated.
- Changes the ``Sim`` representation to e.g. ``Sim("My sim"; n=10,000; 1960-2020; results: b=69,541 ☠=11,920 pop=62,630)``
- *GitHub info*: PR `522 <https://github.com/amath-idm/fp_analyses/pull/522>`__


Version 0.13.2 (2022-05-25)
---------------------------
- Added ASFR as an output of Experiments.
- ``MultiSim.run()`` now automatically labels un-labeled sims; this fixes bugs in MultiSim plotting functions.
- MultiSims also have additional error checking (e.g., they cannot be rerun).
- Refactored data files to be in "tall" instead of "wide" format.
- Removed years and age bins from summary statistics.
- *GitHub info*: PR `517 <https://github.com/amath-idm/fp_analyses/pull/517>`__


Version 0.13.1 (2022-05-25)
---------------------------
- Changed ``MultiSim.plot_method_mix()`` to be able to work with ``Scenarios``
- *GitHub info*: PR `513 <https://github.com/amath-idm/fp_analyses/pull/513>`__


Version 0.13.0 (2022-05-23)
---------------------------
- Changed parameters from a dictionary to a class and added ``parameters.py``. This class has additional validation, the ability to import from/export to JSON, etc.
- Restructured methods, including renaming ``pars['method_efficacy']`` to ``pars['methods']['eff']``, plus a new entry, ``pars['methods']['modern']``, to specify which are modern methods used for calculating MCPR.
- Methods have been reordered, grouping traditional and modern methods and sorting modern methods by longevity (e.g. condoms → pill → implants → IUDs).
- Added ability to add/remove contraceptive methods via ``pars.add_method()`` and ``pars.rm_method()``.
- Added a method to run a single scenario.
- *GitHub info*: PR `503 <https://github.com/amath-idm/fp_analyses/pull/503>`__


Version 0.12.0 (2022-05-22)
---------------------------
- Split FPsim repository from analyses scripts.
- Refactors ``experiment.py`` to load files for a specific location rather than being hard-coded.
- *GitHub info*: PR `504 <https://github.com/amath-idm/fp_analyses/pull/504>`__


Version 0.11.5 (2022-05-21)
---------------------------
- Improvements to the scenarios, including more helpful docstrings and error messages.
- Improved error checking of sims.
- *GitHub info*: PR `502 <https://github.com/amath-idm/fp_analyses/pull/502>`__


Version 0.11.4 (2022-05-20)
---------------------------
- Renamed parameter ``n`` to ``n_agents``, and adds parameter ``scaled_pop``.
- Tracking of switch events is disabled by default; set ``pars['track_switching'] = True`` to re-enable.
- Update default end year from 2019 to 2020.
- *GitHub info*: PR `496 <https://github.com/amath-idm/fp_analyses/pull/496>`__


Version 0.11.3 (2022-05-20)
---------------------------
- Tidied ``tests`` folder.
- Removed the calibration database by default (to keep, use ``fp.Calibration(keep_db=True)``.
- *GitHub info*: PR `495 <https://github.com/amath-idm/fp_analyses/pull/495>`__


Version 0.11.2 (2022-05-20)
---------------------------
- Added a ``people.make_pregnant()`` method.
- *GitHub info*: PR `494 <https://github.com/amath-idm/fp_analyses/pull/494>`__


Version 0.11.1 (2022-05-20)
---------------------------
- Replaced ``high`` and ``low`` breastfeeding duration parameters with Gumbel distribution parameters ``mu`` and ``beta``.
- *GitHub info*: PR `493 <https://github.com/amath-idm/fp_analyses/pull/493>`__


Version 0.11.0 (2022-05-20)
---------------------------
- Major refactor of ``senegal.py``, organizing parameters into groups and renaming.
- Parameter names made more consistent, e.g. ``exposure_correction`` → ``exposure_factor``, ``maternal_mortality_multiplier`` → ``maternal_mortality_factor``.
- Added comprehensive parameter checking.
- Updates to the default representation: ``print(sim)`` is now a very brief representation; use ``sim.disp()`` to get the old behavior.
- *GitHub info*: PR `492 <https://github.com/amath-idm/fp_analyses/pull/492>`__


Version 0.10.7 (2022-05-19)
---------------------------
- Updated ``fp.Scenarios()`` API.
- Added a new ``fp.Scenario()`` class, with a convenience function ``fp.make_scen()`` for creating new scenarios, for later use with ``fp.Scenarios()``.
- *GitHub info*: PR `488 <https://github.com/amath-idm/fp_analyses/pull/488>`__


Version 0.10.6 (2022-05-19)
---------------------------
- Adds ``fp.parallel()`` to quickly run multiple sims in parallel and return a ``MultiSim`` object.
- Adds an ``fp.change_par()`` intervention.
- *GitHub info*: PR `487 <https://github.com/amath-idm/fp_analyses/pull/487>`__


Version 0.10.5 (2022-05-18)
---------------------------
- Changes how the matrices are implemented. For example, ``sim['methods']['probs']['18-25']`` has been renamed ``sim['methods']['raw']['annual']['18-25']``; ``sim['methods']['probs']['18-25']`` has been renamed ``sim['methods']['adjusted']['annual']['18-25']``; ``sim['methods_postpartum']['probs1to6']['18-25']`` has been renamed ``sim['methods']['adjusted']['pp1to6']['18-25']``; etc.
- Various other parameters were renamed for consistency (e.g. ``years`` → ``year``).
- Various other methods were renamed for clarity (e.g. ``maternal_mortality()`` → ``check_maternal_mortality()``; ``check_mcpr()`` → ``track_mcpr()``).
- Input validation has been added to the ``Scenarios`` class.
- Fixed ``fp.update_methods()`` so it can no longer produce probabilities >1.
- Removed a circular import in ``scenarios.py``.
- *GitHub info*: PR `482 <https://github.com/amath-idm/fp_analyses/pull/482>`__


Version 0.10.4 (2022-05-17)
---------------------------
- Fixes bugs with the MCPR growth implementation, as well as the wrong matrix being used.
- Added three new parameters: ``mcpr_growth_rate``, ``mcpr_max``, and ``mcpr_norm_year``, to control how MCPR growth is projected into the future.
- Updated ``sim.run()`` to return ``self`` rather than ``self.results``.
- *GitHub info*: PR `480 <https://github.com/amath-idm/fp_analyses/pull/480>`__


Version 0.10.3 (2022-05-12)
---------------------------
- Move country-specific parameters from ``fpsim.data`` to ``fpsim.locations``.
- *GitHub info*: PR `464 <https://github.com/amath-idm/fp_analyses/pull/464>`__


Version 0.10.2 (2022-05-10)
---------------------------
- Refactored ``People.get_method()`` to use more efficient looping.
- Numbafied ``n_multinomial()`` to get a ~20% speed increase.
- Added a ``method_timestep`` parameter to allow skipping contraceptive matrix updates (saves significant time for small sims).
- Added ``fp.pars(location='test')`` to use defaults for testing (e.g. small population size).
- Fixed divide-by-zero bug for small population sizes in total fertility rate.
- Refactored tests; they should now run locally in ~15 s.
- *GitHub info*: PR `448 <https://github.com/amath-idm/fp_analyses/pull/448>`__


Version 0.10.1 (2022-05-09)
---------------------------
- Fix ``Scenarios`` class.
- *GitHub info*: PR `433 <https://github.com/amath-idm/fp_analyses/pull/433>`__


Version 0.10.0 (2022-05-08)
---------------------------
- Moved Senegal parameters into FPsim.
- Added age of sexual debut.
- *GitHub info*: PR `427 <https://github.com/amath-idm/fp_analyses/pull/427>`__


Version 0.9.0 (2022-05-05)
--------------------------
- Added a new ``Scenarios`` class.
- *GitHub info*: PR `416 <https://github.com/amath-idm/fp_analyses/pull/416>`__


Version 0.8.0 (2021-08-28)
--------------------------
- Refactored the ``People`` object to use a new filtering-based approach.
- *GitHub info*: PR `219 <https://github.com/amath-idm/fp_analyses/pull/219>`__


Version 0.7.3 (2021-07-15)
--------------------------
- Fix bug to ensure that at least one process runs on each worker.
- *GitHub info*: PR `163 <https://github.com/amath-idm/fp_analyses/pull/163>`__


Version 0.7.2 (2021-07-14)
--------------------------
- Allow ``total_trials`` to be passed to an ``fp.Calibration`` object.
- *GitHub info*: PR `162 <https://github.com/amath-idm/fp_analyses/pull/162>`__


Version 0.7.1 (2021-07-14)
--------------------------
- Allow ``weights`` to be passed to an ``fp.Calibration`` object.
- *GitHub info*: PR `161 <https://github.com/amath-idm/fp_analyses/pull/161>`__


Version 0.7.0 (2021-06-29)
--------------------------
- Added new calibration plotting methods.
- Separated Experiment and Calibration into separate files, and renamed ``model.py`` to ``sim.py``.
- Fixed a bug where the age pyramid was being unintentionally modified in-place.
- *GitHub info*: PR `144 <https://github.com/amath-idm/fp_analyses/pull/144>`__


Version 0.6.5 (2021-06-11)
--------------------------
- Added R support; see ``examples/example_sim.R``.
- Fixed a bug where the age pyramid was being unintentionally modified in-place.
- *GitHub info*: PR `128 <https://github.com/amath-idm/fp_analyses/pull/128>`__


Version 0.6.4 (2021-06-10)
--------------------------
- Added a ``MultiSim`` class, which can handle parallel runs and uncertainty bounds.
- *GitHub info*: PR `124 <https://github.com/amath-idm/fp_analyses/pull/124>`__


Version 0.6.3 (2021-06-10)
--------------------------
- Fixed a bug where exposure correction by age was accidentally being clipped to the range [0,1], restoring behavior of the array-based model to match the object-based model (notwithstanding stochastic effects and other bugfixes).
- *GitHub info*: PR `119 <https://github.com/amath-idm/fp_analyses/pull/119>`__


Version 0.6.2 (2021-05-10)
--------------------------
- Added ``fp.Intervention`` and ``fp.Analyzer`` classes, which are much more flexible ways to modify and record the state of the simulation, respectively.
- Fixed a bug with only females being born.
- *GitHub info*: PR `100 <https://github.com/amath-idm/fp_analyses/pull/100>`__


Version 0.6.1 (2021-05-02)
--------------------------
- Renamed ``fp.Calibration`` to ``fp.Experiment``, and added a new ``fp.Calibration`` class, using Optuna.
- This allows the user to do e.g. ``calib = fp.Calibration(pars); calib.calibrate(calib_pars)``
- Calibrating a single parameter takes about 20 seconds for a single parameter and a small population size (500 people). Realistic calibrations should take roughly 10 - 60 minutes.
- *GitHub info*: PR `93 <https://github.com/amath-idm/fp_analyses/pull/93>`__


Version 0.6.0 (2021-05-01)
--------------------------
- Refactored the model to use an array-based implementation, instead of a loop over individual people.
- This results in a performance increase of roughly 20-100x, depending on the size of the simulation. In practice, this means that 50,000 people can be run in roughly the same amount of time as 500 could be previously.
- *GitHub info*: PR `92 <https://github.com/amath-idm/fp_analyses/pull/92>`__


Version 0.5.2 (2021-04-30)
--------------------------
- Added a new script, ``preprocess_data.py``, that takes large raw data files and preprocesses them down to only the essentials used in the model.
- This increases the performance of ``calib.run()`` (**not** counting model runtime) by a factor of 1000.
- *GitHub info*: PR `91 <https://github.com/amath-idm/fp_analyses/pull/91>`__


Version 0.5.1 (2021-04-29)
--------------------------
- Added ``summarize()`` and ``to_json()`` methods to ``Calibration``. Also added an ``fp.diff_summaries()`` method for comparing them.
- Added regression and benchmarking tests (current total time: 24 s).
- Added a code coverage script (current code coverage: 59%).
- Added default flags for which quantities to compute.
- Split the logic of ``Calibration`` out into more detail: e.g., initialization, running, and post-processing.
- *GitHub info*: PR `90 <https://github.com/amath-idm/fp_analyses/pull/90>`__
