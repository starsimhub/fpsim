# Data Processing for FPsim

This folder contains scripts and resources to prepare location-specific input data for FPsim models. It includes:

- `DHS_PMA_scripts/`: R scripts to process user-supplied DHS and PMA survey datasets
- `UN_WB_data_scraping/`: Python scripts to automatically pull global indicators from UN and World Bank APIs
- `country_codes.csv`: ISO code reference for use in Python scripts
- `DHS_PMA_scripts/config.R`: Configuration file to customize file paths and behavior for R scripts
- `DHS_PMA_scripts/run_all.R`: Controller script to execute selected R processing scripts in sequence

Each output CSV file represents a specific metric needed by FPsim (e.g., `popsize.csv`, `cpr.csv`, `maternal_mortality.csv`, etc.).
**NOTE: These scripts do not create all of the data files needed to run a model. See 'Model Parameters' chart in locations/README.md
for all of the required files (some are created from studies, etc.)

---

## DHS and PMA Data Scripts (`DHS_PMA_scripts/`)

These R scripts process raw DHS and PMA survey data for use in FPsim. They generate country-specific input files such as sexual activity rates, method mix matrices, birth spacing distributions, etc.

### Requirements

1. **Download latest DHS datasets**  
   Create a DHS account and download an Individual Recode (IR) DTA file from
   https://dhsprogram.com/data/. We suggest using the most recent Standard DHS dataset DTA file, or you may use a Continuous DHS dataset DTA file if it has the needed variables for the script(s) you would like to run.

2. **Download PMA datasets for the last 3 phases**  
   Register and download data from https://www.pmadata.org/data/available-datasets. Use the PMA 'Household and Female' datasets DTA files (all three phases).

3. **Configure paths**  
   Edit `config.R` to:
   - Set local paths to DHS and PMA data
   - Define output directory
   - Customize other processing options if needed

4. **Run processing pipeline**  
   Use `run_all.R` to selectively or fully execute all R scripts in sequence.

---

## UN and World Bank Data Scraping Scripts (`UN_WB_data_scraping/`)

These Python scripts use public APIs to fetch demographic and health indicators from:

- **UN World Population Prospects**
- **UN Data Portal**
- **World Bank Open Data**

### Requirements 
1. **Set UN Authorization Token**   
   To use the UN Data Portal API, you must email population@un.org requesting the authorization token to access the
 Data Portal data endpoints. Note that the standard format is: Bearer xyz (with the space between the word Bearer and the token itself).
 Set the 'xyz' portion of the token (excluding 'Bearer ') as an environment variable 'UN_AUTH_TOKEN'. This will then be
 read in by the script when calling the UN Portal endpoints.
2. **Set Configuration**  
   Modify the configuration section at the top of the respective script to designate the country name and ISO code (which can be referenced in `country_codes.csv`.)

### Scripts

- `UN_data_scraping.py`: 
  - Fetches data on population pyramids, mortality rates, ASFR, urban proportions, etc.
- `WorldBank_data_scraping.py`: 
  - Fetches MMR, IMR, population size, TFR, and other health metrics

### Output

Scraped data by default are saved to `UN_WB_data_scraping/scraped_data/<country_name>/`


---

## Outputs Overview

Below is a reference table summarizing each script, the metric it processes, and the output file(s) it creates.

| Script                         | Metric                                                                                                                                                           | Output Filename                                                                       |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `age_at_first_birth.R`         | Age at first birth                                                                                                                                               | `afb.table.csv`                                                                       |
| `age_partnership.R`            | Age-based probability of partnership                                                                                                                             | `age_partnership.csv`                                                                 |
| `ageparity.R`                  | Age-parity distribution                                                                                                                                          | `ageparity.csv`                                                                       |
| `birth_spacing.R`              | Birth spacing probabilities                                                                                                                                      | `birth_spacing_dhs.csv`                                                               |
| `breastfeeding_stats.R`        | Breastfeeding duration parameters                                                                                                                                | `bf_stats.csv`                                                                        |
| `contra_coeffs.R`              | Contraception function coefficients                                                                                                                              | `contra_coef_*.csv`, `spline_25_40.csv`                                                |
| `education_dhs_pma.R`          | Education objective/stop/init                                                                                                                                    | `edu_objective.csv`, `edu_stop.csv`, `edu_initialization.csv`                         |
| `extract_urban.R`              | Urban proportion                                                                                                                                                 | `urban.csv`                                                                           |
| `make_matrices.R`              | Method mix switching matrices                                                                                                                                    | `method_mix_matrix_switch.csv`                                                        |
| `method_mix.R`                 | Method use / mix                                                                                                                                                 | `mix.csv`, `use.csv`                                                                  |
| `postpartum_recode.R`          | LAM and postpartum sexual activity                                                                                                                               | `lam.csv`, `sexually_active_pp.csv`                                                   |
| `sexual_activity.R`            | Sexual activity of debuted women                                                                                                                                 | `sexually_active.csv`                                                                 |
| `sexual_debut_age_probs.R`     | Sexual debut age distribution                                                                                                                                    | `debut_age.csv`                                                                       |
| `subnational.R`                | Subnational metrics                                                                                                                                              | **Needs to be updated                                                                 |
| `time_on_method.R`             | Time on method coefficients                                                                                                                                      | `method_time_coefficients.csv`                                                        |
| `wealth.R`                     | Wealth quintile distribution                                                                                                                                     | `wealth.csv`                                                                          |
| `UN_data_scraping.py`          | Contraceptive prevalence rate, mortality probability, mortality trend, age-specific fertility rate, age-specific population pyramid by 5-yr age groups           | `cpr.csv`, `mortality_prob.csv`, `mortality_trend.csv`, `asfr.csv`, `age_pyramid.csv` |
| `WorldBank_data_scraping.py` | Population size, total fertility rate, maternal mortality, infant mortality, maternal mortality ratio, infant mortality rate, crude death rate, crude birth rate | `basic_wb.yaml`                                                                       |


---

## Related Files and Documentation

See the main `locations/README.md` for an overview of how these data outputs are integrated into FPsim models.

---


