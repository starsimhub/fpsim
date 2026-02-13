"""
Updated 7/1/25, Emily Driano

This script can be run to pull country data from the UN Data Portal and UN World Population Prospects (WPP). If the 'get'
options below are set to True, this script will utilize the UN Data Portal API to scrape the data - and create files for:
- contraceptive prevalence rate (cpr.csv)
- mortality probability (mortality_prob.csv)
- mortality trend (mortality_trend.csv)
- age-specific fertility rate (asfr.csv)
- age-specific population pyramid, by 5-year age groups (age_pyramid.csv)

The setup required is to set the country name and either the location ID or ISO2 code (all of which
 can be found in the file country_codes.csv, located in the data_processing/data_scraping folder). Please note that the
 'country' parameter MUST match the country in the 'name' column in country_codes.csv; otherwise, the mortality_prob and
 population pyramid will not be able to be generated.

 The script will store any created files in a location directory for each respective country, in data_processing/data_scraping/scraped_data/{country_name}.
 To use the UN Data Portal API, you must also email population@un.org requesting the authorization token to access the
 Data Portal data endpoints. They will provide you with a token in the following format: Bearer xyz (with the space between
 the word Bearer and the token itself). Set ONLY  the 'xyz' portion of the token (excluding 'Bearer ') as an environment
 variable 'UN_AUTH_TOKEN'. This will then be read in by the script when calling the UN Portal endpoints.

As the WPP files are location-agnostic and can be reused for other contexts, they are stored as CSVs in data_scraping/scraped_data.

For more information about the UN Data Portal API, visit this site: https://population.un.org/dataportal/about/dataapi
For more information about the UN World Population Prospects: https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=CSV%20format

"""
import gzip
import os
import shutil
import pandas as pd
import requests
import sciris as sc
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

######## VARIABLES TO MODIFY ########

# COUNTRY CODES FOR REFERENCE
# Senegal ID = 686
# Kenya ID = 404
# Ethiopia ID = 231
# India ID = 356
# Côte d'Ivoire (CI,CIV) ID = 384, country = 'Côte d\'Ivoire'
# Niger ID = 562 
# Nigeria ID = 566 
# Pakistan = 586 

# Country name and location id(s) (per UN Data Portal); User must update these two variables prior to running.
country = 'Côte d\'Ivoire'
country_foldername = 'cotedivoire' # same as country variable (except country='Côte d\'Ivoire', country_foldername='cotedivoire')
location_id = 384

# Default global variables
startYear = 1950            # Specifies the start year of projection data retrieved (e.g. data from '1960'). Used for cpr, asfr, and mortality trend
endYear = 2023              # Specifies the year desired to indicate the 'most recently available data' (ensure available in the UN; may be updated over time); used for mortality prob, asfr, and age pyramid
endProjectionYear = 2030    # Specifies the end year of projection data retrieved, e.g. data until '2030' (ensure available in the UN; may be updated over time); used for cpr and mortality trend

# Set options for web scraping of data; all True by default
get_cpr = True  # Contraceptive prevalence rate
get_asfr = True  # Age-specific fertility rate
get_mortality_trend = True  # Mortality trend
get_mortality_prob = True  # Mortality prob
get_pop = True  # Population pyramid (5-year age groups for both male/female sex)

# Authorization token from the UN Data Population Division
auth_token = os.getenv('UN_AUTH_TOKEN')
if not auth_token:
    raise ValueError("No authorization token provided. Please set UN_AUTH_TOKEN environment variable.")
headers = {"Authorization": f"Bearer {auth_token}"}

# Variables used in pulling WPP data. **NOTE: The exact stem for these may change over time, i.e. from 'WPP2023' to 'WPP2024'
pop_stem = 'WPP2024_Population1JanuaryByAge5GroupSex_Medium'
female_mort_stem = 'WPP2024_Life_Table_Complete_Medium_Female_1950-2023'
male_mort_stem = 'WPP2024_Life_Table_Complete_Medium_Male_1950-2023'

#####################################

# Store local directories as variables
country_dir = '../../locations/' + country_foldername + '/data/' #relative path to country
filesdir = country_dir + '/scraped_data/'

# API Base URLs
base_url = "https://population.un.org/dataportalapi/api/v1"
wpp_base_url = "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/"

# If country folder doesn't already exist, create it (location in which created data files will be stored)
if not os.path.exists(f'{country_dir}'):
    os.makedirs(f'{country_dir}')
if not os.path.exists(f'{filesdir}'):
    os.makedirs(f'{filesdir}')

def get_UN_data(target):
    '''
    Scrapes data from UN data portal using the UN Data Portal API
    https://population.un.org/dataportal/about/dataapi
    '''
    # Get the response, which includes the first page of data as well as information on pagination and number of records
    response = requests.get(target, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Request failed: {response.status_code} for URL: {target}")

    # Converts call into JSON
    j = response.json()

    # Converts JSON into a pandas DataFrame.
    df = pd.json_normalize(j['data'])  # pd.json_normalize flattens the JSON to accommodate nested lists within the JSON structure

    # Loop until there are new pages with data
    while j['nextPage'] is not None:
        # call the API for the next page
        next_page = f'{target}?pageNumber={j["pageNumber"]+1}&pageSize=100' # NOTE: Currently using j['nextPage'] as target has issue; resolving with UN Pop Division
        response = requests.get(next_page, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Request failed: {response.status_code} for URL: {target}")

        # Convert response to JSON format
        j = response.json()

        # Store the next page in a data frame
        df_temp = pd.json_normalize(j['data'])

        # Append next page to the data frame
        df = pd.concat([df, df_temp])
        logger.info("Pulling UN data...")

    df = df.loc[(df['variant'] == "Median")]

    return df


def get_UN_WPP_data(label='', file_stem=None, columns=None, force=False, tidy=True):
    '''
    Downloads data from UN Population Division
    '''

    sc.heading(f'Downloading {label} data...')
    T = sc.timer()

    # Download data if it's not already in the directory
    url = f'{wpp_base_url}{file_stem}.csv.gz'
    local_csv = f'{filesdir}/{file_stem}.csv'
    local_gz = f'{filesdir}/{file_stem}.csv.gz'

    if force or not os.path.exists(local_csv):
        logger.info(f'\nDownloading from {url}, this may take a while...')
        sc.download(url, filename=local_gz)

        # Decompress the .csv.gz file
        with gzip.open(local_gz, 'rb') as f_in:
            with open(local_csv, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Extract the parts used in the model and save
        df = pd.read_csv(local_csv, usecols=columns)
        df.to_csv(f'{local_csv}')

        if tidy:
            logger.info(f'Removing {local_gz}')
            os.remove(local_gz)
    else:
        logger.info(f'Skipping {local_csv}, already downloaded')

    T.toctic(label=f'  Done with {label}')

    T.toc(doprint=False)
    logger.info(f'Done with {label}: took {T.timings[:].sum():0.1f} s.')

    return


# Called if creating country file cpr.csv
if get_cpr:
    # Set params used in API
    cpr_ind = 1
    mcpr_ind = 2

    # Call API
    target = base_url + f"/data/indicators/{cpr_ind},{mcpr_ind}/locations/{location_id}/start/{startYear}/end/{endProjectionYear}"
    df = get_UN_data(target)

    # Reformat data
    df_cpr = df.loc[(df['category'] == 'All women') & (df['variantLabel'] == 'Median') & (df['indicatorId'] == cpr_ind)]
    df_mcpr = df.loc[(df['category'] == 'All women') & (df['variantLabel'] == 'Median') & (df['indicatorId'] == mcpr_ind)]
    df_cpr = df_cpr.filter(['timeLabel', 'value'])
    df_mcpr = df_mcpr.filter(['timeLabel', 'value'])
    df_cpr.rename(columns={'timeLabel': 'year', 'value': 'cpr'}, inplace=True)
    df_mcpr.rename(columns={'timeLabel': 'year', 'value': 'mcpr'}, inplace=True)
    df2 = pd.merge(df_cpr, df_mcpr, on='year')

    # Save file
    logger.info(f'Writing {country_dir}/cpr.csv')
    df2.to_csv(f'{country_dir}/cpr.csv', index=False)


# Called if creating country file asfr.csv
if get_asfr:
    # Set params used in API
    ind = 17

    # Call API
    target = base_url + f"/data/indicators/{ind}/locations/{location_id}/start/{startYear}/end/{endYear}"
    df = get_UN_data(target)

    # Reformat data
    df = df.loc[(df['variantLabel'] == 'Median')]
    df.rename(columns={'timeLabel': 'year'}, inplace=True)
    df = df.pivot(index='year', columns='ageLabel', values='value')
    df = df.round(decimals=3).drop('50-54', axis=1)

    # Save file
    logger.info(f'Writing {country_dir}/asfr.csv')
    df.to_csv(f'{country_dir}/asfr.csv')


# Called if creating country file mortality_trend.csv
if get_mortality_trend:
    # Set params used in API
    ind = 59

    # Call API
    target = base_url + f"/data/indicators/{ind}/locations/{location_id}/start/{startYear}/end/{endProjectionYear}"
    df = get_UN_data(target)

    # Reformat data
    df = df.filter(['timeLabel', 'value'])
    df.rename(columns={'timeLabel': 'year', 'value': 'crude_death_rate'}, inplace=True)

    # Save file
    logger.info(f'Writing {country_dir}/mortality_trend.csv')
    df.to_csv(f'{country_dir}/mortality_trend.csv', index=False)


# Called if creating country file mortality_prob.csv
if get_mortality_prob:

    # Get mortality data from UN Portal
    columns = ["Location", "Time", "AgeGrpStart", "qx"]
    get_UN_WPP_data(label='age', file_stem=female_mort_stem, columns=columns)
    get_UN_WPP_data(label='age', file_stem=male_mort_stem, columns=columns)

    # Load female data from scraped data
    df = pd.read_csv(f'{filesdir}/{female_mort_stem}.csv')
    df_female = df.loc[(df['Location']==country) & (df['Time']==endYear)]
    df_female = df_female.filter(["AgeGrpStart", "qx"])
    df_female.rename(columns={'AgeGrpStart': 'age', 'qx': 'female'}, inplace=True)

    # Load male data from scraped data
    df = pd.read_csv(f'{filesdir}/{male_mort_stem}.csv')
    df_male = df.loc[(df['Location']==country) & (df['Time']==endYear)]
    df_male = df_male.filter(["AgeGrpStart", "qx"])
    df_male.rename(columns={'AgeGrpStart': 'age', 'qx': 'male'}, inplace=True)

    # Combine two dataframes
    df_combined = pd.merge(df_female, df_male, on="age")

    # Save file
    logger.info(f'Writing {country_dir}/mortality_prob.csv')
    df_combined.to_csv(f'{country_dir}/mortality_prob.csv', index=False)


# Called if scraping population data from WPP to create the population pyramid
if get_pop:
    ''' Import population sizes by age from UNPD '''
    columns = ["Location", "Time", "AgeGrpStart", "PopMale", "PopFemale"]
    get_UN_WPP_data(label='pop', file_stem=pop_stem, columns=columns)

    # Load female data from scraped data
    df = pd.read_csv(f'{filesdir}/{pop_stem}.csv')
    filtered_df = df.loc[(df['Location']==country) & (df['Time']==endYear)]
    filtered_df = filtered_df.filter(["AgeGrpStart", "PopMale", "PopFemale"])
    filtered_df.rename(columns={'AgeGrpStart': 'age', 'PopMale': 'male', 'PopFemale': 'female'}, inplace=True)
    filtered_df[['male', 'female']] = (filtered_df[['male', 'female']] * 1000).astype(int)

    # Save file
    logger.info(f'Writing {country_dir}/age_pyramid.csv')
    filtered_df.to_csv(f'{country_dir}/age_pyramid.csv', index=False)
