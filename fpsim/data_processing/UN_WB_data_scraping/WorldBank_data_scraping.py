"""
4/25/25, Emily Driano

This script can be run to pull the country data from the World Bank used for calibration. If the 'get' options below
are set to True, this script will utilize the World Bank API to scrape the data - and create files for:
- population (popsize.csv)
- total fertility rate (tfr.csv)
- maternal mortality (maternal_mortality.csv)
- infant mortality (infant_mortality.csv)
- basic world bank metrics: maternal mortality ratio, infant mortality rate, crude death rate, and crude birth rate (basic_wb.yaml)

The only setup required is to set the country name (all lower-case) and location iso code (which can be found in the file
country_codes.csv, located in the data_processing/data_scraping directory). The script will store any created files in a location
 directory for each respective country, in data_processing/data_scraping/scraped_data/{country_name}.

For more information about the World Bank database:
    https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation

"""

import os
import pandas as pd
import requests
import sciris as sc # Need to install with either "pip install sciris" or "conda install -c conda-forge sciris"
import json

######## VARIABLES TO MODIFY ########

# ISO2 COUNTRY CODES FOR REFERENCE
# Senegal = SN
# Kenya = KE
# Ethiopia = ET, country name = 'ethiopia
# Côte d'Ivoire (CI,CIV) ID = 384, country name = 'cotedivoire'
# Niger (NE,NER) ID = 562
# Nigeria (NG,NGA) ID = 566
# Pakistan (PK,PAK) = 586 

# Country name and location id(s); User must update these two variables prior to running.
country = 'cotedivoire'
iso_code = 'CI'

# Default global variables
startYear = 1960
endYear = 2030

# Set options for web scraping of data; all True by default
get_pop = True  # Annual population size
get_tfr = True  # Total fertility rate
get_maternal_mortality = True  # Maternal mortality
get_infant_mortality = True  # Infant mortality
get_basic_wb = True  # Includes maternal mortality, infant mortality, crude birth rate, & crude death rate

#####################################

country_dir = '../../locations/' + country + '/data/' #relative path to country

# API Base URL
base_url = "https://api.worldbank.org/v2"
url_suffix = 'format=json'

# If country folder doesn't already exist, create it (location in which created data files will be stored)
if not os.path.exists(f'{country_dir}'):
    os.makedirs(f'{country_dir}')

# Function that calls a GET request to the UN Data Portal API given the target/uri specified
def get_data(target):
    '''
    Pulls data using World Bank API
    https://api.worldbank.org/v2
    '''

    # Get the response, which includes the first page of data as well as information on pagination and number of records
    response = requests.get(target)
    if response.status_code != 200:
        raise ValueError(f"Request failed: {response.status_code} for URL: {target}")

    # Converts call into JSON
    j = response.json()

    # Converts JSON into a pandas DataFrame.
    df = pd.json_normalize(
        j[1])  # pd.json_normalize flattens the JSON to accommodate nested lists within the JSON structure

    # Loop until there are new pages with data
    while j[0]['page'] < j[0]['pages']:
        # Reset the target to the next page
        next = j[0]['page']+1
        target = target + f"&page={next}"

        # call the API for the next page
        response = requests.get(target)
        if response.status_code != 200:
            raise ValueError(f"Request failed: {response.status_code} for URL: {target}")

        # Convert response to JSON format
        j = response.json()
        if not isinstance(j, list) or len(j) < 2:
            raise ValueError(f"Unexpected API response format: {j}")

        # Store the next page in a data frame
        df_temp = pd.json_normalize(j[1])

        # Append next page to the data frame
        df_temp = df_temp.dropna(axis=1, how='all')  # Drop all-NA columns
        df = df.dropna(axis=1, how='all')  # Also drop from the main df
        df = pd.concat([df, df_temp.copy()], ignore_index=True)

        print("writing file...")

    return df


if get_pop:
    # Set params used in API
    ind = 'SP.POP.TOTL'

    # Call API
    target = base_url + f"/country/{iso_code}/indicators/{ind}?" + url_suffix
    df = get_data(target)

    # Process data to correct format
    df = df.filter(['date', 'value'])
    df = df.rename(columns={'date': 'year', 'value': 'population'})
    df.dropna(subset=['population'], inplace=True)
    df = df.sort_values('year')

    # Write data to csv
    df.to_csv(f'{country_dir}/popsize.csv', index=False)


if get_tfr:
    # Set params used in API
    ind = 'SP.DYN.TFRT.IN'

    # Call API
    target = base_url + f"/country/{iso_code}/indicators/{ind}?" + url_suffix
    df = get_data(target)

    # Process data to correct format
    df = df.filter(['date', 'value'])
    df = df.rename(columns={'date': 'year', 'value': 'tfr'})
    df.dropna(subset=['tfr'], inplace=True)
    df = df.sort_values('year')

    # Write data to csv
    df.to_csv(f'{country_dir}/tfr.csv', index=False)


if get_maternal_mortality:
    # Set params used in API
    ind = 'SH.STA.MMRT'

    # Call API
    target = base_url + f"/country/{iso_code}/indicators/{ind}?" + url_suffix
    df = get_data(target)

    # Process data to correct format
    df = df.filter(['date', 'value'])
    df = df.rename(columns={'date': 'year', 'value': 'probs'})
    df.dropna(subset=['probs'], inplace=True)
    df['probs'] = df['probs'].astype(int)
    df = df.sort_values('year')

    # Write data to csv
    df.to_csv(f'{country_dir}/maternal_mortality.csv', index=False)


if get_infant_mortality:
    # Set params used in API
    ind = 'SP.DYN.IMRT.IN'

    # Call API
    target = base_url + f"/country/{iso_code}/indicators/{ind}?" + url_suffix
    df = get_data(target)

    # Process data to correct format
    df = df.filter(['date', 'value'])
    df = df.rename(columns={'date': 'year', 'value': 'probs'})
    df.dropna(subset=['probs'], inplace=True)
    df = df.sort_values('year')

    # Write data to csv
    df.to_csv(f'{country_dir}/infant_mortality.csv', index=False)


if get_basic_wb:

    # Indicator codes
    mmr_ind = 'SH.STA.MMRT'     # Maternal mortality rate
    imr_ind = 'SP.DYN.IMRT.IN'  # Infant mortality rate
    cbr_ind = 'SP.DYN.CBRT.IN'  # Crude birth rate
    cdr_ind = 'SP.DYN.CDRT.IN'  # Crude death rate

    # Get MMR for latest_data_year
    target = base_url + f"/country/{iso_code}/indicators/{mmr_ind}?" + url_suffix
    df = get_data(target)
    # Store latest year of data available
    latest_data_year = df[df['value'].notna()].sort_values('date', ascending=False).iloc[0]['date']
    mmr = df[df['date'] == latest_data_year]['value'].values[0]

    # Get IMR for latest_data_year
    target = base_url + f"/country/{iso_code}/indicators/{imr_ind}?date={latest_data_year}&" + url_suffix
    df = get_data(target)
    imr = df['value'].values[0]

    # Get CBR for latest_data_year
    target = base_url + f"/country/{iso_code}/indicators/{cbr_ind}?date={latest_data_year}&" + url_suffix
    df = get_data(target)
    cbr = df['value'].values[0]

    # Get CDR for latest_data_year
    target = base_url + f"/country/{iso_code}/indicators/{cdr_ind}?date={latest_data_year}&" + url_suffix
    df = get_data(target)
    cdr = df['value'].values[0]

    # Define dictionary for basic_wb_csv
    basic_wb_data = {
        'maternal_mortality_ratio': mmr, # Per 100,000 live births, From World Bank: https://data.worldbank.org/indicator/SH.STA.MMRT?locations=KE
        'infant_mortality_rate': imr,  # Per 1,000 live births, From World Bank
        'crude_death_rate': cdr,  # Per 1,000 inhabitants, From World Bank
        'crude_birth_rate': cbr,  # Per 1,000 inhabitants, From World Bank
    }

    # Write to yaml file
    with open(f'{country_dir}/basic_wb.yaml', 'w') as file:
        file.write(json.dumps(basic_wb_data))
