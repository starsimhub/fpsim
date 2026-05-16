import pandas as pd
import numpy as np
import subprocess
import sys

# List of required packages
required_packages = ['argparse']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def parse_dhs_debutage(data_filepath, output_filepath, country_value, value_code = 'v531', country_code = 'v000',  sample_weight_code = 'v005', region_code=None, region_value=None):
    """
    Parse raw DHS .dta file and save probability of sexual debut age by age.

    Inputs: 
        data_file: Data files must be raw individual recode DHS data; <dir>
            (available for request at dhsprogram.com), in STATA format (.dta).
            For example, Kenya's 2014 DHS data could be 'KEIR72FL.DTA'or 'Kenya_Female_2014.DTA'
        value_code: debut_age column code in .dta file, either 'v531' or 'v525'; <str>
            v531 - age at first sex (imputed with age at first union) 
            (alternatively: v525 - age at first sexual intercourse)
        country_code: country column location is the .dta, v000; <str>
        country_value:  two letter country code and phase number from value_code v000; <str>
            Examples: Kenya (2014 = KE6), Cote d'Ivoir (2021 = CI8, 2012 = CI6),
                Niger (2012 = NI6), Pakistan (2018 = PK7), Nigeria (2018 = NG7)
        region_code: region code; <str>
            Examples: Pakistan (Sindh) 2018 .dta and Nigeria Kano & Kaduna 1990 .dta = v024, 
            Nigeria Kano & Kaduna 2018 .dta = 'sstate'
        region_code: region integer from value_code v024; <int>
            Examples: Pakistan Sindh (v024 2018) = 2; 
            Nigeria Kano & Kaduna (sstate 2018) = 100 & 110, respectively
            Nigeria Kano & Kaduna (v024 1990) = 12 & 11, respectively

    Returns:
        output_file: saved output .csv file to full path provided; <path>
    
    Other information:
        Data Source: DHS
        Data Source Code: v531
        Data Source Question: Age at first sexual intercourse - imputed. 
            This is the same as V525, except for respondents
            who reported that their first sexual intercourse was at the time of their union. For these
            cases, the age at first sex is taken from the age at first union. In cases where the age at first
            sex was inconsistent with the age at conception of the first child, but only by one year (V532
            = 3), the age at first sex was reduced by one year, consistent with the "Rule of one" applied
            in DHS I. Other cases flagged as inconsistent on variable V532 (codes 1, 2, 4, 5) are recoded
            as 97 (inconsistent). Cases coded 6 on V532 are not changed.
        Data Source Encoding: 
            0 No flag
            1 After interview
            2 After conception >= 1 year
            3 After conception < 1 year
            4 At marriage, but never married
            5 At marriage, but after conception
            6 After marriage
            (na) Not applicable
        
        Data Source Code: v525
        Data Source Question: "Age at first sexual intercourse. Respondents who had never had sex are coded 0. 
            The response category "First sexual intercourse at first union" has been added in DHS III."
        Data Source Encoding: 
            0: Not had sex
            7-49: age
            96: At first union
            97: Inconsistent
            98: Don't know
            99: Missing(m)
    """

    import pandas as pd
    import math
    import warnings

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    # Need raw.dta DHS data file here.
    dhs = pd.read_stata(data_filepath, convert_categoricals = False)
    # print(dhs['v000'].value_counts())

    # If only calculating for a specific region like for Pakistan (Sindh)
    # and Nigeria (Kano & Kaduna)
    # Filter to only the relevent region
    if (country_value not in dhs[country_code].values):
        print('Error: wrong country_value for this file')
        print("Country_values in this file:\n", dhs[country_code].value_counts())
        exit()
    
    if (country_value != None) & (region_code != None) & (region_value != None):
        if (country_value in dhs[country_code].values) & (region_value in dhs[region_code].values):
            dhs = dhs[(dhs[country_code] == country_value) & (dhs[region_code] == region_value)]
        else:
            print('Error: country_value,  region_code, or region_value not found in this file, please check inputs\n')
            print(dhs[country_code].value_counts())
            # print(dhs[region_code].value_counts())
            exit()
            
    
    # Calculate Women's individual sample weight from DHS value 'v005'
    dhs['wgt'] = dhs['v005'].values/1000000
    
    # Filters all responses that are not an age between 9 and 50
    dhs_debut = dhs[(dhs[value_code] != 0) & (dhs[value_code] < 50) & (dhs[value_code] >= 10)]

    # Calculates 
    age_probs = dhs_debut[[value_code, 'wgt']].groupby([value_code]).sum()

    # Calculate age probability weights 
    age_probs['prob_weights'] = age_probs['wgt']/age_probs['wgt'].sum()
    probs_sum = age_probs['prob_weights'].sum()
    assert math.isclose(1, probs_sum), f'Sexual debut age probabilities should add to 1, not {probs_sum}'

    age_probs = age_probs[['prob_weights']]
    age_probs = age_probs.reset_index().rename(columns= {value_code: 'age', 'prob_weights': 'probs'})
    age_probs = age_probs.set_index('age')
    
    if (country_value != None) & (region_value != None):
        age_probs['country'] = country_value
        age_probs['region'] = region_value
        age_probs[['country', 'region', 'probs']].to_csv(output_filepath + '/debut_age.csv')
    elif (country_value != None):
        age_probs['country'] = country_value
        age_probs[['country', 'probs']].to_csv(output_filepath + '/debut_age.csv')
    else:
        age_probs.to_csv(output_filepath + '/debut_age.csv')

def main():
    """
    Run dhs_parse_debutage.py from terminal for a single country or region.
    e.g. python dhs_parse_debutage.py /home/dhs_files/Kenya_Female_2014.DTA output/ KE
    """
    
    import argparse

    parser = argparse.ArgumentParser(description="Parse debut age probability weights from DHS .dta files")
    parser.add_argument('data_filepath', type=str, help='Full path to DHS .dta file')
    parser.add_argument('output_filepath', type=str, help='Full path to save the output dhs__debut_age.csv file')
    parser.add_argument('country_value', type=str, help='Two letter country code, e.g. Kenya 2014 KE6, Pakistan 2018 PK7')
    parser.add_argument('--value_code', type=str, default='v531', help='DHS value code for debut age: default = v531, option = v525')
    parser.add_argument('--sample_weight_code', type=str, default='v005', help="DHS column code for women's sample weight: default = v005")
    parser.add_argument('--country_code', type=str, default='v000', help='DHS country code: default = v000')
    parser.add_argument('--region_code', type=str, help='Region code, e.g. Pakistan 2018, Nigeria 1990 = v024, Nigeria 2018 = sstate')
    parser.add_argument('--region_value', type=int, help='Region integer from region_code, e.g. Pakistan Sindh = 2, Nigeria Kano 2018 = 100, Nigeria Kaduna 2018 = 110')

    # Parsing a list
    # '--record_numbers',
    # nargs='+',  # Accepts multiple values
    # type=int,
    # help='A list of record integers'
    # )

    args = parser.parse_args()
    if args.region_code and args.region_value:
        parse_dhs_debutage(args.data_filepath, args.output_filepath, country_value = args.country_value, 
                           value_code = args.value_code, country_code = args.country_code,    
                           region_code=args.region_code, region_value = args.region_value)
    else:
        parse_dhs_debutage(args.data_filepath, args.output_filepath, country_value = args.country_value,
                           value_code = args.value_code, sample_weight_code = args.sample_weight_code,
                           country_code = args.country_code)


if __name__ == '__main__':
    main()




