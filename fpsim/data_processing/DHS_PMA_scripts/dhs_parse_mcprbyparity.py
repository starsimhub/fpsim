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

def parity_mcpr(data, name, parity_code, value_code, sample_weight, data_source):
    """
    Calculate relationship between parity (# of children given birth to)
    and modern contraception prevalence rate (mCPR).  
    How mCPR varies across different parity groups. 
    """
    
    total = data.groupby(parity_code)[sample_weight].sum()
    parity_group = data.groupby([parity_code, value_code])[sample_weight].sum()
    if data_source == 'dhs':
        n = total / 1000000
    else:
        n = total
    totals = n
    parity_group = 100 * parity_group.divide(total).fillna(0)
    parity_group.name = name
    
    return parity_group, totals

def caculate_mcpr(data, value_code, sample_weight_code, parity_code):
    """
    Calculate modern contraception prevalence rate (mCPR) from DHS data.


    """
    method_group = data.groupby([value_code])[sample_weight_code].sum()
    total = data[sample_weight_code].sum()
    method_group = 100 * method_group.divide(total).fillna(0)
    method_group.name = 'Percent'
    
    return method_group[3].round(1)

def process_cpr(data, method_value, method_code = 'v313'):
    """
    Read DHS parity data for method code and method category integer

    data: pandas dataframe processed by parity_mcpr function.
    method_code: DHS column code, default = v313 
    method_value: method type integer, <int>
        1 = Folkloric method
        2 = traditional method
        3 = Modern method
    """
    data = data.reset_index()
    method = data[data[method_code]==method_value]
    method.reset_index(drop=True, inplace=True)

    return method

def parse_dhs_mcprbyparity(data_filepath, output_filepath, country_value, value_code = 'v313', 
                           parity_code = 'v201', sample_weight_code = 'v005', country_code = 'v000', 
                           region_code=None, region_value=None, do_plot=False, plot_filepath=None):
    """
    Parse raw DHS .dta file and save probability of sexual debut age by age.

    Inputs: 
        data_file: Data files must be raw individual recode DHS data; <dir>
            (available for request at dhsprogram.com), in STATA format (.dta).
            For example, Kenya's 2014 DHS data could be 'KEIR72FL.DTA'or 'Kenya_Female_2014.DTA'
        value_code: contraceptive method column code in .dta file, default = v313; <str>
        parity_code: parity column code in .dta file, default = v201, <str>
        country_code: country column location is the .dta, v000; <str>
        country_value:  two letter country code and phase number from value_code v000; <str>
            Examples: Kenya (2014 = KE6), Cote d'Ivoir (2021 = CI8, 2012 = CI6),
                Niger (2012 = NI6), Pakistan (2018 = PK7), Nigeria (2018 = NG7)
        region_code: region code; <str>
            Examples: Pakistan (Sindh) 2018 .dta and Nigeria Kano & Kaduna 1990 .dta = v024, 
            Nigeria Kano & Kaduna 2018 .dta = 'sstate'
        region_value: region integer from value_code v024; <int>
            Examples: Pakistan Sindh (v024 2018) = 2; 
            Nigeria Kano & Kaduna (sstate 2018) = 100 & 110, respectively
            Nigeria Kano & Kaduna (v024 1990) = 12 & 11, respectively

    Returns:
        output_file: saved output .csv file to full path provided; <path>
        
    Other information:
        Data Source: DHS
        
        Data Source Code: v313
        Data Source Description: Type of contraceptive method categorizes the 
            current contraceptive method as either a
            modern method, a traditional method, or a folkloric method.
        Data Source Encoding:
            0 No method
            1 Folkloric method
            2 Traditional method
            3 Modern method
            8 Don't know
            9 Missing (m)
        
        Data Source Code: v201
        Data Source Description: Parity - Total number of children ever born. If there are fewer 
            than twenty births then this is the same as V224 (Number of entries in the birth history), 
            but if there are more than twenty births then this gives the full number, while V224 will be 20.
            Data Source Encoding: Total children ever born 0:20
        
        Data Source Code: v005
        Data Source Description: Women's individual sample weight is an 8 digit variable 
            with 6 implied decimal places. To use the sample
            weight divide it by 1000000 before applying the weighting factor. All sample weights are
            normalized such that the weighted number of cases is identical to the unweighted number of
            cases when using the full dataset with no selection. This variable should be used to weight
            all tabulations produced using the data file. For self-weighting samples this variable is equal
            to 1000000

    """

    import pandas as pd
    import math
    import matplotlib.pyplot as plt

    # Need raw.dta DHS data file here.
    dhs = pd.read_stata(data_filepath, convert_categoricals = False)

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
             exit()
            

    # To truncate parity at 7+
    parity_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:7, 9:7, 10:7, 11:7, 12:7, 13:7}
    dhs[parity_code] = dhs[parity_code].map(parity_dict)
    
    # Calculate total mCPR in DHS data to check matches STAT Compiler. Example Target for Kenya = 39.1%.
    mcpr_overall = caculate_mcpr(dhs, value_code, sample_weight_code, parity_code) # 
    data_source = 'dhs'
    table_dhs, totals = parity_mcpr(dhs, 'percent', parity_code, value_code, sample_weight_code, data_source)

    # Create DataFrames and select out only modern or traditional contraception
    parity_df_mod_dhs = pd.DataFrame(table_dhs).copy(deep=True)
    # parity_df_trad_dhs = pd.DataFrame(table_dhs).copy(deep=True)

    # Extract modern and traditional DHS values
    parity_modern_dhs = process_cpr(dhs, 3, method_code = 'v313') #process_mcpr(parity_df_mod_dhs, method='v313')
    # parity_trad_dhs = process_cpr(dhs, 2, method_code = 'v313') #process_tcpr(parity_df_trad_dhs, method='v313')

    # Calculate modern contraception use by parity
    totals_df = pd.DataFrame(totals)
    totals_df.columns = ['mcpr_totals']
    overall_percent = (totals_df['mcpr_totals'] / totals_df['mcpr_totals'].sum()) * 100
    totals_df['mcpr_percent'] = overall_percent
    totals_df = totals_df.reset_index().rename(columns={'v201': 'parity'})
    totals_df = totals_df.set_index(['parity'])

    # Save to provided output_filepath
    totals_df['country_value'] = country_value
    if (country_value != None) & (region_value != None):
        totals_df['country'] = country_value
        totals_df['region'] = region_value
        totals_df[['country', 'region', 'mcpr_percent']].to_csv(output_filepath + '/mcpr_by_parity.csv')
    elif (country_value != None):
        totals_df['country'] = country_value
        totals_df[['country', 'mcpr_percent']].to_csv(output_filepath + '/mcpr_by_parity.csv')
    else:
        totals_df.to_csv(output_filepath + '/mcpr_by_parity.csv')

    # Plotting
    if do_plot:
        mCPR_dhs = totals_df['mcpr_percent']
        ind = totals_df.index
        width = 0.35
    
        fig = plt.figure(figsize=(9, 7), dpi=120)
        ax = fig.add_subplot(111)
        rects1 = ax.bar(ind, mCPR_dhs, width, color='royalblue', label = f'Data - overall mCPR {mcpr_overall}')
    
        ax.set_ylabel('mCPR - %')
        ax.set_xlabel('Parity')
        ax.set_title('mCPR by parity in DHS data') # change title to reflect DHS country and year being used
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels( ('0', '1', '2', '3', '4', '5', '6', '7+'))
    
        ax.legend(loc = 'upper left')

        if plot_filepath:
            plt.savefig(plot_filepath, format='png')
            
def main():
    """
    Run dhs_parse_debutage.py from terminal for a single country or region.
    e.g. python dhs_parse_debutage.py /home/dhs_files/Kenya_Female_2014.DTA output/ KE
    """
    
    import argparse
    parser = argparse.ArgumentParser(description="Parse debut age probability weights from DHS .dta files")
    parser.add_argument('data_filepath', type=str, help='Full path to DHS .dta file')
    parser.add_argument('output_filepath', type=str, help='Full path to save the output debut_age.csv file')
    parser.add_argument('country_value', type=str, help='Two letter country code, e.g. "KE" for Kenya, "PK7" for Pakistan = "sstate"')
    parser.add_argument('--value_code', type=str, default='v313', help='DHS column code contraceptive method category: default = v313')
    parser.add_argument('--parity_code', type=str, default='v201', help='DHS column code for parity (total births): default = v201')
    parser.add_argument('--sample_weight_code', type=str, default='v005', help="DHS column code for women's sample weight: default = v005")
    parser.add_argument('--country_code', type=str, default='v000', help='DHS country code: default = v000')
    parser.add_argument('--region_code', type=str, help='Region code, e.g. Pakistan 2018, Nigeria 1990 = v024, Nigeria 2018 = "sstate"')
    parser.add_argument('--region_value', type=int, help='Region integer from region_code, e.g. Pakistan Sindh = 2, Nigeria Kano 2018 = 100, Nigeria Kaduna 2018 = 110')
    parser.add_argument('--do_plot', type=str, default = False, help='Turn on the mCPR by parity bar plots')
    parser.add_argument('--plot_filepath', type=str, default = 'dhs__mcpr_by_parity_plot', help='Full path to save png file')


    # Parsing a list
    # '--record_numbers',
    # nargs='+',  # Accepts multiple values
    # type=int,
    # help='A list of record integers'
    # )

    # Save all arguments in args
    args = parser.parse_args()

    if (args.region_code and args.region_value and args.do_plot):
        parse_dhs_mcprbyparity(args.data_filepath, args.output_filepath, args.country_value, 
                               value_code = args.value_code, parity_code = args.parity_code, 
                               sample_weight_code = args.sample_weight_code, country_code = args.country_code,
                               region_code = args.region_code, region_value = args.region_value, 
                               do_plot = args.do_plot, plot_filepath = args.plot_filepath)
    elif (args.region_code and args.region_value):
        parse_dhs_mcprbyparity(args.data_filepath, args.output_filepath, args.country_value, 
                               value_code = args.value_code, parity_code = args.parity_code, 
                               sample_weight_code = args.sample_weight_code, country_code = args.country_code,
                               region_code = args.region_code, region_value = args.region_value)
    else:
        parse_dhs_mcprbyparity(args.data_filepath, args.output_filepath, args.country_value, 
                               value_code = args.value_code, parity_code = args.parity_code, 
                               sample_weight_code = args.sample_weight_code, country_code = args.country_code)

if __name__ == '__main__':
    main()




