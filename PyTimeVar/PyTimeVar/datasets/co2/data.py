from PyTimeVar.datasets.utils import load_csv
import pandas as pd
import numpy as np

def load(start_date=None, end_date=None, regions=None):
    """
    Load the CO2 emissions dataset and optionally filter by date range and/or countries.
    This dataset contains the emissions data from 1900 to 2017, for a range of countries.
    
    Parameters
    ----------
    start_date : str, optional
        The start year to filter the data. 
        Format 'YYYY'. 
        Minimum start year is 1900.
    end_date : str, optional
        The end year to filter the data.
        Format 'YYYY'. 
        Maximum end year is 2017.
    regions : list, optional
        Regions to be selected from data. 
        Available options are:
            AUSTRALIA, AUSTRIA, BELGIUM, CANADA, DENMARK, FINLAND, FRANCE, GERMANY,
            ITALY, JAPAN, NETHERLANDS, NEW ZEALAND, NORWAY, PORTUGAL, SPAIN, SWEDEN,
            SWITZERLAND, UNITED KINGDOM, UNITED STATES, CHILE, SRI LANKA, URUGUAY, 
            BRAZIL, GREECE, PERU, VENEZUELA, COLOMBIA, ECUADOR, INDIA, MEXICO

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the filtered data with columns 'Date' and regions.

    Warnings
    --------
    Prints warnings if any provided regions are not found in the dataset.
    Prints warnings if the start year is earlier than the minimum year in the data or the end year is later than the maximum year in the data.
    """
    
    data = load_csv(__file__,"Emissions_19002017.csv", sep=",")
    
    # Convert the 'Date' column to YYYY-MM-DD format for filtering
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Ensure the minimum start date is Jan 5 2015
    if start_date is None:
        start_date = '1900-01-01'
    if end_date is None:
        end_date = '2017-01-01'
    if pd.to_datetime(start_date) < pd.to_datetime('1900-01-01'):
            print(f"Warning: The start_date {start_date} is earlier than the minimum year in the data 1900. Data starts at minimum date.")
            start_date = '1900-01-01'
    data = data[data['Date'] >= pd.to_datetime(start_date)]
    
    if end_date is not None:
        if pd.to_datetime(end_date) > pd.to_datetime('2017-01-01'):
            print(f"Warning: The end_date {end_date} is later than the maximum year in the data 2017-01-01. Data endsd at maximum date.")
            end_date = '2017-01-01'
    data = data[data['Date'] <= pd.to_datetime(end_date)]
    dates = data['Date']
    
    # Select the 'Date' and variable columns
    available_columns = data.columns
    if regions:
        regions = [x.upper() for x in regions]
        selected_columns = regions
        invalid_regions = [region for region in regions if region not in available_columns]

        if invalid_regions:
            print(
                f"Warning: The following regions are not available in the dataset and will be ignored: {invalid_regions}"
            )

        data = data[selected_columns]
  
    data['Date'] = dates
    # Set 'Date' as the index
    data.set_index('Date', inplace=True)
    
    return data
