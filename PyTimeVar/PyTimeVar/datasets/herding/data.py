from PyTimeVar.datasets.utils import load_csv
import pandas as pd
import numpy as np

def load(start_date=None, end_date=None):
    """
    Load the Herding dataset and optionally filter by date range.
    This dataset contains the herding data from Jan 5 2015 to Apr 29 2022.
    
    Parameters
    ----------
    start_date : str, optional
        The start_year to filter the data. Format 'YYYY-MM-DD'. Minimum start year is 1900.
    end_date : str, optional
        The end_year to filter the data. Format 'YYYY'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the filtered data with columns 'Date' and 'Herding'.

    Warnings
    --------
    Prints warnings if any provided regions are not found in the dataset.
    Prints warnings if the start_year is earlier than the minimum year in the data or the end_year is later than the maximum year in the data.
    """
    
    data = load_csv(__file__,"CN HERDING.csv", sep=",") # change to load_csv
    
    # Convert the 'Date' column to YYYY-MM-DD format for filtering
    data['Date'] = pd.to_datetime(data['Date'])
    
    
    # Ensure the minimum start date is Jan 5 2015
    if start_date is None:
        start_date = '2015-05-01'
    if end_date is None:
        end_date = '2022-04-29'
    if pd.to_datetime(start_date) < pd.to_datetime('2015-05-01'):
            print(f"Warning: The start_date {start_date} is earlier than the minimum year in the data 2015-05-01. Data starts at minimum date.")
            start_date = '2015-05-01'
    data = data[data['Date'] >= pd.to_datetime(start_date)]
    
    if end_date is not None:
        if pd.to_datetime(end_date) > pd.to_datetime('2022-04-29'):
            print(f"Warning: The end_date {end_date} is later than the maximum year in the data 2022-04-29. Data endsd at maximum date.")
            end_date = '2022-04-29'
    data = data[data['Date'] <= pd.to_datetime(end_date)]
    
    # Select the 'Date' and variable columns
    data = data[['Date', 'CSAD_AVG', 'AVG_RTN']]
    data['RTN_ABS'] = data['AVG_RTN'].abs()
    data['RTN_2'] = data['AVG_RTN']**2
    data['Intercept'] = 1
    
    # Set 'Date' as the index
    data.set_index('Date', inplace=True)
    
    
    return data

if __name__ == '__main__':
    # Test the load function
    data = load(start_date='2015-01-05', end_date='2022-01-05')
    print(data.head())
    print(data.tail())