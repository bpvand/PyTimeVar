from PyTimeVar.datasets.utils import load_csv
import pandas as pd
import numpy as np

def load(start_date=None, end_date=None):
    """
    Load the inflation dataset, construct inflation rate dataset and optionally filter by date range.
    This dataset contains the inflation rate data from 1947 to 2024.
    
    Parameters
    ----------
    start_date : str, optional
        The start year-month to filter the data. 
        Format 'YYYY-MM'. 
        Minimum start year-month is 1947-02.
    end_date : str, optional
        The end year-month to filter the data.
        Format 'YYYY'. 
        Maximum end year is 2024-08.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the filtered data with columns 'Date' and CPIAUCSI.

    Warnings

    Prints warnings if the start year is earlier than the minimum year in the data or the end year is later than the maximum year in the data.
    """
    
    data = load_csv(__file__,"CPIAUCSL.csv")
    data = data.set_index('DATE')
    logdiff = (np.log(data.values[1:]) - np.log(data.values[:-1])) * 100
    data.iloc[1:, :] = logdiff
    data = data.iloc[1:, :]
    # Convert the 'Date' column to YYYY-MM-DD format for filtering
    data.index = pd.to_datetime(data.index)
    
    # Ensure the minimum start date is Jan 5 2015
    if start_date is None:
        start_date = '1947-02-01'
    if end_date is None:
        end_date = '2024-08-01'
    if pd.to_datetime(start_date) < pd.to_datetime('1947-01-01'):
            print(f"Warning: The start_date {start_date} is earlier than the minimum year in the data 1947. Data starts at minimum date.")
            start_date = '1947-01-01'
    data = data[data.index >= pd.to_datetime(start_date)]
    
    if end_date is not None:
        if pd.to_datetime(end_date) > pd.to_datetime('2024-08-01'):
            print(f"Warning: The end_date {end_date} is later than the maximum year in the data 2024-08-01. Data ends at maximum date.")
            end_date = '2024-08-01'
    data = data[data.index <= pd.to_datetime(end_date)]



    
    return data
