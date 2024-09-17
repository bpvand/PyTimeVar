from PyTimeVar.datasets.utils import load_csv
import pandas as pd
import numpy as np

def load(start_date=None, end_date=None, data_replication=False):
    """
    Load the Herding dataset and optionally filter by date range.
    This dataset contains the herding data from Jan 5 2015 to Apr 29 2022.
    
    Parameters
    ----------
    start_date : str, optional
        The start date to filter the data. 
        Format 'YYYY-MM-DD'.
        Minimum start date is 2015-01-05.
    end_date : str, optional
        The end date to filter the data.
        Format 'YYYY-MM-DD'.
        Maximum end date is 2022-04-29.
    data_replication : bool, optional
        If True, the data is returned to replicate the output in Section 4.2 of the reference paper Song et al. (2024).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the filtered data with columns 'Date' and regressors.

    Warnings
    --------
    Prints warnings if the start_date is earlier than the minimum date in the data or the end_date is later than the maximum date in the data.
    """
    
    data = load_csv(__file__,"CN HERDING.csv", sep=",") # change to load_csv
    
    # Convert the 'Date' column to YYYY-MM-DD format for filtering
    data['Date'] = pd.to_datetime(data['Date'])
    
    
    # Ensure the minimum start date is Jan 5 2015
    if start_date is None:
        start_date = '2015-01-05'
    if end_date is None:
        end_date = '2022-04-29'
    if pd.to_datetime(start_date) < pd.to_datetime('2015-01-05'):
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
    # For modelling gamma_3

    if data_replication == True:
        CSAD = data["CSAD_AVG"]
        MRTN = data["AVG_RTN"]
        MRTN_abs = np.abs(MRTN)
        MRTN_2 = MRTN ** 2

        mX = np.zeros(shape=(1777, 5))
        mX[:, 0] = np.ones(1777)
        mX[:, 1] = MRTN[1:]
        mX[:, 2] = MRTN_abs[1:]
        mX[:, 3] = MRTN_2[1:]
        mX[:, 4] = CSAD[:1777]
        vY = CSAD[1:]
        return vY.values, mX

    
    return data

if __name__ == '__main__':
    # Test the load function
    data = load(start_date='2015-01-05', end_date='2022-04-29')
    print(data.head())
    print(data.tail())
