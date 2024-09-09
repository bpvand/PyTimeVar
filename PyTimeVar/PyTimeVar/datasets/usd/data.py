from PyTimeVar.datasets.utils import load_csv
from pandas import to_datetime
import pandas as pd

def load(type="Open", start_date=None, end_date=None):
    """
    Load the USD index dataset and optionally filter by date range.

    Parameters
    ----------
    type : str, optional
        The type of data to load. Available options are:
        ['Open', 'High', 'Low', 'Close']
    start_date : str, optional
        The start date to filter the data. Format 'YYYY-MM-DD'.
    end_date : str, optional
        The end date to filter the data. Format 'YYYY-MM-DD'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the filtered data.

    Warnings
    --------
    Prints warnings if any provided currencies are not found in the dataset.
    Prints warnings if the start_date is earlier than the minimum date in the data or the end_date is later than the maximum date in the data.
    """
    # Load the data
    data = load_csv(__file__, "USDIndex.csv", sep=",")
    
    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close']
    
    # Set the 'Date' column as the index
    data.set_index('Date', inplace=True)
    
    # Determine the date range
    min_date = data.index.min()
    max_date = data.index.max()
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        if start_date < min_date:
            print("Warning: start_date is earlier than the minimum date in the data.")
        min_date = max(min_date, start_date)
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        if end_date > max_date:
            print("Warning: end_date is later than the maximum date in the data.")
        max_date = min(max_date, end_date)
    
    # Create a complete date range from min_date to max_date
    all_dates = pd.date_range(start=min_date, end=max_date)
    
    # Reindex the data to include all dates, filling missing dates with NaN
    data = data.reindex(all_dates)
    
    # Select the specified type of data
    if type not in ['Open', 'High', 'Low', 'Close']:
        raise ValueError("Invalid type specified. Available options are: ['Open', 'High', 'Low', 'Close']")
    
    # Return the filtered data with only the specified column
    return data[[type]]

if __name__ == "__main__":
    # Test the load function
    data = load()
    print(data.head())