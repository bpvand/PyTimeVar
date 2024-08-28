from PyTimeVar.datasets.utils import load_csv
import pandas as pd
import numpy as np

def load(start_date=None, end_date=None):
    """
    Load the CO2 dataset and optionally filter by date range.
    This dataset contains the Long-run CO₂ concentration on the world from 803719 B.C. till now.
    NOTE: This dataset contains a lot of missing values.
    
    Parameters
    ----------
    start_date : str, optional
        The start_year to filter the data. Format 'YYYY'. Minimum start year is 1900.
    end_date : str, optional
        The end_year to filter the data. Format 'YYYY'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the filtered data with columns 'Year' and 'Long-run CO₂ concentration'.

    Warnings
    --------
    Prints warnings if any provided regions are not found in the dataset.
    Prints warnings if the start_year is earlier than the minimum year in the data or the end_year is later than the maximum year in the data.
    """
    data = load_csv(__file__, "co2.csv", sep=",")
    
    # Convert the 'Year' column to numeric format for filtering
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    
    # Ensure the minimum start year is 1900
    if start_date is None or int(start_date) < 1900:
        start_date = 1900
    if end_date is None:
        end_date = pd.Timestamp.now().year
    if start_date is not None:
        start_date = int(start_date)
        if start_date < data['Year'].min():
            print(f"Warning: The start_date {start_date} is earlier than the minimum year in the data {data['Year'].min()}")
        data = data[data['Year'] >= start_date]
    
    if end_date is not None:
        end_date = int(end_date)
        if end_date > data['Year'].max():
            print(f"Warning: The end_date {end_date} is later than the maximum year in the data {data['Year'].max()}")
        data = data[data['Year'] <= end_date]
    
    # Create a complete range of years from start_date to end_date
    years = pd.DataFrame({'Year': range(start_date, end_date + 1)})
    
    # Merge the complete range of years with the data to ensure all years are present
    data = pd.merge(years, data, how='left', on='Year')
    
    # Select only the 'Year' and 'Long-run CO₂ concentration' columns
    data = data[['Year', 'Long-run CO₂ concentration']]
    
    # Set 'Year' as the index
    data.set_index('Year', inplace=True)
    
    # Print where the NaN values are and how many
    nan_locations = data[data['Long-run CO₂ concentration'].isna()]
    print(f"NaN values are located at the following years:\n{nan_locations.index.values}")
    print(f"Total number of NaN values: {nan_locations.shape[0]}")
    
    return data

if __name__ == "__main__":
    # Test the load function
    data = load(start_date='1900', end_date='2024')
    print(data.head())
    print(data.tail())
