from PyTimeVar.datasets.utils import load_csv
from pandas import to_datetime

def load(start_date=None, end_date=None, regions=None):
    """
    Load the temperature dataset and optionally filter by by date range and/or regions.
    This dataset contains the average yearly temperature change in degrees Celsius for different regions of the world from 1961 to 2023.
    
    Parameters
    ----------
    start_date : str, optional
        The start year to filter the data. 
        Format 'YYYY'. 
        Minimum start year is 1961.
    end_date : str, optional
        The end year to filter the data.
        Format 'YYYY'. 
        Maximum end year is 2023.
    regions : list, optional
        List of regions to filter the dataset by. Available options are:
            World, Africa, Asia, Europe, North America, Oceania, South America

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the filtered data with columns 'Date' and regions.

    Warnings
    --------
    Prints warnings if any provided regions are not found in the dataset.
    Prints warnings if the start_date is earlier than the minimum year in the data or the end_date is later than the maximum year in the data.
    """
    data = load_csv(__file__, "temperature.csv", sep=",")
    data.rename(columns={'Year': 'Date'}, inplace=True)
    # Convert Date column to datetime
    data["Date"] = to_datetime(data["Date"], format="%Y")
    # Set Date as index
    data.set_index("Date", inplace=True)
    # Replace commas and convert #N/A to NaN, then convert to float
    for column in data.columns:
        data[column] = data[column].astype(float)

    min_date = data.index.min()
    max_date = data.index.max()

    available_columns = data.columns
    if regions:
        selected_columns = regions
        invalid_regions = [region for region in regions if region not in available_columns]

        if invalid_regions:
            print(
                f"Warning: The following regions are not available in the dataset and will be ignored: {invalid_regions}"
            )

        data = data[selected_columns]

    if start_date:
        start_date = to_datetime(start_date)
        if start_date < min_date:
            print(
                f"Warning: The provided start_date {start_date.date()} is earlier than the minimum date in the dataset {min_date.date()}. Adjusting to {min_date.date()}."
            )
            start_date = min_date
        data = data[data.index >= start_date]

    if end_date:
        end_date = to_datetime(end_date)
        if end_date > max_date:
            print(
                f"Warning: The provided end_date {end_date.date()} is later than the maximum date in the dataset {max_date.date()}. Adjusting to {max_date.date()}."
            )
            end_date = max_date
        data = data[data.index <= end_date]

    return data

if __name__ == "__main__":
    # Test the load function
    data = load()
    print(data.head())
