from PyTimeVar.datasets.utils import load_csv
from pandas import to_datetime
import numpy as np
import pandas as pd

def load(start_date=None, end_date=None, currencies=None):
    """
    Load the gold dataset and optionally filter by specific date range and/or currencies.
    This dataset contains the gold price data from Dec 29 1978 to Sept 6 2024, for a range of currencies.

    Parameters
    ----------
    start_date : str, optional
        The start date to filter the data. 
        Format 'YYYY-MM-DD'.
        Minimum start date is 1978-12-29.
    end_date : str, optional
        The end date to filter the data.
        Format 'YYYY-MM-DD'.
        Maximum end date is 2024-09-06.
    currencies : list, optional
        Currencies to be selected from data. Available options are:
            USD, EUR, JPY, GBP, CAD, CHF, INR, CNY, TRY, SAR, IDR, AED, THB, VND,
            EGP, KRW, RUB, ZAR, AUD


    Returns
    -------
    pandas.DataFrame
        DataFrame containing the filtered data with columns 'Date' and currencies.

    Warnings
    --------
    Prints warnings if any provided currencies are not found in the dataset.
    Prints warnings if the start_date is earlier than the minimum date in the data or the end_date is later than the maximum date in the data.
    """
    data = load_csv(__file__, "gold.csv", sep=",")
    # print(data)

    # Convert Date column to datetime
    data["Date"] = to_datetime(data["Date"], dayfirst=False)

    # Replace commas and convert #N/A to NaN, then convert to float
    for column in data.columns:
        if column != "Date":
            data[column] = data[column].replace("#N/A", np.nan)
            # data[column] = (
            #     data[column].str.replace(".", "").str.replace(",", ".").astype(float)
            # )
            
    min_date = data["Date"].min()
    max_date = data["Date"].max()
    
    data.set_index("Date", inplace=True)

    available_columns = data.columns
    valid_currencies = [col for col in available_columns if col != "Date"]

    if currencies:
        selected_columns = []#["Date"]
        invalid_currencies = []

        for currency in currencies:
            if currency in available_columns:
                selected_columns.append(currency)
            else:
                invalid_currencies.append(currency)

        if invalid_currencies:
            print(
                f"Warning: The following currencies are not available in the dataset and will be ignored: {invalid_currencies}"
            )

        data = data[selected_columns]

    if start_date is not None:
        start_date = to_datetime(start_date)
        if start_date < min_date:
            print(
                f"Warning: The provided start_date {start_date.date()} is earlier than the minimum date in the dataset {min_date.date()}. Adjusting to {min_date.date()}."
            )
            start_date = min_date
        data = data[data.index >= start_date]
    else:
        start_date = min_date

    if end_date is not None:
        end_date = to_datetime(end_date)
        if end_date > max_date:
            print(
                f"Warning: The provided end_date {end_date.date()} is later than the maximum date in the dataset {max_date.date()}. Adjusting to {max_date.date()}."
            )
            end_date = max_date
        data = data[data.index <= end_date]
    else:
        end_date = max_date

    # Drop duplicate dates
    # data = data.drop_duplicates(subset="Date")

    # # Create a complete date range from start_date to end_date
    # all_dates = pd.date_range(start=start_date, end=end_date)

    # Set 'Date' column as the index
    # data.set_index("Date", inplace=True)

    # # Reindex the DataFrame to include all dates, filling missing dates with NaN
    # data = data.reindex(all_dates)

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test the load function
    data = load()
    # print(data["USD"].head())
    # print(data["USD"].tail())
