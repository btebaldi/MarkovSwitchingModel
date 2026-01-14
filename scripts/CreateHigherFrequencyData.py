"""
Financial Data Processing and Aggregation Module
===============================================

Purpose
-------
This module reads **daily** financial time series from CSV files and produces
**weekly** and **monthly** datasets using configurable aggregation rules.

Typical use cases:
- Converting daily market data (prices, volumes, rates) into lower-frequency panels
- Creating consistent weekly/monthly datasets for econometric or forecasting models
- Standardizing (z-scoring) selected columns to ease cross-series comparison

High-level workflow
-------------------
For each configuration entry in the JSON file:
1) Read daily financial data from a CSV file
2) Set the DataFrame to a business-day frequency ('B')
3) Aggregate from daily to weekly frequency (e.g., last close per week)
4) Aggregate from daily to monthly frequency (e.g., last close per month)
5) Optionally standardize selected columns (add *_std columns)
6) Save the resulting weekly and monthly datasets to CSV

Configuration structure
-----------------------
The JSON configuration (dic_files) is expected to map a key (dataset name) to a dict
containing at least:

- filePath (str): directory containing the input CSV
- fileName (str): input CSV filename
- saveFilePath (str): base output directory (weekly/monthly subfolders are used)
- saveFileName (str): output filename (same used for weekly and monthly)
- delimiter (str): CSV delimiter (e.g. ',', ';', '\t')
- decimal (str): decimal separator (e.g. '.' or ',')
- parse_dates (list): columns to parse as datetime (commonly the date column)
- date_format (str): expected date string format (e.g. '%Y-%m-%d')
- index (str): column name to set as the DataFrame index (must be datetime-like)
- aggregation (dict): mapping of column -> aggregation function, e.g.
    {
        "Close": "last",
        "Volume": "sum"
    }

Important assumptions and notes
-------------------------------
- The daily data is *assumed* to be a time series with one row per day (or most days).
- After loading, the index is set to business-day frequency via .asfreq('B').
  This introduces business-day rows for missing dates and fills them with NaN.
  That is often desirable if downstream steps expect an explicit business-day grid.
- Weekly aggregation uses `.resample('W')` which (by default) labels weeks ending on Sunday.
  If you want weeks ending on Friday (common in finance), consider 'W-FRI'.
- Monthly aggregation uses `.resample('ME')` meaning "month end".
"""

import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler


def LoadConfiguration(file_path: str | None) -> dict:
    """
    Load a JSON configuration file.

    Parameters
    ----------
    file_path : str
        Path to the JSON configuration file. If None, a default path is used.

    Returns
    -------
    dict
        Parsed JSON content. Expected to be a dict where each key is a dataset name
        and each value is a configuration dictionary for that dataset.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the JSON is invalid (cannot be parsed).
    """

    # Default configuration path if none is supplied
    if file_path is None:
        file_path = "./conf/HigherFrequencyConfiguration.json"

    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        # Explicit message helps debugging when running from different working dirs
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError:
        # Raised when JSON is malformed (trailing commas, missing braces, etc.)
        raise ValueError(f"Invalid JSON format in configuration file: {file_path}")

    return config


def SaveConfiguration(config: dict, file_path: str):
    """
    Save a configuration dictionary to a JSON file.

    Parameters
    ----------
    config : dict
        Configuration dictionary to serialize as JSON.
    file_path : str
        Output JSON path. If None, a default path is used.

    Notes
    -----
    - Uses json.dump(..., indent=4) for readable formatting.
    """
    if file_path is None:
        file_path = "./conf/HigherFrequencyConfiguration.json"

    with open(file_path, "w") as outfile:
        json.dump(config, outfile, indent=4)


def ReadDailyDatabase(config: dict) -> pd.DataFrame:
    """
    Read the daily dataset specified by a configuration entry and enforce a business-day index.

    Parameters
    ----------
    config : dict
        Configuration dictionary for one dataset. Must include:
        filePath, fileName, delimiter, decimal, parse_dates, date_format, index.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by the configured date column, aligned to business-day frequency.

    Key behavior
    ------------
    - Reads CSV using parsing parameters from config.
    - Sets the specified column as index (must be datetime-like).
    - Forces business-day frequency with `.asfreq('B')`:
      any missing business days are inserted with NaN values.
    """
    # Build full input file path: <filePath>/<fileName>
    file_path = os.path.join(config['filePath'], config['fileName'])

    print(f"Processing file: {file_path}")

    # Load CSV with consistent parsing rules (delimiter/decimal/date handling)
    df = pd.read_csv(
        file_path,
        delimiter=config['delimiter'],
        decimal=config['decimal'],
        header=0,
        parse_dates=config['parse_dates'],
        date_format=config['date_format']
    )

    print(f"Data read successfully. DataFrame shape: {df.shape}")

    # Use a date column (provided in config['index']) as the DataFrame index.
    # This is required for resampling operations.
    df.set_index(config['index'], inplace=True)

    # Enforce business-day grid ('B'):
    # - Ensures the index is evenly spaced in business days.
    # - Missing business dates become explicit rows with NaN values.
    # See: pandas offset aliases documentation.
    df = df.asfreq('B')

    return df


def CreateWeeklyDatabase(daily_df: pd.DataFrame, aggregation: dict) -> pd.DataFrame:
    """
    Aggregate daily data into weekly data using the provided aggregation mapping.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily time series indexed by datetime.
    aggregation : dict
        Dictionary mapping column names to aggregation functions (e.g., "last", "sum").

    Returns
    -------
    pd.DataFrame
        Weekly aggregated DataFrame.

    Notes
    -----
    - Uses `.resample('W')` which groups data into weeks ending on Sunday by default.
      For finance-style weeks (ending Friday), 'W-FRI' is often preferred.
    """
    print("Creating weekly database.")
    weekly_df = daily_df.resample('W').agg(aggregation)
    return weekly_df


def CreateMonthlyDatabase(daily_df: pd.DataFrame, aggregation: dict) -> pd.DataFrame:
    """
    Aggregate daily data into monthly data using the provided aggregation mapping.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily time series indexed by datetime.
    aggregation : dict
        Dictionary mapping column names to aggregation functions.

    Returns
    -------
    pd.DataFrame
        Monthly aggregated DataFrame.

    Notes
    -----
    - Uses `.resample('ME')` meaning month-end frequency.
    """
    print("Creating monthly database.")
    monthly_df = daily_df.resample('ME').agg(aggregation)
    return monthly_df


def SaveDatabase(df, filepath):
    """
    Save a DataFrame to CSV, creating parent directories if needed.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to write to disk.
    filepath : str
        Full output path including filename.

    Notes
    -----
    - Ensures the output directory exists (mkdir -p behavior).
    - Uses pandas `.to_csv()` with default settings (includes index by default).
      If you do NOT want the index column in the output, pass index=False in the call.
    """
    # Directory portion of the path (everything except filename)
    directory = os.path.dirname(filepath)

    # Create directory if it doesn't exist (safe for nested paths)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Persist CSV to disk
    df.to_csv(filepath)
    print(f"Saved: {filepath}")


def DataPadronization(df: pd.DataFrame, target_column: list[str], sufix: str = "_std") -> pd.DataFrame:
    """
    Standardize selected columns (z-score) and append standardized versions as new columns.

    Standardization
    ---------------
    For each column x, StandardScaler computes:
        x_std = (x - mean(x)) / std(x)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to be modified in place (new columns are added).
    target_column : list[str]
        List of columns to standardize.
    sufix : str, default "_std"
        Suffix appended to each standardized column name.

    Returns
    -------
    pd.DataFrame
        The same DataFrame object, now containing additional standardized columns.

    Notes
    -----
    - StandardScaler expects a 2D array; df[[col]] preserves a DataFrame shape.
    - `.ravel()` flattens the result into a 1D array to fit into a single DataFrame column.
    - If a column contains NaNs, StandardScaler will raise unless handled upstream.
    """
    for col in target_column:
        scaler = StandardScaler()
        sufix_col = col + sufix

        # Fit on the column and transform to standardized values (as 1D array)
        df[sufix_col] = scaler.fit_transform(df[[col]]).ravel()

    return df


if __name__ == "__main__":
    # Load the configuration dict. Each entry describes one dataset to process.
    dic_files = LoadConfiguration(file_path=None)

    # Loop over datasets defined in the configuration file
    for key, config in dic_files.items():
        print(f"Processing configuration for {key}...")

        # Defensive validation: ensure config includes all required keys
        required_keys = [
            'filePath', 'fileName',
            'saveFilePath', 'saveFileName',
            'delimiter', 'decimal', 'parse_dates',
            'date_format', 'index', 'aggregation'
        ]

        if not all(k in config for k in required_keys):
            print(f"Configuration for {key} is missing required keys.")
            raise ValueError("Missing required keys.")

        # Load daily time series and align to business-day index
        daily_data = ReadDailyDatabase(config)

        # Aggregate daily -> weekly/monthly using the same aggregation map
        weekly_data = CreateWeeklyDatabase(daily_data, aggregation=config['aggregation'])
        monthly_data = CreateMonthlyDatabase(daily_data, aggregation=config['aggregation'])

        # Add standardized versions of the aggregated columns
        # weekly: explicitly converts keys to list
        weekly_data = DataPadronization(
            df=weekly_data,
            target_column=list(config['aggregation'].keys()),
            sufix="_std"
        )

        # monthly: config['aggregation'].keys() is acceptable as an iterable
        monthly_data = DataPadronization(
            df=monthly_data,
            target_column=config['aggregation'].keys(),
            sufix="_std"
        )

        # Ensure base output directory exists
        if not os.path.exists(config['saveFilePath']):
            os.makedirs(config['saveFilePath'])

        # Save results into weekly/ and monthly/ subfolders under saveFilePath
        SaveDatabase(weekly_data, os.path.join(config['saveFilePath'], "weekly/", config["saveFileName"]))
        SaveDatabase(monthly_data, os.path.join(config['saveFilePath'], "monthly/", config["saveFileName"]))
