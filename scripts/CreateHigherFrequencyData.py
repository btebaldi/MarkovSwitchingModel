
# libraries
"""
Financial Data Processing and Aggregation Module
This module processes financial time series data from CSV files, performing daily-to-weekly
and daily-to-monthly aggregations. It supports multiple stock indices and currency pairs
with configurable file paths, delimiters, and aggregation methods.
Main Workflow:
    1. Read daily financial data from CSV files
    2. Set business day frequency ('B') for the daily data
    3. Aggregate daily data to weekly frequency (using last closing price)
    4. Aggregate daily data to monthly frequency (using last closing price)
    5. Save processed weekly and monthly data to output CSV files
Configuration Structure (dic_files):
    Each file entry contains:
    - filePath (str): Directory path containing the input CSV file
    - fileName (str): Name of the input CSV file
    - saveFilePath (str): Directory path for saving output files (subdirectory 'weekly' and 'monthly')
    - saveFileName (str): Base name for output files
    - delimiter (str): CSV delimiter character (e.g., ',')
    - decimal (str): Decimal separator character (e.g., '.')
    - parse_dates (list): Column names to parse as datetime
    - date_format (str): Format string for date parsing (e.g., '%Y-%m-%d')
    - index (str): Column name to use as DataFrame index
    - aggregation (dict): Dictionary mapping column names to aggregation functions (e.g., 'last', 'sum')
Functions:
    read_daily_database(config: dict) -> pd.DataFrame:
        Reads daily financial data from CSV and sets business day frequency
    create_weekly_database(daily_df: pd.DataFrame, aggregation: dict) -> pd.DataFrame:
        Aggregates daily data to weekly frequency using specified aggregation functions
    create_monthly_database(daily_df: pd.DataFrame, aggregation: dict) -> pd.DataFrame:
        Aggregates daily data to monthly frequency using specified aggregation functions
    save_database(df: pd.DataFrame, filepath: str) -> None:
        Writes processed DataFrame to CSV file at specified path
"""
import pandas as pd
import os
import json

def Load_configuration(file_path : str) -> dict:
    """Load configuration from a JSON file."""
    if file_path is None:
        file_path = "./conf/HigherFrequencyConfiguration.json"

    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in configuration file: {file_path}")
    
    return config

def Save_configuration(config : dict, file_path : str):
    """Save configuration to a JSON file."""
    if file_path is None:
        file_path = "./conf/HigherFrequencyConfiguration.json"

    with open(file_path, "w") as outfile:
        json.dump(config, outfile, indent=4)
    
def read_daily_database(config : dict) -> pd.DataFrame:
    """Read daily database from CSV file based on configuration dictionary"""
    
    # Create full file path by joining directory and filename
    file_path = os.path.join(config['filePath'], config['fileName'])
    
    print(f"Processing file: {file_path}")
    
    # reading data
    df = pd.read_csv(file_path,
                        delimiter=config['delimiter'],
                        decimal=config['decimal'],
                        header=0,
                        parse_dates=config['parse_dates'],
                        date_format=config['date_format'])

    print(f"Data read successfully. DataFrame shape: {df.shape}")

    # Define index and frequency of the DataFrame.
    # Current files are assumed to have a Dayly Frequency. This can be changedate column named 'Data'.
    df.set_index(config['index'], inplace=True)
    
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    df = df.asfreq('B')

    return df

def create_weekly_database(daily_df : pd.DataFrame, aggregation : dict) -> pd.DataFrame:
    """Aggregate daily data to weekly frequency"""

    print("Creating weekly database.")
    weekly_df = daily_df.resample('W').agg(aggregation)
    return weekly_df

def create_monthly_database(daily_df : pd.DataFrame, aggregation : dict) -> pd.DataFrame:
    """Aggregate daily data to monthly frequency"""

    print("Creating monthly database.")
    monthly_df = daily_df.resample('M').agg(aggregation)
    return monthly_df

def save_database(df, filepath):
    """Save processed database to CSV"""

    # Extract directory from filepath
    directory = os.path.dirname(filepath)

    # Check if directory exists, create if necessary
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    df.to_csv(filepath)
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    # Read daily data
    dic_files =  Load_configuration(file_path=None)

    for key, config in dic_files.items( ) :
        print(f"Processing configuration for {key}...")
    
        # check all necessary keys are present
        required_keys = ['filePath', 'fileName',
                         'saveFilePath', 'saveFileName',
                         'delimiter', 'decimal', 'parse_dates',
                        'date_format', 'index', 'aggregation']
        
        if not all(k in config for k in required_keys):
            print(f"Configuration for {key} is missing required keys.")
            raise ValueError(f"Missing required keys.")

        # read daily data
        daily_data = read_daily_database(config)
        
        # Create aggregations
        weekly_data = create_weekly_database(daily_data, aggregation = config['aggregation'])
        monthly_data = create_monthly_database(daily_data, aggregation = config['aggregation'])
        
        # Check if save directory exists, create if not
        if not os.path.exists(config['saveFilePath']):
            os.makedirs(config['saveFilePath'])
        
        # Save results
        save_database(weekly_data, os.path.join(config['saveFilePath'], "weekly/", config["saveFileName"]) )
        save_database(monthly_data, os.path.join(config['saveFilePath'], "monthly/", config["saveFileName"]) )