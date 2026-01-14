"""
Apply Kalman Filter to Time Series Data
=====================================

This script applies a **local level Kalman Filter** (state-space model) to one or more
time-series columns in CSV files in order to:

- Smooth noisy observations
- Filter signals in real time
- Impute missing values using state estimates

The workflow is configuration-driven via a JSON file, allowing batch processing of
multiple datasets with different parsing rules.

Key Outputs per series:
- Smoothed level
- Filtered level
- Fitted values
- Residuals
- Filled series (missing values replaced by smoothed estimates)

No modeling assumptions beyond a random-walk level with Gaussian noise are imposed.
"""

import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
import os
import warnings
import json
from typing import Optional

def LoadConfiguration(file_path : Optional[str]) -> dict:
    """
    Load configuration from a JSON file.
    
    Parameters:
    -----------
    file_path : Optional[str]
        Path to JSON configuration file. If None, uses default path.
    
    Returns:
    --------
    dict
        Configuration dictionary with file paths, parsing options, and column specs.
    
    Raises:
    -------
    FileNotFoundError
        If configuration file does not exist.
    ValueError
        If JSON format is invalid.
    """
    if file_path is None:
        file_path = "./conf/ApplyKalmanFilterConfiguration.json"

    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in configuration file: {file_path}")
    
    return config


if __name__ == "__main__":
    # Load all dataset configurations from JSON file
    dic_files = LoadConfiguration(None)

    # Iterate over each dataset configuration
    for key, config in dic_files.items():

        # check all necessary keys are present
        required_keys = ['filePath', 'fileName', 'saveFilePath', 'saveFileName', 'delimiter', 'decimal', 'parse_dates', 'date_format', 'cols']
        if not all(k in config for k in required_keys):
            print(f"Configuration for {key} is missing required keys.")
            continue

        # Extract column names to process
        cols = config['cols']

        # Create full file path by joining directory and filename
        file_path = os.path.join(config['filePath'], config['fileName'])
        
        print(f"Processing file: {file_path}")

        # reading data
        try:
            df = pd.read_csv(file_path,
                                delimiter=config['delimiter'],
                                decimal=config['decimal'],
                                header=0,
                                parse_dates=config['parse_dates'],
                                date_format=config['date_format'])
        except FileNotFoundError:
            print(f">>> Error: File not found: {file_path}")
            continue
        except pd.errors.EmptyDataError:
            print(f">>> Error: File is empty: {file_path}")
            continue
        except Exception as e:
            print(f">>> Error reading file {file_path}: {e}")
            continue


        # For each column provided, we apply a KF to fill missing data.
        for col in config['cols']:
            
            # Parse dates and set index
            try:
                series = df[col]
            except KeyError as e:
                print(f"reading file {file_path} on column {col}. Does the column exist?\nError detail: {e}")
                continue
            except Exception as e:
                print(f">>> Error reading file {file_path}: {e}")
                continue

            # STATE-SPACE MODEL: Fit local level model (random walk + noise)
            # ================================================================
            # Local level model: y_t = level_t + noise_t
            #                    level_t = level_{t-1} + innovation_t
            model = UnobservedComponents(series, level='local level', trend=False, seasonal=None)

            # Fit model with warnings suppressed (MLE can produce convergence notices)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # code with warnings
                result = model.fit()

            # Save result to new column
            df[f"{col}_level_Smoothed"] =  result.level.smoothed
            df[f"{col}_level_filtered"] =  result.level.filtered
            df[f"{col}_fittedvalues"] =  result.fittedvalues
            df[f"{col}_resid"] =  result.resid

            # Create new series with missing values filled by smoothed values
            filled_series = series.copy()
            filled_series[series.isna()] = result.level.smoothed[series.isna()]
            df[f"{col}_filled"] =  filled_series

        # Check if save directory exists, create if not
        if not os.path.exists(config['saveFilePath']):
            os.makedirs(config['saveFilePath'])

        # Save the filled DataFrame to a new CSV file   
        save_path = os.path.join(config['saveFilePath'], config['saveFileName'])
        try:
            df.to_csv(save_path, index=False)
            print(f"Saved: {save_path}")
        except PermissionError:
            print(f"Permission denied saving file: {save_path}")
        except OSError as e:
            print(f"OS error saving file {save_path}: {e}")
        except Exception as e:
            print(f"Unexpected error saving file {save_path}: {e}")
