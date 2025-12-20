import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
import os
import warnings

# Dictionary containing file configurations for data loading
# Each entry specifies file path, delimiter, header row, date columns, and columns to use
dic_files = {
    'file01': {
        'filePath': "./database/",
        'fileName': "IBOV.csv",
        'saveFilePath': "./filled/",
        'saveFileName': "IBOV_filled.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'cols': ["Close"]
    },
    'file02': {
        'filePath': "./database/",
        'fileName': "NASDAQ.csv",
        'saveFilePath': "./filled/",
        'saveFileName': "NASDAQ_filled.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'cols': ["Close"]
    },
    'file03': {
        'filePath': "./database/",
        'fileName': "S&P 500.csv",
        'saveFilePath': "./filled/",
        'saveFileName': "SnP500_filled.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'cols': ["Close"]
    },
    'file04': {
        'filePath': "./database/",
        'fileName': "SMLL.csv",
        'saveFilePath': "./filled/",
        'saveFileName': "SMLL_filled.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'cols': ["Close"]
    },
    'file05': {
        'filePath': "./database/",
        'fileName': "T-BOND10.csv",
        'saveFilePath': "./filled/",
        'saveFileName': "T-BOND10_filled.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'cols': ["Close"]
    },
    'file06': {
        'filePath': "./database/",
        'fileName': "DOLOF.csv",
        'saveFilePath': "./filled/",
        'saveFileName': "DOLARF_filled.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'cols': ["Close"]
    }
} # end of dic_files

for key, config in dic_files.items():

    # check all necessary keys are present
    required_keys = ['filePath', 'fileName', 'saveFilePath', 'saveFileName', 'delimiter', 'decimal', 'parse_dates', 'date_format', 'cols']
    if not all(k in config for k in required_keys):
        print(f"Configuration for {key} is missing required keys.")
        continue

    # header = 0
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

            # df.set_index('Data', inplace=True)
            # series = df['Close'].asfreq('B')
            series = df[col]
        except KeyError as e:
            print(f"reading file {file_path} on column {col}. Does the column exist?\nError detail: {e}")
            continue
        except Exception as e:
            print(f">>> Error reading file {file_path}: {e}")
            continue

        # Fit local level model (random walk + noise)
        model = UnobservedComponents(series, level='local level', trend=False, seasonal=None)

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
