import pandas as pd
import statsmodels.api as sm
import numpy as np
import os


dic_files = {
    'file01': {
        'filePath': "./filled/",
        'fileName': "IBOV_filled.csv",
        'saveFilePath': "./output/",
        'saveFileName': "IBOV_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'file02': {
        'filePath': "./filled/",
        'fileName': "SMLL_filled.csv",
        'saveFilePath': "./output/",
        'saveFileName': "SMLL_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'file03': {
        'filePath': "./filled/",
        'fileName': "SnP500_filled.csv",
        'saveFilePath': "./output/",
        'saveFileName': "SnP500_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'file04': {
        'filePath': "./filled/",
        'fileName': "NASDAQ_filled.csv",
        'saveFilePath': "./output/",
        'saveFileName': "NASDAQ_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'file05': {
        'filePath': "./filled/",
        'fileName': "DOLARF_filled.csv",
        'saveFilePath': "./output/",
        'saveFileName': "DOLARF_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
# ----------------------------
    'Mfile01': {
        'filePath': "./output/",
        'fileName': "monthly_IBOV_output.csv",
        'saveFilePath': "./output/",
        'saveFileName': "M_IBOV_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'Mfile02': {
        'filePath': "./output/",
        'fileName': "monthly_SMLL_output.csv",
        'saveFilePath': "./output/",
        'saveFileName': "M_SMLL_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'Mfile03': {
        'filePath': "./output/",
        'fileName': "monthly_SnP500_output.csv",
        'saveFilePath': "./output/",
        'saveFileName': "M_SnP500_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'Mfile04': {
        'filePath': "./output/",
        'fileName': "monthly_NASDAQ_output.csv",
        'saveFilePath': "./output/",
        'saveFileName': "M_NASDAQ_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'Mfile05': {
        'filePath': "./output/",
        'fileName': "monthly_DOLARF_output.csv",
        'saveFilePath': "./output/",
        'saveFileName': "M_DOLARF_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
# ----------------------------
    'Wfile01': {
        'filePath': "./output/",
        'fileName': "weekly_IBOV_output.csv",
        'saveFilePath': "./output/",
        'saveFileName': "W_IBOV_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'Wfile02': {
        'filePath': "./output/",
        'fileName': "weekly_SMLL_output.csv",
        'saveFilePath': "./output/",
        'saveFileName': "W_SMLL_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'Wfile03': {
        'filePath': "./output/",
        'fileName': "weekly_SnP500_output.csv",
        'saveFilePath': "./output/",
        'saveFileName': "W_SnP500_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'Wfile04': {
        'filePath': "./output/",
        'fileName': "weekly_NASDAQ_output.csv",
        'saveFilePath': "./output/",
        'saveFileName': "W_NASDAQ_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    },
    'Wfile05': {
        'filePath': "./output/",
        'fileName': "weekly_DOLARF_output.csv",
        'saveFilePath': "./output/",
        'saveFileName': "W_DOLARF_output.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 2, # number of regimes
            'trend': 'c', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    }

} # end of dic_files

for key, config in dic_files.items( ) :

    # check all necessary keys are present
    required_keys = ['filePath', 'fileName', 'saveFilePath', 'saveFileName', 'delimiter', 'decimal', 'parse_dates',
                      'date_format', 'index', 'cols']
    if not all(k in config for k in required_keys):
        print(f"Configuration for {key} is missing required keys.")
        continue

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
        print(f"Error: File not found: {file_path}")
        continue
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty: {file_path}")
        continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        continue

    print(f"Data read successfully. DataFrame shape: {df.shape}")
    # print(df.describe(include='all'))


    # Define index and frequency of the DataFrame.
    # Current files are assumed to have a Dayly Frequency. This can be changedate column named 'Data'.
    df.set_index(config['index'], inplace=True)
    
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    # df = df.asfreq('B')

    print(f"Data read successfully. DataFrame shape: {df.shape}")
    # print(df.describe(include='all'))

    df['log_return'] = np.log(df['Close_filled'] / df['Close_filled'].shift(1))



    # Define the model. We use a model with 2 regimes on the differenced series.
    # The regimes will have different intercept and different variance. 
    mod_MKS = sm.tsa.MarkovRegression(
        df['log_return'].dropna(),  # Time series data
        k_regimes = 2,                       # Number of regimes
        trend = 'c',                         # Trend specification
        switching_trend = True,              # Whether trend coefficients switch across regimes
        switching_variance = True            # Variance is the same across regimes
    )

    # Fit the model using maximum likelihood estimation
    # The .fit() method automatically finds good starting parameters via the EM algorithm
    # MKS_result : sm.tsa.markov_switching.markov_regression.MarkovRegressionResults = None
    try:
        MKS_result = mod_MKS.fit()
        # MKS_result.summary()
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error fitting Markov Switching Model to file {file_path}: {e}")
    except Exception as e:
        print(f"Error fitting Markov Switching Model to file {file_path}: {e}")


    # Check if save directory exists, create if not
    if not os.path.exists(config['saveFilePath']):
        os.makedirs(config['saveFilePath'])
    
    # Save regime transitions and probabilities to a text file
    output_file = os.path.join(config['saveFilePath'], config["saveFileName"])
    
    
    with open(output_file, 'w') as f:
        f.write("\n\n=== Model Summary ===\n")
        f.write(MKS_result.summary().as_text())
        f.write("=== Markov Switching Model - Regime Analysis ===\n\n")
        f.write("\n\nRegime Transitions:\n")
        f.write(str(MKS_result.regime_transition))
        f.write("Filtered Regime Probabilities:\n")
        f.write(MKS_result.filtered_marginal_probabilities.to_string())
        f.write("\n\nSmoothed Regime Probabilities:\n")
        f.write(MKS_result.smoothed_marginal_probabilities.to_string())


    print(f"Regime analysis saved to: {output_file}")