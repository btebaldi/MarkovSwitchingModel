import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import os



dic_files = {
    'file01': {
        'filePath': "./filled/",
        'fileName': "IBOV_filled.csv",
        'saveFilePath': "./filled/",
        'saveFileName': "IBOV_filled.csv",
        'delimiter': ",",
        'decimal': ".",
        'parse_dates': ["Data"],
        'date_format': "%Y-%m-%d",
        'index': "Data",
        'cols': {
            'variavel' : 'Close_filled',
            'k_regimes': 3, # number of regimes
            # Array of exogenous regressors, shaped nobs x k. If None, no exogenous regressors are included.
            'order': 4, # Order of the model describes the dependence of the likelihood on previous regimes.
            'trend': 'ct', # trend{‘n’, ‘c’, ‘t’, ‘ct’} none ='n'; intercept='c' (default);trend = 't'; both = 'ct'.
            'switching_trend': True, # 'switching_trend': True / False / iterable; whether or not all trend coefficients are switching across regimes. If iterable: each element is a boolean describing whether the corresponding coefficient is switching.
            'switching_ar': True, # whether or not the AR coefficients are switching across regimes.
            'switching_variance': False # whether or not the error variance is switching across regimes.
            }
    }
} # end of dic_files


config = dic_files['file01']


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
except pd.errors.EmptyDataError:
    print(f"Error: File is empty: {file_path}")
except Exception as e:
    print(f"Error reading file {file_path}: {e}")





df.set_index(config['index'], inplace=True)
df = df.asfreq('B')

print(f"Data read successfully. DataFrame shape: {df.shape}")
print(df.describe(include='all'))


# Define the model
# Using Hamilton's specification: 2 regimes, AR(4) process, only the mean (intercept) switches
mod_MKS = sm.tsa.MarkovRegression(
    df.loc['2018-01-12':]['Close_filled']/100000,  # Time series data
    k_regimes = 2,               # Number of regimes
    exog=df.loc['2018-01-11':]['Close_filled'][1:]/100000,
    trend = 'c',                 # Trend specification
    switching_trend = True,     # Whether trend coefficients switch across regimes
    # switching_exog  = True,
    switching_variance = False    # Variance is the same across regimes
)

# Fit the model using maximum likelihood estimation
# The .fit() method automatically finds good starting parameters via the EM algorithm
# MKS_result = mod_MKS.fit(disp=True, maxiter=1000, full_output=True, method='basinhopping', transformed=False,
#                          cov_type='none',callback=None)
MKS_result = mod_MKS.fit()




# try:
#     MKS_result = mod_MKS.fit(disp=True)
#     # MKS_result.summary()
# except np.linalg.LinAlgError as e:
#     print(f"Linear algebra error fitting Markov Switching Model to file {file_path}: {e}")
# except Exception as e:
#     print(f"Error fitting Markov Switching Model to file {file_path}: {e}")