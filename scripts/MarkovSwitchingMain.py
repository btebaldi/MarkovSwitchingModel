# Load necessary libraries
import numpy as np
import pandas as pd
import MarkovSwitchingModel as MKM
import MarkovSwitchingModelEstimator as MKE
import MarkovSwitchingModelHelper as MKH
from datetime import datetime
import pathlib as plib

class ParameterGuess:
    """
    Class to generate initial guesses for the parameters of a Markov Switching Model.
    """

    def __init__(self) -> None:
        self.Beta = None
        self.Omega = None
        self.TransitionMatrix = None
        self.UnconditionalStateProbability = None

    def GetParameters(self):
        """
        Returns the initial parameter guesses.

        Returns:
            Tuple: A tuple containing Beta, Omega, TransitionMatrix, and UnconditionalStateProbability.
        """
        return self.Beta, self.Omega, self.TransitionMatrix, self.UnconditionalStateProbability

def LoadModel(filePath, 
              variable, Xvariable: list | None = None,
              ar = 1,
              level = False, intercept = True, trend = True,
              regimes = 2,
              decimal='.', delimiter=',', parse_dates=None, date_format="%Y-%m-%d",
              index_col=None, data_ini = None, data_fim = None,
              initial_guess=None) -> MKM.MarkovSwitchingModel:
    """
    Loads and preprocesses data from a CSV file to create a Markov Switching Model.
    
    This function handles data loading, variable transformation (level vs log-returns),
    lag construction for autoregressive modeling, and initial parameter generation.

    Parameters:
        filePath (str): Path to the CSV file containing the data.
        variable (str): Name of the dependent variable column in the CSV.
        ar (int): Autoregressive order (number of lags to include). Default: 1
        level (bool): If True, use variable at level; if False, compute log-returns. Default: False
        intercept (bool): If True, include intercept (constant term) in X. Default: True
        trend (bool): If True, include trend variable in X. Default: True
        regimes (int): Number of regimes in the Markov Switching Model. Default: 2
        decimal (str): Decimal separator in CSV file. Default: '.'
        delimiter (str): Column delimiter in CSV file. Default: ','
        parse_dates (list): List of column names to parse as dates. Default: None
        date_format (str): Format string for date parsing. Default: "%Y-%m-%d"
        index_col (int/str): Column to use as index (datetime index). Default: None
        data_ini (str): Start date for filtering data in format matching date_format. Default: None
        data_fim (str): End date for filtering data in format matching date_format. Default: None

    Returns:
        model (MKM.MarkovSwitchingModel): An instance of the Markov Switching Model initialized with 
                                         preprocessed data and random initial parameters.
    """

    # Initialize parameter names dictionary 
    param_names =  MKM.GetEmptyParamNames()
     
    # Convert filePath to Path object if it's a string
    filePath = plib.Path(filePath)

    # ===== LOAD DATA FROM CSV FILE =====
    # Read the CSV file with specified formatting parameters
    # delimiter: character used to separate columns (comma, semicolon, etc.)
    # decimal: character used as decimal separator (period or comma)
    # header=0: use first row as column names
    # parse_dates: convert specified columns to datetime format
    # date_format: specify the format of date strings for proper parsing
    # index_col: set specified column as the DataFrame index (useful for time series)
    df = pd.read_csv(filePath,
                     delimiter=delimiter,
                     decimal=decimal,
                     header=0,
                     parse_dates=parse_dates,
                     date_format=date_format,
                     index_col=index_col)
    
    # Check if 'Y' column exists and rename it to avoid conflicts
    if ('Y' in df.columns) and (variable == 'Y'):
        df = df.rename(columns={'Y': 'Y_original'})
        variable = 'Y_original'

    # if the data is to be extract ate the level construct the dependent variable directly
    # otherwise, compute log-returns
    if level:
        df['Y'] = df[variable]
        ytrasnsformation = MKM.TypeOfTransformation.LEVEL
    else:
        df['Y'] = np.log(df[variable] / df[variable].shift(1))
        ytrasnsformation = MKM.TypeOfTransformation.LOG_DIFF
        
        #  Remove the first row which contains NaN due to the log-return calculation
        df = df.iloc[1:, :]

    # ===== Construct lags =====
    for lag in range(1, ar + 1):
        df[f"Y_lag_{lag}"] = df["Y"].shift(lag)
    
    # Remove the rows which contains NaN due to the lag operation
    df = df.iloc[ar:, :]

    # ===== Filter the database on the desired range =====
    if data_ini is not None or data_fim is not None:
        if index_col is not None:
            df = df.loc[data_ini: data_fim]
        else:
            raise ValueError("data_ini or data_fim specified but no index column provided. please provide an index column.")
   
    # ===== Extract Y and X from the database =====
    Y = df['Y'].to_numpy().reshape((-1, 1))
    param_names['Y'][0] = MKM.GetDictRepresentation(name = variable, type = MKM.TypeOfVariable.DEPENDENT, classX = None, transformation = ytrasnsformation, ar = None)

    # Create a trend variable (linear time index from 0 to T-1)
    # Note: This variable is currently unused in the model specification below
    data_trend = np.arange(Y.shape[0])
    
    # Create an intercept column (vector of ones) for the constant term in the regression
    data_intercept = np.ones(Y.shape[0])
    
    # add intercept and trend if specified
    X = np.empty((Y.shape[0], 0))  # Initialize X as an empty array

    nColumn_X = 0
    if intercept :
        X = np.column_stack([X, data_intercept])
        param_names['X'][nColumn_X] = MKM.GetDictRepresentation(name = 'Intercept',
                                                        type = MKM.TypeOfVariable.INDEPENDENT,
                                                        classX = MKM.TypeOfDependentVariable.INTERCEPT,
                                                        transformation = MKM.TypeOfTransformation.NONE, ar = None)
        nColumn_X = nColumn_X + 1

    if trend :
        X = np.column_stack([X, data_trend])
        param_names['X'][nColumn_X] = MKM.GetDictRepresentation(name = 'Trend',
                                                        type = MKM.TypeOfVariable.INDEPENDENT,
                                                        classX = MKM.TypeOfDependentVariable.TREND,
                                                        transformation = MKM.TypeOfTransformation.NONE, ar = None)
        nColumn_X = nColumn_X + 1


    if Xvariable is not None:
        for Xvar in Xvariable:
            X = np.column_stack([X, df[Xvar].to_numpy()])
            param_names['X'][nColumn_X] = MKM.GetDictRepresentation(name = Xvar,
                                                            type = MKM.TypeOfVariable.INDEPENDENT,
                                                            classX = MKM.TypeOfDependentVariable.EXOGENOUS,
                                                            transformation = MKM.TypeOfTransformation.NONE, ar = None)
            nColumn_X = nColumn_X + 1

    # Build the independent variable matrix X by horizontally stacking the intercept and lagged dependent variable
    # X has dimensions (T x 2): first column is the constant, second column is lagged Close-Level
    for lag in range(1, ar + 1):
        X = np.column_stack([X, df[f'Y_lag_{lag}'].to_numpy()])
        # param_names['X'][f'Lag_{lag}'] = MKM.TypeOfXVariable.AUTO_REGRESSIVE
        param_names['X'][nColumn_X] = MKM.GetDictRepresentation(name = f"Lag_{lag}",
                                                            type = MKM.TypeOfVariable.INDEPENDENT,
                                                            classX = MKM.TypeOfDependentVariable.AUTO_REGRESSIVE,
                                                            transformation = MKM.TypeOfTransformation.NONE, ar = lag)
        nColumn_X = nColumn_X + 1
    
    if initial_guess is None:
        # ===== Generate initial parameter guesses =====
        initial_guess = ParameterGuess()
    
    model = MKM.MarkovSwitchingModel(Y, X, num_regimes=regimes,
                                    beta=initial_guess.Beta, 
                                     omega=initial_guess.Omega, 
                                     param_names=param_names,
                                     dates_label=df.index,
                                     model_name=filePath.stem)

    return model

if __name__ == "__main__" :
    ParameterGuess_instance = ParameterGuess()
    ParameterGuess_instance.Beta = np.array([[-0.0002, 0.0003,  0.0023],
                                               [0.11, 0.10, -0.08]])
    ParameterGuess_instance.Beta = np.array([[-0.0002, 0.0003,  0.0023],
                                               [0.11, 0.10, -0.08]])
    ParameterGuess_instance.Omega = np.array([[0.04**2, 0.09**2, 0.02**2]])

    list_paths = [".\\database\\filled\\DOLARF_filled.csv",
                  ".\\database\\filled\\IBOV_filled.csv",   
                   ".\\database\\filled\\NASDAQ_filled.csv",
                   ".\\database\\filled\\SnP500_filled.csv",
                   ".\\database\\filled\\SMLL_filled.csv",
                   ".\\database\\filled\\T-BOND10_filled.csv"]
    
    list_paths = [".\\database\\filled\\NASDAQ_filled.csv"]

    for path in list_paths:

        model = LoadModel(filePath = path,
                        variable = "Close_filled", # specify the dependent variable
                        regimes = 3,               # specify the number of regimes
                        level = False,              # specify if the dependent variable is at level (true) or log-returns (false)
                        trend = False,              # specify if trend variable is to be included
                        intercept = True,          # specify if intercept is to be included
                        ar = 1,                    # specify the autoregressive order
                        decimal='.', delimiter=',',
                        parse_dates=["Data"], date_format="%Y-%m-%d",
                        index_col=0,
                        data_ini="2005-11-22",
                        data_fim="2025-11-20",
                        initial_guess=ParameterGuess_instance)

        # Create an estimator instance
        ModelEstimator = MKE.MarkovSwitchingEstimator(model)
        
        # Define file stem pattern for saving results
        fileStem = f"{datetime.today().strftime('%Y-%m-%d %H-%M-%S')} - {ModelEstimator.Model.ModelName}"

        # Fit the model
        ModelEstimator.Fit(traceLevel=1)


        print(MKH.GetRegimeClassification(ModelEstimator.Model))

        # Save model summary to a text file
        print(ModelEstimator.Model)
        with open(f"{fileStem}.txt", 'w', encoding='utf-8') as fileStream:
            print(ModelEstimator.Model, file=fileStream)

        # Make predictions
        new_Y, new_X, new_Xi = ModelEstimator.Predict(h=30)
        print("Predicted Y:\n", new_Y)

        print(MKH.GetRegimeNames(ModelEstimator.Model))

        # Generate and save smoothed probabilities plot
        MKH.GenerateSmoothProbabilitiesPlot(ModelEstimator.Model, fileStem=fileStem)

    print("Finished Estimation")
