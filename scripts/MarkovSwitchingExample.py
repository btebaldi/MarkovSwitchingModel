# Load necessary libraries
from datetime import date, datetime
import numpy as np
import pandas as pd
import MarkovSwitchingModel as MKM
import MarkovSwitchingModelEstimator as MKE
import MarkovSwitchingModelHelper as MKH
import pathlib as plib



class ParameterGuess:
    """
    Container for optional initial parameter guesses.

    Why this exists
    ---------------
    Markov Switching estimation can be sensitive to starting values (local maxima, slow convergence).
    This class gives you a clean interface to optionally pass:
    - Beta: regime-specific regression coefficients
    - Omega: regime-specific variances (or scale parameters)
    - TransitionMatrix: initial Markov transition probabilities
    - UnconditionalStateProbability: initial stationary probabilities

    """

    def __init__(self) -> None:
        # Regime-specific coefficients: expected shape (K, M)
        # K = number of regressors in X
        # M = number of regimes
        
        self.Beta: np.ndarray | None = None

        # Regime-specific variances (or covariance parameters): expected shape (1, M)
        self.Omega: np.ndarray | None = None

        # Markov transition matrix: expected shape (M, M)
        self.TransitionMatrix: np.ndarray | None = None

        # Unconditional regime probabilities: expected length M
        self.UnconditionalStateProbability: np.ndarray | None = None

    def GetParameters(self):
        """
        Return all stored initial guesses as a tuple.

        Returns
        -------
        tuple
            (Beta, Omega, TransitionMatrix, UnconditionalStateProbability)
        """
        return self.Beta, self.Omega, self.TransitionMatrix, self.UnconditionalStateProbability


def LoadModel(filePath, 
              variable, Xvariable: list | None = None,
              ar = 1,
              level = False, intercept = True, trend = True,
              regimes = 2,
              decimal='.', delimiter=',', parse_dates=None, date_format: str| None="%Y-%m-%d",
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
    else:
        df['Y'] = np.log(df[variable] / df[variable].shift(1))
        
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
            df = df.loc[data_ini:data_fim]
        else:
            raise ValueError("data_ini or data_fim specified but no index column provided. please provide an index column.")
   
    # ===== Extract Y and X from the database =====
    Y = df['Y'].to_numpy().reshape((-1, 1))
    param_names['Y'][0] = MKM.GetDictRepresentation(name = variable, type = MKM.TypeOfVariable.DEPENDENT, classX = None, ar = None)

    # Create a trend variable (linear time index from 0 to T-1)
    # Note: This variable is currently unused in the model specification below
    data_trend = np.arange(Y.shape[0])
    
    # Create an intercept column (vector of ones) for the constant term in the regression
    data_intercept = np.ones(Y.shape[0])
    
    # Create an instance of the Markov Switching Model
    # Initialize X as an empty array
    X = np.empty((Y.shape[0], 0))

    nColumn_X = 0
    if intercept :
        X = np.column_stack([X, data_intercept])
        param_names['X'][nColumn_X] = MKM.GetDictRepresentation(name = 'Intercept',
                                                        type = MKM.TypeOfVariable.INDEPENDENT,
                                                        classX = MKM.TypeOfDependentVariable.INTERCEPT,
                                                        ar = None)
        nColumn_X = nColumn_X + 1

    if trend :
        X = np.column_stack([X, data_trend])
        param_names['X'][nColumn_X] = MKM.GetDictRepresentation(name = 'Trend',
                                                        type = MKM.TypeOfVariable.INDEPENDENT,
                                                        classX = MKM.TypeOfDependentVariable.TREND,
                                                        ar = None)
        nColumn_X = nColumn_X + 1


    if Xvariable is not None:
        for Xvar in Xvariable:
            X = np.column_stack([X, df[Xvar].to_numpy()])
            param_names['X'][nColumn_X] = MKM.GetDictRepresentation(name = Xvar,
                                                            type = MKM.TypeOfVariable.INDEPENDENT,
                                                            classX = MKM.TypeOfDependentVariable.EXOGENOUS,
                                                            ar = None)
            nColumn_X = nColumn_X + 1

    # Build the independent variable matrix X by horizontally stacking the intercept and lagged dependent variable
    # X has dimensions (T x 2): first column is the constant, second column is lagged Close-Level
    for lag in range(1, ar + 1):
        X = np.column_stack([X, df[f'Y_lag_{lag}'].to_numpy()])
        # param_names['X'][f'Lag_{lag}'] = MKM.TypeOfXVariable.AUTO_REGRESSIVE
        param_names['X'][nColumn_X] = MKM.GetDictRepresentation(name = f"Lag_{lag}",
                                                            type = MKM.TypeOfVariable.INDEPENDENT,
                                                            classX = MKM.TypeOfDependentVariable.AUTO_REGRESSIVE,
                                                            ar = lag)
        nColumn_X = nColumn_X + 1
    
    # ===== Generate initial parameter guesses =====
    if initial_guess is None:
        initial_guess = ParameterGuess()

    # ===== Create the dates label =====
    if not all(isinstance(x, (date, datetime)) for x in list(df.index)):
        dates_label = None
    else:
        dates_label = list(df.index)
    
    model = MKM.MarkovSwitchingModel(Y, X, num_regimes=regimes,
                                    beta = initial_guess.Beta, 
                                     omega = initial_guess.Omega, 
                                     transitionMatrix = initial_guess.TransitionMatrix,
                                     unconditional_state_probs = initial_guess.UnconditionalStateProbability,
                                     param_names = param_names,
                                     dates_label = dates_label,
                                     model_name = filePath.stem)

    return model


def TwoRegimeExample():

    ParameterGuess_instance = ParameterGuess()
    ParameterGuess_instance.Beta = np.array([[1, 20],
                                             [10,  1]])
    ParameterGuess_instance.Omega = np.array([[3**2, 10**2]])

    path = ".\\Examples\\TwoRegimes\\R2_DGP.csv"

    model = LoadModel(filePath = path,
                    variable = "y", # specify the dependent variable
                    Xvariable = ["x"],        # specify exogenous variables
                    regimes = 2,               # specify the number of regimes
                    level = True,              # specify if the dependent variable is at level (true) or log-returns (false)
                    trend = False,              # specify if trend variable is to be included
                    intercept = True,          # specify if intercept is to be included
                    ar = 0,                    # specify the autoregressive order
                    decimal=',', delimiter=';',
                    parse_dates=None,
                    date_format=None,
                    index_col=None,
                    data_ini=None,
                    data_fim=None,
                    initial_guess=ParameterGuess_instance)
    
    # Create an estimator instance
    ModelEstimator = MKE.MarkovSwitchingEstimator(model)
        


    # Fit the model
    ModelEstimator.Fit(traceLevel=1)

    # Define file stem pattern and dir for saving results
    fileStem = f"TwoRegimes_output - {ModelEstimator.Model.ModelName}"
    dirPath = plib.Path(".\\Examples\\TwoRegimes")

    # Save model summary to a text file
    print(ModelEstimator.Model)
    with open(dirPath / f"{fileStem}.txt", 'w', encoding='utf-8') as fileStream:
        print(ModelEstimator.Model, file=fileStream)

    # Save the model matrices to CSV files
    pd.DataFrame(ModelEstimator.Model.Y).to_csv(dirPath / f"{fileStem}_Y.csv", index=False, header=False)
    pd.DataFrame(ModelEstimator.Model.X).to_csv(dirPath / f"{fileStem}_X.csv", index=False, header=False)
    pd.DataFrame(ModelEstimator.Model.Xi_smoothed).to_csv(dirPath / f"{fileStem}_Xi_smoothed.csv", index=False, header=False)

    # Generate and save smoothed probabilities plot
    MKH.GenerateSmoothProbabilitiesPlot(ModelEstimator.Model, filePath=dirPath / f"{fileStem} markov_switching_plot.png")


if __name__ == "__main__" :
    

    TwoRegimeExample()

     
    print("Finished Estimation")
