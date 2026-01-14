# Load necessary libraries
import numpy as np
import pandas as pd
import MarkovSwitchingModel as MKM
import MarkovSwitchingModelEstimator as MKE
import MarkovSwitchingModelHelper as MKH
from datetime import date, datetime
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
              decimal='.', delimiter=',', parse_dates=None, date_format="%Y-%m-%d",
              index_col=None, data_ini = None, data_fim = None,
              initial_guess=None) -> MKM.MarkovSwitchingModel:
    """
    Load data from CSV, engineer regressors, and build a MarkovSwitchingModel instance.

    Core idea
    ---------
    Loads and preprocesses data from a CSV file to create a Markov Switching Model.

        Y_t = X_t * Beta_{S_t} + error_t
        error_t ~ N(0, Omega_{S_t})
        S_t follows a Markov chain with M regimes

    where the regressors X_t can include:
    - intercept
    - trend
    - exogenous variables (Xvariable)
    - autoregressive lags of Y (AR terms)

    Parameters (high-level)
    -----------------------
    filePath : str or Path
        CSV file containing the dataset (time-series).
    variable : str
        Column in CSV used to create the dependent series.
    Xvariable : list[str] or None
        Optional list of additional exogenous regressors to include in X.
    ar : int
        Number of autoregressive lags of Y to add to X.
    level : bool
        If True: model Y as the original series level.
        If False: model Y as log-returns: log(variable_t / variable_{t-1}).
    intercept : bool
        If True: add a constant regressor.
    trend : bool
        If True: add a deterministic time trend regressor (0,1,2,...).
    regimes : int
        Number of regimes (Markov states).
    decimal, delimiter, parse_dates, date_format, index_col
        CSV parsing options passed to pandas.read_csv.
    data_ini, data_fim
        Optional date filtering bounds (works only if index_col is set).
    initial_guess : ParameterGuess or None
        Optional container with starting values for Beta/Omega (and potentially more).

    Returns
    -------
    MKM.MarkovSwitchingModel
        A model object with (Y, X), regime count, initial parameters, and parameter metadata.

    Important implementation notes
    ------------------------------
    - Building Y via log-returns drops the first row (because of shift(1) -> NaN).
    - Building AR lags drops the first 'ar' rows (because of shift(lag) -> NaN).
    - If date filtering is requested, you must have an index_col set so the DataFrame index is datelike.
    """

    # Initialize parameter names dictionary 
    # Used for reporting/interpretation (e.g., printing "Intercept", "Lag_1", etc.)
    # Your MarkovSwitchingModel uses this metadata to label outputs.
    param_names = MKM.GetEmptyParamNames()

    # Normalize filePath into a Path object (works with strings or Paths)
    filePath = plib.Path(filePath)

    # =========================
    # 1) LOAD DATA FROM CSV
    # =========================
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
    param_names['Y'][0] = MKM.GetDictRepresentation(name = variable, type = MKM.TypeOfVariable.DEPENDENT, classX = None,
                                                    ar = None)

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
                                    beta=initial_guess.Beta, 
                                     omega=initial_guess.Omega,
                                     transitionMatrix = initial_guess.TransitionMatrix,
                                     unconditional_state_probs = initial_guess.UnconditionalStateProbability, 
                                     param_names=param_names,
                                     dates_label=dates_label,
                                     model_name=filePath.stem)

    return model

if __name__ == "__main__" :
    # =========================================================
    # Example usage: estimate a Markov Switching AR model
    # =========================================================
    # This section demonstrates:
    # - setting explicit starting values for Beta and Omega
    # - loading one (or multiple) datasets
    # - fitting the Markov Switching model
    # - saving summaries and plots
    # - forecasting h steps ahead

    ParameterGuess_instance = ParameterGuess()
    ParameterGuess_instance.Beta = np.array([[-0.0002, 0.0003,  0.0023],
                                               [0.11, 0.10, -0.08]])
    ParameterGuess_instance.Beta = np.array([[-0.0002, 0.0003,  0.0023],
                                               [0.11, 0.10, -0.08]])
    ParameterGuess_instance.Omega = np.array([[0.04**2, 0.09**2, 0.02**2]])

    # Candidate dataset paths to estimate on
    list_paths = [
        ".\\database\\filled\\DOLARF_filled.csv",
        ".\\database\\filled\\IBOV_filled.csv",
        ".\\database\\filled\\NASDAQ_filled.csv",
        ".\\database\\filled\\SnP500_filled.csv",
        ".\\database\\filled\\SMLL_filled.csv",
        ".\\database\\filled\\T-BOND10_filled.csv"
    ]

    # Override to run only one dataset (NASDAQ) for faster testing/debugging
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
        
        # Construct a timestamped filename stem for outputs (summary text + plot PNG)
        fileStem = f"{datetime.today().strftime('%Y-%m-%d %H-%M-%S')} - {ModelEstimator.Model.ModelName}"

        # Fit the model
        ModelEstimator.Fit(traceLevel=1)

        # Regime classification:
        # Groups consecutive time periods by most-likely regime and summarizes durations/probabilities.
        print(MKH.GetRegimeClassification(ModelEstimator.Model))

        # Print and persist the model summary (coefficients, variances, transition matrix, ICs, etc.)
        print(ModelEstimator.Model)
        # with open(f"{fileStem}.txt", 'w', encoding='utf-8') as fileStream:
        #     print(ModelEstimator.Model, file=fileStream)

        # Forecasting:
        # Predict h steps ahead using estimated transition dynamics and regime-weighted forecasts.
        new_Y, new_X, new_Xi = ModelEstimator.Predict(h=30)
        print("Predicted Y:\n", new_Y)

        # Human-readable regime names based on statistical properties (helper-defined heuristic)
        print(MKH.GetRegimeNames(ModelEstimator.Model))

        # Define file stem pattern and dir for saving results
        fileStem = f"{ModelEstimator.Model.ModelName}"
        dirPath = plib.Path(".\\")

        # Visualization:
        # Save smoothed regime probability plots for interpretation and reporting.
        MKH.GenerateSmoothProbabilitiesPlot(ModelEstimator.Model, filePath=dirPath / f"{fileStem} - SmoothProbabilities.png")

    print("Finished Estimation")
