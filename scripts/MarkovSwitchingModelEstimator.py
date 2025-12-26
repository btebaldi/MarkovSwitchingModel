# Load necessary libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import MarkovSwitchingModel as MKM
import matplotlib.pyplot as plt
from datetime import datetime

class MarkovSwitching_estimator: 

    def __init__(self, model: MKM.MarkovSwitchingModel):
        """
        Initializes the Class MarkovSwitching_estimator with a Markov Switching Model.

        Parameters:
            model (MarkovSwitchingModel): An instance of the Markov Switching Model containing data and parameters.
        """
        self.Model = model

    def EstimateEta(self) -> np.ndarray:
        """
        Calculates the "quasi-densities" for each regime in a Markov Switching model.

        Returns:
            np.ndarray: The calculated quasi-densities for each regime.
        """
        # TODO: function should be Multivariate Normal, not univariate
        # The following lines are commented out as they are not implemented yet.
        
        # Calculate the quasi-densities using the normal probability density function

        current_Eta = np.ones((self.Model.NumObservations, 0))  # Initialize joe with ones

        # To make the procedure faster one could vectorize the calculation here.
        for regime in range(self.Model.NumRegimes):
            # estimate the quantile of each observation in the regime
            q = self.Model.Y[:, :] - (self.Model.X @ self.Model.Beta[:, regime].reshape(-1,1))
            
            # estimate the quasi-density for each observation in the regime
            q_prob = stats.norm.pdf(x=q, loc=0, scale=self.Model.Omega[0, regime]**0.5)     
            
            # For numerical stability, ensure no zero probabilities
            q_prob = np.where(q_prob > 0, q_prob, 1e-6)

            # Stack the calculated quasi-densities for each regime
            current_Eta = np.column_stack([current_Eta, q_prob])

            # The commented line below suggests a future implementation using multivariate normal distribution.
            # scipy.stats.multivariate_normal.pdf(x = Y, mean = X @ Betas, cov = Omegas)
            
        # Store the calculated quasi-densities
        self.Model.Eta = current_Eta
        return self.Model.Eta

    def EstimateXiFiltered(self) -> np.ndarray:
        """
        Calculates the filtered Xi (regime probabilities p(S_t | info up to t) in a Markov Switching model.
        Equations 5a and 5b in MC (2020).

        Returns:
        - Xi: array (T x M) with filtered probabilities by regime.
        """

        # Initialize Xi with zeros
        for i in range(self.Model.NumObservations):
            if i == 0:
                Xi_t1 = self.Model.UnconditionalStateProbs  # Use unconditional probabilities for the first observation
            else:
                Xi_t1 = self.Model.TransitionMatrix.T @ self.Model.Xi_filtered[i-1]  # Transition from the previous state
            # TODO: Check if we are doing the right matrix multiplication here
            # Update filtered probabilities
            self.Model.Xi_filtered[i] = (self.Model.Eta[i] * Xi_t1) / (self.Model.Eta[i] @ Xi_t1)
            self.Model.Xi_t1_filtered[i] = self.Model.TransitionMatrix.T @ self.Model.Xi_filtered[i]

        return self.Model.Xi_filtered

    def EstimateXiSmoothed(self) -> np.ndarray:
        """
        Calculates the smoothed Xi (regime probabilities p(S_t | info all time points) in a Markov Switching model.
        Equations 6 in MC (2020).

        Returns:
        - Xi_s: array (T x M) with smoothed probabilities by regime.
        """
        Xi_s = self.Model.Xi_filtered.copy()  # Create a copy of filtered Xi

        # Iterate backwards to smooth the probabilities
        for i in range(self.Model.NumObservations - 1, 0, -1):
            if i == (self.Model.NumObservations - 1) :
                Xi_t_given_T = self.Model.Xi_filtered[i]  # Use unconditional probabilities for the first observation
            else:
                # Update filtered probabilities
                Xi_t_given_T = (self.Model.TransitionMatrix @ (Xi_s[i+1] / self.Model.Xi_t1_filtered[i])) * self.Model.Xi_filtered[i] 

            Xi_s[i] = Xi_t_given_T

        # Store the smoothed probabilities
        self.Model.Xi_smoothed = Xi_s  
        return self.Model.Xi_smoothed

    def EstimateUnconditionalStateProbs(self) -> np.ndarray:
        """
        Estimates the unconditional state probabilities.
        Equations 7 in MC (2020).

        Returns:
        - Unconditional probabilities for each regime.
        """
        # Average over the smoothed probabilities

        self.Model.UnconditionalStateProbs = self.Model.Xi_smoothed.mean(axis=0)
        return self.Model.UnconditionalStateProbs 

    def EstimateTransitionMatrix(self) -> np.ndarray:
        """
        Estimates the transition matrix based on smoothed probabilities.
        Equations 8 in MC (2020).

        Returns:
        - transitionMatrix: array (M x M) with estimated transition probabilities.
        """

        # Initialize a transitional matrix with ones
        PP = np.ones((self.Model.NumRegimes, self.Model.NumRegimes))
            
        # This can be vectorized for better performance.
        for regime_from in range(self.Model.NumRegimes):
            for regime_to in range(self.Model.NumRegimes):
            
                xxi = np.append(self.Model.Xi_smoothed[1:, regime_to], self.Model.Xi_filtered[-1, regime_to])

                PP_ij = sum((self.Model.Xi_filtered[ :, regime_from] /  self.Model.Xi_t1_filtered[ :, regime_to]) * xxi)                
                Denominator = sum(self.Model.Xi_smoothed[ :, regime_from])
                
                PP[regime_from, regime_to] = self.Model.TransitionMatrix[regime_from, regime_to] * PP_ij/Denominator

        # Update the transition matrix
        self.Model.TransitionMatrix = PP    
        return self.Model.TransitionMatrix


    def EstimateTransitionMatrix_Hamilton(self) -> np.ndarray:
        """
        Estimates the transition matrix based on smoothed probabilities.
        Equations 8 in MC (2020).

        Returns:
        - transitionMatrix: array (M x M) with estimated transition probabilities.
        """
        # Create a zero matrix for transition counts
        transitionMatrix = np.zeros((self.Model.NumRegimes, self.Model.NumRegimes))

        # Estimate the most likely state at each time point
        estimated_state = np.argmax(self.Model.Xi_smoothed, axis=1)

        # Convert the estimation to a column vector
        estimated_state = estimated_state.reshape(-1, 1)

        # Calculate the unconditional state probabilities for initial state estimation
        UnconditionalStateProbs = self.EstimateUnconditionalStateProbs()
        estimated_initial_state = np.argmax(UnconditionalStateProbs, axis=0)  # Initial state with highest probability

        # Create lagged version of estimated_state
        lagged_state = np.roll(estimated_state, 1, axis=0)
        lagged_state[0] = estimated_initial_state  # Set the first value to the estimated initial state

        # Create a new Xi_smoothed by adding estimated_state and lagged_state as new columns
        Xi_smoothed_helper = np.hstack([
            self.Model.Xi_smoothed,
            estimated_state,
            lagged_state
        ])

        # Estimate transition probabilities based on smoothed probabilities
        for state_i in range(self.Model.NumRegimes):
            for state_j in range(self.Model.NumRegimes):
                # Count_ij : P(S_t = j, S_t-1 = i)
                Count_ij = Xi_smoothed_helper[(Xi_smoothed_helper[:, self.Model.NumRegimes] == state_j) & (Xi_smoothed_helper[:, self.Model.NumRegimes + 1] == state_i),].shape[0]

                # Count_i : P(S_t-1 = i)
                Count_i = Xi_smoothed_helper[Xi_smoothed_helper[:, self.Model.NumRegimes + 1] == state_i, ].shape[0]

                # Update the transition counts matrix
                # Calculate transition probability
                transitionMatrix[state_i, state_j] = Count_ij / Count_i

        # Store the estimated transition matrix
        self.Model.TransitionMatrix = transitionMatrix  
        return self.Model.TransitionMatrix

    def EstimateBeta(self) -> np.ndarray:
        """
        Estimates the coefficients (Betas) for each regime using OLS.
        Equations 9 in MC (2020).

        Returns:
        - Betas_estimated: array (K x M) with estimated coefficients for each regime.
        """
        # Initialize an empty array for estimated Betas
        Betas_estimated = np.zeros((self.Model.NumXVariables, 0))

        # For each regime, run OLS to estimate the regression
        for regime_number in range(self.Model.NumRegimes):
            # Check dimensions of Omega
            # Create weights based on smoothed probabilities
            Weights_Matrix = np.diag(self.Model.Xi_smoothed[:, regime_number])

            # Calculate the design matrix
            A = (self.Model.X.T @ Weights_Matrix @ self.Model.X) 

            try:
                # Inverse of the design matrix
                A_inv = np.linalg.inv(A) 
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                A_inv = np.linalg.pinv(A)

            # Calculate Betas
            B = A_inv @ self.Model.X.T @ Weights_Matrix @ self.Model.Y 

            # Stack the estimated Betas for each regime
            Betas_estimated = np.column_stack([Betas_estimated, B])

        # Store the estimated Betas
        self.Model.Beta = Betas_estimated  
        return Betas_estimated

    def EstimateResiduals(self) -> np.ndarray:
        """
        Estimates the residuals from the model.
        
        Returns:
        - Residuals from the model.
        """
        # Call method to get residuals
        return self.Model.GetResiduals() 

    def EstimateOmega(self) -> np.ndarray:
        """
        Estimates the variance (Omega) for each regime.
        Equations 10 in MC (2020).

        Returns:
        - Omega_estimated: array (1 x M) with estimated variances for each regime.
        """
        Omega_estimated = np.ones((1, 0))  # Initialize Omega estimates

        # For each regime, calculate the variance
        residuals = self.Model.GetResiduals()  # Get residuals
        for regime_number in range(self.Model.NumRegimes):
            Weights_Matrix = np.diag(self.Model.Xi_smoothed[:, regime_number])  # Create weights based on smoothed probabilities

            # Calculate Omega as the weighted sum of squared residuals
            # Omega = (residuals[:, regime_number].T @ Weights_Matrix @ residuals[:, regime_number]) / sum(self.Model.Xi_smoothed[:, regime_number])
            Omega = sum(residuals[:, regime_number]**2 * self.Model.Xi_smoothed[:, regime_number])/sum(self.Model.Xi_smoothed[:, regime_number])
            # Stack the estimated Omega for each regime
            Omega_estimated = np.column_stack([Omega_estimated, Omega])

        self.Model.Omega = Omega_estimated  # Store the estimated Omega, excluding the initial zeros
        return self.Model.Omega
    
    def Fit(self, maxInterations = 1000, minInterations = 5, precision: float = 1e-6, traceLevel : int = 0) -> None:
        """
        Fits the Markov Switching Model by estimating all parameters.

        Returns:
            None
        """

        delta = float('inf')
        interationCounter = 0
        while (delta > precision) or (interationCounter < minInterations):

            if(interationCounter < maxInterations):
                interationCounter += 1
            else:
                break

            # Store previous parameters for convergence check
            if interationCounter == 1:
                prev_ll = float('-inf')
            else:
                prev_ll = self.Model.GetLogLikelihood()
            
            # Estimate all parameters
            self.EstimateEta()
            if traceLevel > 1:
                print(f"Iteration {interationCounter}: Estimated Eta")
            self.EstimateXiFiltered()
            if traceLevel > 1:
                print(f"Iteration {interationCounter}: Estimated Xi Filtered")
            self.EstimateXiSmoothed()
            if traceLevel > 1:
                print(f"Iteration {interationCounter}: Estimated Xi Smoothed")
            self.EstimateUnconditionalStateProbs()
            if traceLevel > 1:
                print(f"Iteration {interationCounter}: Estimated Unconditional State Probs")
            self.EstimateTransitionMatrix()
            if traceLevel > 1:
                print(f"Iteration {interationCounter}: Estimated Transition Matrix")
                print(f"{self.Model.TransitionMatrix}\n")

            self.EstimateBeta()
            if traceLevel > 1:
                print(f"Iteration {interationCounter}: Estimated Beta")
                print(f"{self.Model.Beta}\n")
            # self.EstimateResiduals()
            # if traceLevel > 1:
            #     print(f"Iteration {interationCounter}: Estimated Residuals")
            self.EstimateOmega()
            if traceLevel > 1:
                print(f"Iteration {interationCounter}: Estimated Omega")
            
            # Calculate convergence error
            cur_ll = self.Model.GetLogLikelihood()
            delta = cur_ll - prev_ll
            if traceLevel > 0:
                print(f"Iteration {interationCounter}: Estimated LogLikelihood {cur_ll:.6f} with change {delta:9.6f} - {self.Model.LogLikeOx():.6e}{' Warning: model is diverging. Log-Likelihood decreased.' if delta < 0 else ''}")
            delta = abs(cur_ll - prev_ll)
            
def LoadModel(filePath, 
              variable, Xvariable = None, ar = 1, level = False, intercept = True, trend = True,
              regimes = 2,
              decimal='.', delimiter=',', parse_dates=None, date_format="%Y-%m-%d",
              index_col=None, data_ini = None, data_fim = None) -> MKM.MarkovSwitchingModel:
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
        df[f"Y_lag_{lag}"] = df[variable].shift(lag)
    
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

    # Create a trend variable (linear time index from 0 to T-1)
    # Note: This variable is currently unused in the model specification below
    data_trend = np.arange(Y.shape[0])
    
    # Create an intercept column (vector of ones) for the constant term in the regression
    data_intercept = np.ones(Y.shape[0])
    
    # Create an instance of the Markov Switching Model
    Total_Exo_vars = len(Xvariable) if Xvariable is not None else 0
    param_names = {'Y':'Close_filled', 'X':[f'Exo_{i}' for i in range(Total_Exo_vars)] + 
                        ['Intercept', 'Trend'] + 
                    [f'Lag_{i}' for i in range(1, ar + 1)]}
 
    # add intercept and trend if specified
    X = np.empty((Y.shape[0], 0))  # Initialize X as an empty array

    if Xvariable is not None:
        X = np.column_stack([X, df[Xvariable].to_numpy()])

    if trend :
        X = np.column_stack([X, data_trend])
    else:
        param_names['X'].remove('Trend')
 
    if intercept :
        X = np.column_stack([X, data_intercept])
    else:
        param_names['X'].remove('Intercept')

    # Build the independent variable matrix X by horizontally stacking the intercept and lagged dependent variable
    # X has dimensions (T x 2): first column is the constant, second column is lagged Close-Level
    for lag in range(1, ar + 1):
        X = np.column_stack([X, df[f'Y_lag_{lag}'].to_numpy()])
    
    
    # make a random selection of initial beta values for the lags
    Beta_initial = np.zeros((0, regimes))
    for k in range(X.shape[1]):
        random_values = np.random.uniform(-0.5, 0.5, size=regimes)
        Beta_initial = np.vstack([Beta_initial, random_values])
    
    # ===== Initial beta values =====
    Beta_initial = np.array([[10, 60],
                  [0.04, 0.05]])
    # np.random.uniform(-0.5, 0.5, size=regimes)

    Omega_initial = np.array([[50**2, 40**2]])  # Initial omega values``

    model = MKM.MarkovSwitchingModel(Y, X, num_regimes=regimes,
                                    beta=Beta_initial, 
                                     omega=Omega_initial, 
                                     param_names=param_names,
                                     dates_label=df.index)

    return model

def GenerateSmoothProbabilitiesPlot(model: MKM.MarkovSwitchingModel) -> None:
    """
    Visualizes the smoothed regime probabilities for each regime over time.
    
    This function creates a multi-panel plot where each subplot displays the smoothed
    probability of being in a specific regime across all time periods. The plot is saved
    as a PNG file for later analysis.
    
    Parameters:
        model (MKM.MarkovSwitchingModel): The fitted Markov Switching Model containing
                                          smoothed probabilities (Xi_smoothed) and dates.
    
    Returns:
        None: Saves the plot to 'markov_switching_plot.png' without returning a value.
    """
    # Create a figure with subplots, one for each regime
    # figsize=(10, 6) sets the figure dimensions to 10 inches wide by 6 inches tall
    fig, axes = plt.subplots(model.NumRegimes, figsize=(10, 6))

    # Iterate through each regime to plot its smoothed probability over time
    for cur_mod in range(model.NumRegimes):
        # Select the current subplot (axis) for the current regime
        ax = axes[cur_mod]
        
        # Plot the smoothed probabilities for the current regime against the date labels
        # model.DatesLabel contains the time index (dates or periods)
        # model.Xi_smoothed[:, cur_mod] contains the smoothed probability for regime cur_mod at each time point
        ax.plot(model.DatesLabel, model.Xi_smoothed[:, cur_mod])
        
        # Set the y-axis label to indicate which regime this subplot represents
        ax.set_ylabel(f"Regime {cur_mod}")
    
    # Adjust the layout to prevent overlapping labels and titles
    fig.tight_layout()

    # Save the figure to a PNG file with high resolution (300 DPI)
    # bbox_inches='tight' ensures no content is cut off at the figure edges
    plt.savefig(f"markov_switching_plot_{datetime.today().strftime('%Y-%m-%d %H-%M-%S')}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__" :
    # Load the model from a CSV file
    
    # model = LoadModel(filePath = "./database/Validation dataset/teste com 3 regimes.csv",
    #                   variable = "Y",
    #                   regimes=3,
    #                   trend=False,
    #                   level = True,
    #                   intercept = True,
    #                   ar = 5,
    #                   decimal=',', delimiter=';', parse_dates=None, date_format=None)
    
    # model = LoadModel(filePath = ".\\database\\filled\\monthly\\DOLARF_output.csv",
    #                   variable = "Close_filled", # specify the dependent variable
    #                   regimes = 2,               # specify the number of regimes
    #                   level = False,              # specify if the dependent variable is at level (true) or log-returns (false)
    #                   trend = False,              # specify if trend variable is to be included
    #                   intercept = False,          # specify if intercept is to be included
    #                   ar = 2,                    # specify the autoregressive order
    #                   decimal='.', delimiter=',',
    #                   parse_dates=["Data"], date_format="%Y-%m-%d",
    #                   index_col=0,
    #                   data_ini=None,
    #                   data_fim="2015-12-31")

    model = LoadModel(filePath = "./database/Validation dataset/Regime_2/resultados.csv",
                    variable = "y", # specify the dependent variable
                    Xvariable = ["x"], # specify the dependent variable
                    regimes = 2,               # specify the number of regimes
                    level = True,              # specify if the dependent variable is at level (true) or log-returns (false)
                    trend = False,              # specify if trend variable is to be included
                    intercept = True,          # specify if intercept is to be included
                    ar = 0,                    # specify the autoregressive order
                    decimal=',', delimiter=';' )
    
    
    # Create an estimator instance
    ModelEstimator = MarkovSwitching_estimator(model)
    
    ModelEstimator.Fit(traceLevel=1)
    
    with open(f"{datetime.today().strftime('%Y-%m-%d %H-%M-%S')}_Console (Maddalena).txt", 'w', encoding='utf-8') as fileStream:
        print(ModelEstimator.Model, file=fileStream)
    print(ModelEstimator.Model)


    GenerateSmoothProbabilitiesPlot(ModelEstimator.Model)
    print("Finished Estimation")
