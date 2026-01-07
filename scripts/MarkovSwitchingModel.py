import numpy as np
from datetime import datetime
import scipy.stats as stats
from enum import Enum

class TypeOfXVariable(Enum):
    """Enumeration for TypeX with values 1, 2, 3."""
    INTERCEPT = 1
    TREND = 2
    AUTO_REGRESSIVE = 3
    EXOGENOUS = 4

class MarkovSwitchingModel:
    """
    A class to represent time series data and regime-switching model parameters.
    Implements a Markov regime-switching model for modeling behavior that changes
    across different economic or market regimes.
    """

    def __init__(self, Y, X, num_regimes, beta=None, omega=None,
                  transitionMatrix=None, unconditional_state_probs=None, param_names = dict | None, dates_label=None, model_name=None):
        """
        Initialize a MarkovSwitchingModel instance with data and parameters.

        Parameters:
            Y (array-like): The dependent variable data (1D or 2D array).
            X (array-like): The independent variable(s), shape (observations, variables).
            num_regimes (int): Number of regimes (states) in the data/model.
            beta (array-like, optional): Coefficient values per regime, shape (variables, regimes).
            omega (array-like, optional): Variance/scale parameters per regime.
            transitionMatrix (array-like, optional): Regime transition probabilities, shape (regimes, regimes).
            unconditional_state_probs (array-like, optional): Initial state probabilities.
        """
        
        # Validate num_regimes: must be greater than 1
        if num_regimes <= 1:
            raise ValueError("Number of regimes must be greater than 1.")

        # Validate Y input: reject 2-column Y (multivariate not supported)
        if len(Y.shape) == 2 and Y.shape[1] >= 2:
            raise NotImplementedError("Method not implemented for two Y columns")

        # Check for missing values or NaN in Y
        if np.any(np.isnan(Y)):
            raise ValueError("Y contains NaN (missing) values.")

        # Store the dependent variable (Y)
        self.Y = Y

        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN (missing) values.")

        # Store the independent variables (X)
        self.X = X

        # Extract number of exogenous variables from X's column dimension
        self.NumXVariables = self.X.shape[1]

        # Validate data alignment: X and Y must have equal observations
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must have the same number of observations (rows).")

        # Store total number of observations (time periods)
        self.NumObservations = self.X.shape[0]

        # Store the number of regimes/states
        self.NumRegimes = num_regimes

        # ===== Initialize Beta (regime-specific coefficients) =====
        if beta is not None:
            # Validate beta shape: must be (num_variables, num_regimes)
            if beta.shape[1] != self.NumRegimes:
                raise ValueError(f"Number of columns ({beta.shape[1]}) in betas must be equal to number of regimes ({self.NumRegimes}).")
            
            # Validate beta rows match X columns (exogenous variables)
            if self.X.shape[1] != beta.shape[0]:
                raise ValueError(f"Number of exogenous variables in X ({self.X.shape[1]}) must match beta dimensions ({beta.shape[0]}).")
            
            self.Beta = beta
        else:
            # Default: initialize all coefficients to zero
            # Assumes X already includes a constant term as first column
            # self.Beta = np.zeros((self.NumXVariables, self.NumRegimes))
            self.Beta = np.random.randn(self.NumXVariables, self.NumRegimes)/100
        
        # ===== Initialize Omega (regime-specific variance/scale) =====
        if omega is not None:
            # Validate omega columns match number of regimes
            if omega.shape[1] != self.NumRegimes:
                raise ValueError(f"Number of columns in omegas({omega.shape[1]}) must be equal to num_regimes ({self.NumRegimes}).")
            
            # Validate omega rows match Y dimensions
            if omega.shape[0] != self.Y.shape[1]:
                raise ValueError(f"Omegas must have the same dimension as Y ({self.Y.shape[1]}).")
            
            # Validate all omega values are positive (variance must be > 0)
            if np.any(omega <= 0):
                raise ValueError("Omega values must be positive (greater than zero).")
            
            self.Omega = omega
        else:
            # Default: initialize to ones for all regimes
            self.Omega = np.ones((1, self.NumRegimes)) + (np.random.randn(1, self.NumRegimes)/10)

        # ===== Initialize state probability matrices =====
        # Eta: filtered conditional probability of being in each regime per observation
        self.Eta = np.zeros((self.NumObservations, self.NumRegimes))
        
        # Xi_filtered: filtered state probabilities (one-step ahead)
        self.Xi_filtered = np.zeros((self.NumObservations, self.NumRegimes))
        
        self.Xi_t1_filtered = np.zeros((self.NumObservations, self.NumRegimes))
        

        # Xi_smoothed: smoothed state probabilities (full-sample inference)
        self.Xi_smoothed = np.zeros((self.NumObservations, self.NumRegimes))

        # ===== Initialize unconditional state probabilities =====
        if unconditional_state_probs is None:
            # Default: equal probability for all regimes
            unconditional_state_probs = np.ones(self.NumRegimes) / self.NumRegimes

        # Validate length matches number of regimes
        if len(unconditional_state_probs) != self.NumRegimes:
            raise ValueError(f"Unconditional state probabilities length ({len(unconditional_state_probs)}) must match number of regimes ({self.NumRegimes})")

        self.UnconditionalStateProbs = unconditional_state_probs

        # ===== Initialize Transition Matrix =====
        if transitionMatrix is not None:
            # Validate square matrix (regimes × regimes)
            if transitionMatrix.shape[0] != transitionMatrix.shape[1]:
                raise ValueError("The Transition Matrix must be square.")
            
            # Validate dimensions match number of regimes
            if transitionMatrix.shape[0] != self.NumRegimes:
                raise ValueError("Rows/columns in Transition Matrix must equal number of regimes.")
            
            self.TransitionMatrix = transitionMatrix
        else:
            # Default: equal transition probabilities between all regimes
            self.TransitionMatrix = np.zeros((self.NumRegimes, self.NumRegimes))
            for i in range(self.NumRegimes):
                for j in range(self.NumRegimes):
                    if i == j:
                        self.TransitionMatrix[i, j] = 0.9
                    else:
                        self.TransitionMatrix[i, j] = 0.1 / (self.NumRegimes - 1)


        # ===== Parameter names for reporting =====
        if param_names is None:
            self.ParamNames = {"Y" : "Dependent Variable", "X" : {k: TypeOfXVariable.EXOGENOUS for k in [f"Exo{i}" for i in range(5)]} }
        else:
            if "Y" not in param_names:
                raise ValueError("param_names must contain the key 'Y'.")
            if "X" not in param_names:
                raise ValueError("param_names must contain the key 'Y'.")
            else:
                if not isinstance(param_names["X"], dict) or len(param_names["X"]) != self.NumXVariables:
                    raise ValueError(f"param_names['X'] must be a dict with exactly {self.NumXVariables} elements.")
            self.ParamNames = param_names

        # ===== Initialize dates label =====
        if dates_label is None:
            # Default: initialize as sequential numbering from 0 to NumObservations-1
            self.DatesLabel = np.arange(self.NumObservations)
        else:
            # Validate dates_label length matches number of observations
            if len(dates_label) != self.NumObservations:
                raise ValueError(f"dates_label length ({len(dates_label)}) must match number of observations ({self.NumObservations}).")
            self.DatesLabel = dates_label

        self._startValues = {"beta":self.Beta, "omega":self.Omega, "transitionMatrix":self.TransitionMatrix, "unconditional_state_probs":self.UnconditionalStateProbs}
        
        if model_name is None:
            self.ModelName = "Untitled Markov Switching Model"
        else:
            self.ModelName = model_name

        self.NumParameters = (self.NumXVariables * self.NumRegimes) + self.NumRegimes * (self.NumRegimes - 1) + self.Omega.size 

    def GetResiduals(self):
        """
        Calculate and return residuals from the model.
        Residuals = actual Y - predicted Y (based on current beta estimates).

        Returns:
            np.ndarray: Residual matrix of shape (observations, regimes).
        """
        residuals = self.Y - self.X @ self.Beta
        return residuals

    def GetSSE(self) -> float:
        """
        Calculate the Sum of Squared Errors (SSE) for the model.
        
        SSE measures the total squared deviation between observed and predicted values.
        A lower SSE indicates better model fit.

        Returns:
            np.ndarray: Sum of squared errors matrix of shape (regimes, regimes).
        """
        residuals = self.GetResiduals()

        SSE = 0
        for regime_number in range(self.NumRegimes):
            SSE += sum(residuals[:, regime_number] * residuals[:, regime_number] * self.Xi_smoothed[:, regime_number])
        
        return SSE

    def GetAIC(self) -> float:
        """
        Compute the Akaike Information Criterion per observation.

        Returns:
            float: AIC derived from the current log-likelihood and parameter count.
        """
        aic = (-2 * self.GetLogLikelihood() + 2*self.NumParameters) / self.NumObservations
        return aic
    
    def GetBIC(self) -> float:
        """
        Compute the Bayesian Information Criterion per observation.

        Returns:
            float: BIC derived from the current log-likelihood and parameter count.
        """
        aic = (-2 * self.GetLogLikelihood() + self.NumParameters*np.log(self.NumObservations)) / self.NumObservations
        return aic
    
    def EstimateResidualVariance(self) -> float:
        residual_variance = self.GetSSE() / (self.NumObservations)
        return residual_variance

    def __str__(self):
        return self.Summary()

    def Summary(self):
        """
        Generate a summary report of the Markov Switching Model results.
        
        Returns:
            str: Formatted summary string containing model statistics and coefficients.
        """
        output = ""
        output += "=" * 88 + "\n"
        output += f"{self.ModelName}\n\n"

        output += f"MarkovSwitching({self.NumRegimes} regimes) Modelling {self.ParamNames['Y']}\n"
        output += f"\nSummary of Results\n"
        output += f"{'Date of estimation:':<22s}{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n"
        output += f"{'No. of observations:':<22s}{self.NumObservations}\n"
        output += f"{'Date range:':<22s}{self.DatesLabel[0]} to {self.DatesLabel[-1]}\n"
        output += f"{'log-likelihood:':<22s}{self.GetLogLikelihood():.4f}\n"
        # output += f"{'log-likelihood (Doornik):':<22s}{self.LogLikeOx():.4f}\n"
        # output += f"{'Varriance:':<22s}{self.EstimateResidualVariance()}\n"
        output += "=" * 88 + "\n"
        output += f"{self.ParamNames['Y']} ~ {' + '.join(self.ParamNames['X'].keys())}"
        output += "\n\n"
        output += f"{'':22s}{'Coefficient':>12s} {'Std.Error':>11s} {'t-value':>9s} {'t-prob':>8s}\n"

        betaVariances = self.GetBetaVariance()

        # ===== Coefficients =====
        for i in range(self.NumXVariables):
            for r in range(self.NumRegimes):
                coef = self.Beta[i, r]
                se   = betaVariances[i, r] ** 0.5
                tval = coef / se
                pval = 2 * (1 - stats.t.cdf(abs(tval), df=self.NumObservations - self.NumXVariables))
                
                stars = ""
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.10:
                    stars = "*"

                name = f"{list(self.ParamNames['X'].keys())[i]}(S = {r})"
                output += (
                    f"{name:<22s}"
                    f"{coef:12.7f} "
                    f"{se:11.7f} "
                    f"{tval:9.4f} "
                    f"{pval:8.4f} {stars}\n"

                    # f"{"Not.Imp.":>11s}"
                    # f"{"Not.Imp.":>9s}"
                    # f"{"Not.Imp.":>8s}\n"

                )
   
        output += "\n"
        output += f"{'':16s}{'Coefficient':>12s} {'Std.Error':>14s}\n"

        # ===== Sigma =====
        for r in range(self.NumRegimes):
            output += (
                f"{f'sigma(S = {r})':<16s}"
                f"{self.Omega[0, r]**0.5:12.7f} "
                f"{"Not.Imp."}\n"
            )

        # ===== Transition probs =====
        for j in range(self.NumRegimes):
            for i in range(self.NumRegimes):
                output += (f"{f'p_{{{j}|{i}}}':<16s}"
                        f"{self.TransitionMatrix[i, j]:12.7f} "
                        f"{"Not.Imp."}\n"
                )


        # ===== Unconditional Probabilities =====
        output += "\n\n"
        for r in range(self.NumRegimes):
            output += (
                f"{f'pi(S = {r})':<16s}"
                f"{self.UnconditionalStateProbs[r]:12.7f} \n"
            )

        output += "\n"
        # ===== Information criteria =====
        output += f"{'AIC':<18s}{self.GetAIC():12.8f}\n{'BIC':<18s}{self.GetBIC():12.8f}\n"


        output += f"Start Values : {self._startValues}\n"
        output += "=" * 88 + "\n"
        
        return output



    def GetBetaVariance(self):

        # f = lambda b: self.GetLogLikelihood(b)
        # hessian = numerical_hessian(f, beta = self.Beta, h = 1e-4)

        Var_Betas_estimated = np.zeros((self.NumXVariables, 0))

        # For each regime, run OLS to estimate the regression
        for regime_number in range(self.NumRegimes):
            # Check dimensions of Omega
            # Create weights based on smoothed probabilities
            Weights_Matrix = np.diag(self.Xi_smoothed[:, regime_number])

            # Calculate the design matrix
            A = (self.X.T @ Weights_Matrix @ self.X) 

            try:
                # Inverse of the design matrix
                A_inv = np.linalg.inv(A) 
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                A_inv = np.linalg.pinv(A)

            # Calculate Betas
            B = np.diag(A_inv * self.Omega[0, regime_number]) 

            # Stack the estimated Betas for each regime
            Var_Betas_estimated = np.column_stack([Var_Betas_estimated, B])
        
        return Var_Betas_estimated
    
    def GetLogLikelihood(self):
        """
        Calculate the log-likelihood of the model given current parameters.
        
        The likelihood combines the SSE with state probabilities weighted by
        the smoothed regime probabilities across all observations.

        Returns:
            float: The log-likelihood value for the current model parameters.
        """

        # myLaggedXi_filtered = self.Xi_filtered.copy()
        myLaggedXi_filtered = self.Xi_t1_filtered.copy()
        myLaggedXi_filtered = np.roll(myLaggedXi_filtered, 1, axis=0)
        myLaggedXi_filtered[0,:] = self.UnconditionalStateProbs
        
        likelihood =  (myLaggedXi_filtered * self.Eta) @ np.ones((self.NumRegimes,1))
        
        # likelihood = np.sum(residual, axis=1)
        
        loglikelihood = sum(np.log(likelihood))
        return loglikelihood.item()