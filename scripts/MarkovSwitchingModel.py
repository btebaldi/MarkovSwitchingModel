import numpy as np

class MarkovSwitchingModel:
    """
    A class to represent time series data and regime-switching model parameters.
    Implements a Markov regime-switching model for modeling behavior that changes
    across different economic or market regimes.
    """

    def __init__(self, Y, X, num_regimes, beta=None, omega=None,
                  transitionMatrix=None, unconditional_state_probs=None):
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
        
        # Validate Y input: reject 2-column Y (multivariate not supported)
        if len(Y.shape) == 2 and Y.shape[1] >= 2:
            raise NotImplementedError("Method not implemented for two Y columns")

        # Store the dependent variable (Y)
        self.Y = Y

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
            self.Beta = np.zeros((self.NumXVariables, self.NumRegimes))
        
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
            self.Omega = np.ones((1, self.NumRegimes))

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
            self.TransitionMatrix = np.ones((self.NumRegimes, self.NumRegimes)) / self.NumRegimes

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

            Weights_Matrix = np.diag(self.Xi_smoothed[:, regime_number])
            SSE += residuals[:, regime_number].T @ Weights_Matrix @ residuals[:, regime_number]
        
        return SSE

    def GetLogLikelihood(self) -> float:
        """
        Calculate the log-likelihood of the model given current parameters.
        
        The likelihood combines the SSE with state probabilities weighted by
        the smoothed regime probabilities across all observations.

        Returns:
            float: The log-likelihood value for the current model parameters.
        """
        loglikelihood = -self.GetSSE()
        for regime in range(self.NumRegimes):
            loglikelihood += np.sum(np.log(self.UnconditionalStateProbs[regime]) * self.Xi_smoothed[:, regime])
        return loglikelihood
