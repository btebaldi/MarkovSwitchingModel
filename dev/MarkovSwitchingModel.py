import numpy as np

class MarkovSwitchingModel:
    """
    A class to represent time series data and regime-switching model parameters.
    """

    def __init__(self, Y, X, num_regimes, beta=None, omega=None):
        """
        Initialize a MarkovSwitchingModel instance with data and parameters.

        Parameters:
            Y (array-like): The dependent variable data.
            X (array-like): The main independent variable(s).
            num_regimes (int): Number of regimes (states) in the data/model.
            beta (array-like, optional): Coefficient values (betas) per regime.
            omega (array-like, optional): Omega (variance or scale) per regime.
        """
        # Check for unsupported cases: raise if input Y has two columns
        if len(Y.shape) == 2 and Y.shape[1] == 2:
            # Not implemented for 2-dimensional Y responses
            raise NotImplementedError("Method not implemented for two Y columns")

        # Store the dependent variable (Y)
        self.Y = Y

        # Store the independent variables (X)
        self.X = X

        # Overwrites self.NumXVariables with the actual shape from X (intended)
        self.NumXVariables = self.X.shape[1]

        # Ensure number of observations matches between X and Y
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must have the same number of observations (rows).")

        # Store number of data observations (rows)
        self.NumObservations = self.X.shape[0]

        # Store number of regimes in the model (as an attribute)
        self.NumRegimes = num_regimes

        # Handle regime-specific beta coefficients
        if beta is not None:
            # beta must be an array of shape (num_variables, num_regimes)
            # Check for correct shape: number of columns in beta must match num_regimes
            if beta.shape[1] != self.NumRegimes:
                raise ValueError("Number of columns in betas must be equal to num_regimes.")
            # check to see if the beta dimension mach the num variables
            
            self.Beta = beta
        else:
            # If not provided, initialize betas to zeros
            # Assumes X already contains the constant in the first column
            self.Beta = np.zeros((self.NumXVariables, self.NumRegimes))
        
        # Handle regime-specific omega (variance/scale)
        if omega is not None:
            # Ensure correct shape for omega: (1, num_regimes)
            if omega.shape[1] != self.NumRegimes:
                raise ValueError("Number of columns in omegas must be equal to num_regimes.")
            if omega.shape[0] != 1:
                raise ValueError("Omegas must have only one dimension (line)")
            self.Omega = omega
        else:
            # If not provided, initialize omega to zeros
            self.Omega = np.zeros((1, self.NumRegimes))

