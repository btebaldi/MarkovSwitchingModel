# Load necessary libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import MarkovSwitchingModel as MKM

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
                print(f"Iteration {interationCounter}: Estimated LogLikelihood {cur_ll:.6f} with change {delta:9.6f} - {self.Model.LogLikeOx():9.6f}")
            delta = abs(cur_ll - prev_ll)
            
    

def LoadModel(filePath):
    """
    Loads the Markov Switching Model from a CSV file.

    Parameters:
        filePath (str): Path to the CSV file containing the data.

    Returns:
        model (MKM.MarkovSwitchingModel): An instance of the Markov Switching Model.
    """
    # Read a CSV file
    # df = pd.read_csv(filePath)
    # df = pd.read_csv("./database/resultados.csv", decimal=',', sep=';')
    df = pd.read_csv("c:\\Users\\bteba\\Downloads\\dadosregimes3.csv", decimal=',', sep=';')

    # Extract dependent and independent variables
    # Y = df['Var1'].to_numpy().reshape((-1, 1))  # Dependent variable
    
    # Y = np.log(df['Close_filled'].to_numpy().reshape((-1, 1)))  # Dependent variable
    Y = df['Y'].to_numpy().reshape((-1, 1))  # Dependent variable
    

    # X = df[['Var2', 'Var3']].to_numpy().reshape((-1, 2))  # Independent variables

    # Create a lagged version of the dependent variable Y
    # X = np.column_stack([np.ones(Y.shape[0]), df['x'].to_numpy()])
    X = np.column_stack([df['X'].to_numpy(), df['Y1'].to_numpy()])
    # X[0,1] = np.nan  # Set the first value to NaN since it has no lagged value
    
    # Y = Y[2:, ]#.to_numpy().reshape((-1, 1))  # Remove the first row with NaN        
    # X = X[2:, ]#.to_numpy().reshape((-1, 2)) 

    # Combine the original and lagged dependent variable into the independent variables

    B = np.array([[1, 2, 3], [0.9, 0.5, 0.2]])  # Initial beta values

    # Create an instance of the Markov Switching Model
    model = MKM.MarkovSwitchingModel(Y, X, num_regimes=3, beta=B)

    return model

def LoadModel_IBOV():
    df = pd.read_csv(".\\database\\filled\\IBOV_filled.csv", decimal='.', sep=',')

    # Extract dependent and independent variables
    # df['log_return'] = np.log(df['Close_filled'] / df['Close_filled'].shift(1))
    # df['log_return2'] = df['log_return'].shift(1)
    # df = df.iloc[2:, :]
    # Y = df['log_return'].to_numpy().reshape((-1, 1))  # Dependent variable
    # X = np.column_stack([np.ones(Y.shape[0]), df['log_return2'].to_numpy()])

    #  Dependent variable
    df['lag_Close_filled'] = df['Close_filled'].shift(1)
    df = df.iloc[1:, :]
    Y = np.log(df['Close_filled'].to_numpy().reshape((-1, 1))) 
    trend = np.arange(Y.shape[0])
    intercept = np.ones(Y.shape[0])
    X = np.column_stack([intercept, trend, np.log(df['lag_Close_filled'].to_numpy())])
    


    # Create an instance of the Markov Switching Model
    param_names = {'Y':'Close_filled', 'X':['Intercept', 'Trend', 'Lagged_Close']}

    TransProb = np.array([[0.950, 0.030, 0.020],
                          [0.025, 0.950, 0.025],
                          [0.040, 0.010, 0.950]])
    
    # B = np.array([[0.01, 0.02, 0.03], [0.1, 0.7, 0.1], [0.3, 0.1, 0.2]])  # Initial beta values
    B = np.array([[0.25752, 0.069911, 0.13291],
                  [2.3547e-03, 2.8407e-03, 2.0194e-03],
                  [0.97664, 0.99355, 0.98788]])

    Om = np.array([[0.017421**2, 0.017353**2, 0.010883**2]])  # Initial omega values

    model = MKM.MarkovSwitchingModel(Y, X, num_regimes=3, beta=B, param_names=param_names, transitionMatrix=TransProb, omega=Om)


    return model



if __name__ == "__main__":
    # Load the model from a CSV file
    # model = LoadModel(filePath='./dev/matrizYX.csv')
    model = LoadModel_IBOV()
    
    # Create an estimator instance
    a = MarkovSwitching_estimator(model)
    
    # Estimate quasi-densities
    # a.EstimateEta()
    # Print the estimated quasi-densities
    # print(a.Model.Eta)
    
    # Estimate filtered probabilities
    # a.EstimateXiFiltered()
    # Print the filtered probabilities
    # print(a.Model.Xi_filtered) 
    
    # print(a.Model.Xi_t1_filtered) 
    # Estimate smoothed probabilities
    # a.EstimateXiSmoothed()
    # print(a.Model.Xi_smoothed) 

    # a.EstimateTransitionMatrix()
    # a.EstimateBeta()
    # a.EstimateOmega()


    # Print the smoothed probabilities
    # print(a.Model.Xi_smoothed)
    # print(a.Model.GetSSE())
    a.Fit(traceLevel=1, precision=1e-4)
    print(a.Model)
    print("Finished Estimation")