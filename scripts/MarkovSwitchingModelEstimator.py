# Carrega bibliotecas necessárias
import numpy as np
import pandas as pd
import scipy.stats as stats
import MarkovSwitchingModel as MKM

# M é o numero de Regimes
# K é o numero de variaveis exogenas (contando a constante)
# Beta é um vetor de coeficientes de dimensao (k x M)

class MK_estimator : 

    def __init__(self, model : MKM.MarkovSwitchingModel) :
        """
        Parameters:
            Y (array-like): The dependent variable data.
            X (array-like): The main independent variable(s).
            num_regimes (int): Number of regimes in the data/model.
            regimes_betas (array-like): Coefficient values (betas) per regime.
            regimes_omegas (array-like): Omega (variance or scale) per regime.
        """
        self.Model = model

    def EstimateEta(self) -> np.ndarray:
        """
        Calcula as "quase densidades" para cada regime em um modelo de Markov Switching.
        """
        # ToDo: function should be Multivariate Normal, not univariate
        # Betas = np.column_stack([model.Beta regime["Beta"] for regime in Regimes.values()])
        # Omegas = np.column_stack([regime["Omega"] for regime in Regimes.values()])
        
        self.Model.Eta = stats.norm.pdf(x = (self.Model.Y - (self.Model.X @ self.Model.Beta)) / self.Model.Omega, loc=0, scale=1)
        # scipy.stats.multivariate_normal.pdf(x = Y, mean = X @ Betas, cov = Omegas)

        return self.Model.Eta

    # def Xi_filter(self, Eta : np.ndarray, transitionMatrix : np.ndarray, unconditional_state_probs  = None) -> np.ndarray:
    def EstimateXiFiltered(self) -> np.ndarray:
        """
        Calcula o Xi filtrado (regime probabilities p(S_t | info até t) em um modelo de Markov Switching.
        
        Parâmetros:
        - Eta: array (T x M) com as "quase densidades" por regime no tempo t.
        - transitionMatrix: array (M x M) com as probabilidades de transição entre estados.
        - unconditional_state_probs: array (M,) com as probabilidades incondicionais dos estados (opcional).

        Retorna:
        - Xi: array (T x M) com as probabilidades filtradas por regime.
        """

        # Eta = np.asarray(Eta, dtype=float)
        # P = np.asarray(transitionMatrix, dtype=float)
        # T, M = Eta.shape

        # if unconditional_state_probs is None:
        #     unconditional_state_probs = np.ones(Eta.shape[1])/Eta.shape[1]

        # if len(unconditional_state_probs) != Eta.shape[1]:
        #     raise ValueError(f"Unconditional state probabilities length ({len(unconditional_state_probs)}) must match number of regimes in Eta ({Eta.shape[1]})")

        # Xi = np.full_like(Eta, 0.0)

        for i in range(model.NumObservations):
            if i == 0:
                Xi_t1 = model.UnconditionalStateProbs
            else:
                Xi_t1 = model.TransitionMatrix.T @ model.Xi_filtered[i-1]
            #  TODO: Check if we are doing the riht matrix multiplication here
            model.Xi_filtered[i] = (model.Eta[i] * Xi_t1)/(model.Eta[i] @ Xi_t1)

        return model.Xi_filtered

    def EstimateXiSmoothed(self) -> np.ndarray:

        Xi_s = self.Model.Xi_filtered.copy()

        for i in range(self.Model.NumObservations-2, 0, -1):
            Xi_s[i] = (self.Model.TransitionMatrix @ (Xi_s[i+1] / (self.Model.TransitionMatrix.T @ self.Model.Xi_filtered[i]))) * self.Model.Xi_filtered[i]

        self.Model.Xi_smoothed = Xi_s
        return self.Model.Xi_smoothed

    def EstimateUnconditionalStateProbs(self) -> np.ndarray:
        return self.Model.Xi_smoothed.mean(axis=0)

    def EstimateTransitionMatrix(self) -> np.ndarray :

        # get the number of regimes from Xi_smoothed
        # M = Xi_smoothed.shape[1]

        # Create a zero matrix for transition counts
        transitionMatrix = np.zeros((self.Model.NumRegimes, self.Model.NumRegimes))

        #  Estimate the most likely state at each time point
        estimated_state = np.argmax(self.Model.Xi_smoothed, axis=1)

        # Convert the estimation to a column vector
        estimated_state = estimated_state.reshape(-1, 1)

        # Calculate the unconditional state probabilities. This will be used to estimate the initial state probabilities.
        UnconditionalStateProbs = self.EstimateUnconditionalStateProbs()

        # Estimate the initial state as the state with the highest unconditional probability
        estimated_initial_state = np.argmax(UnconditionalStateProbs, axis=0)

        # create lagged version of estimated_state, the first value is the estimated initial state
        lagged_state = np.roll(estimated_state, 1, axis=0)
        lagged_state[0] = estimated_initial_state

        #  create a new Xi_smoothed by adding estimated_state and lagged_state as new columns to Xi_smoothed
        Xi_smoothed_helper = np.hstack([
            self.Model.Xi_smoothed,
            estimated_state,
            lagged_state
        ])

        # Goes through the transition probabilities positions and estimate them based on the smoothed probabilities
        for state_i in range(self.Model.NumRegimes):
            for state_j in range(self.Model.NumRegimes):

                # Count_ij : P(S_t = j, S_t-1 = i)
                Count_ij = Xi_smoothed_helper[(Xi_smoothed_helper[:, self.Model.NumRegimes] == state_j) & (Xi_smoothed_helper[:, self.Model.NumRegimes+1] == state_i),].shape[0]

                # Count_i : P(S_t-1 = i)
                Count_i = Xi_smoothed_helper[Xi_smoothed_helper[:, self.Model.NumRegimes+1] == state_i, ].shape[0]

                # Update the transition counts matrix
                transitionMatrix[state_i, state_j] = Count_ij / Count_i

        return transitionMatrix

    def EstimateBeta(self) -> np.ndarray :
        
        # Initialize Beta matrix to store estimated coefficients for each regime
        # Number of regimes from Xi_smoothed columns
        # M = Xi_smoothed.shape[1]

        # Number of variables (columns in X)
        # K = X.shape[1]
            
        # Initialize an empty array with K rows and 0 columns
        Betas_estimated = np.zeros((self.Model.NumXVariables, 0))
        
        #  For each regime run a OLS to estimate the regression
        for regime_number in range(self.Model.NumRegimes):

            # Check dimensions of Omega
            Weights_Matrix = np.diag(self.Model.Xi_smoothed[:, regime_number])

            A = (self.Model.X.T @ Weights_Matrix @ self.Model.X)

            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(A)

            B = A_inv @ self.Model.X.T @ Weights_Matrix @ self.Model.Y

            # para cada regime, empilhar os Betas baseado na estimacao.
            Betas_estimated = np.column_stack([Betas_estimated, B])
            # Betas_estimated = np.column_stack([Betas_estimated, B])
                    # np.kron(B, I)

        self.Model.Beta = Betas_estimated   
        return Betas_estimated

    def EstimateResiduals(self) -> np.ndarray :

        return self.Model.GetResiduals()

    def EstimateOmega(self) -> np.ndarray :
        
        # Initialize Beta matrix to store estimated coefficients for each regime
        # Number of regimes from Xi_smoothed columns
        # M = Xi_smoothed.shape[1]
        
        Omega_estimated = np.zeros((1, self.Model.NumRegimes))

        #  For each regime run a OLS to estimate the regression
        for regime_number in range(self.Model.NumRegimes):

            # Check dimensions of Omega
            Weights_Matrix = np.diag(self.Model.Xi_smoothed[:, regime_number])

            residuals = self.Model.GetResiduals()

            Omega = (residuals[:, regime_number].T @ Weights_Matrix @ residuals[:, regime_number]) / sum(self.Model.Xi_smoothed[:, regime_number])

            # para cada regime, empilhar os Betas baseado na estimacao.
            Omega_estimated = np.column_stack([Omega_estimated, Omega])

        return Omega_estimated
    


def LoadModel(filePath) : 
    # Read a CSV file
    df = pd.read_csv(filePath)

    # df.describe(include='all')
    Y = df['Var1'].values.reshape((-1, 1))
    X = df[['Var2', 'Var3']].values.reshape((-1, 2))

    B = np.array([[1,1], [1,2]])

    model = MKM.MarkovSwitchingModel(Y, X, num_regimes=2, beta=B)
    # def __init__(self, Y, X, num_regimes, beta=None, omega=None):

    return model



if __name__ == "__main__":

    model = LoadModel(filePath = './dev/matrizYX.csv')
    
    a =  MK_estimator(model)
    a.EstimateEta()
    print(a.Model.Eta)
    a.EstimateXiFiltered()
    print(a.Model.Xi_filtered)
    a.EstimateXiSmoothed()
    print(a.Model.Xi_smoothed)

# # Read a CSV file
# df = pd.read_csv('matrizYX.csv')
# print(df.head())

# # df.describe(include='all')
# Y = df['Var1'].values.reshape((-1, 1))
# X = df[['Var2', 'Var3']].values.reshape((-1, 2))
# print(Y[0:3, :])
# print(X[0:3, :])

# print(X.shape)
#     # Read daily data
#     dic_files =  Load_configuration(file_path=None)

#     for key, config in dic_files.items( ) :
#         print(f"Processing configuration for {key}...")
    
#         # check all necessary keys are present
#         required_keys = ['filePath', 'fileName',
#                          'saveFilePath', 'saveFileName',
#                          'delimiter', 'decimal', 'parse_dates',
#                         'date_format', 'index', 'aggregation']
        
#         if not all(k in config for k in required_keys):
#             print(f"Configuration for {key} is missing required keys.")
#             raise ValueError(f"Missing required keys.")

#         # read daily data
#         daily_data = read_daily_database(config)
        
#         # Create aggregations
#         weekly_data = create_weekly_database(daily_data, aggregation = config['aggregation'])
#         monthly_data = create_monthly_database(daily_data, aggregation = config['aggregation'])
        
#         # Check if save directory exists, create if not
#         if not os.path.exists(config['saveFilePath']):
#             os.makedirs(config['saveFilePath'])
        
#         # Save results
#         save_database(weekly_data, os.path.join(config['saveFilePath'], "weekly/", config["saveFileName"]) )
#         save_database(monthly_data, os.path.join(config['saveFilePath'], "monthly/", config["saveFileName"]) )