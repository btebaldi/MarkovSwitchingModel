# Carrega bibliotecas necessárias
import numpy as np
import pandas as pd
import scipy.stats as stats
import MarkovSwitchingModel

# M é o numero de Regimes
# K é o numero de variaveis exogenas (contando a constante)
# Beta é um vetor de coeficientes de dimensao (k x M)

class MK_estimator : 

    def __init__(self, model : MarkovSwitchingModel) :
        """
        Parameters:
            Y (array-like): The dependent variable data.
            X (array-like): The main independent variable(s).
            num_regimes (int): Number of regimes in the data/model.
            regimes_betas (array-like): Coefficient values (betas) per regime.
            regimes_omegas (array-like): Omega (variance or scale) per regime.
        """
        self.Model = model

    def Eta_t(self.model) -> np.ndarray:
        """
        Calcula as "quase densidades" para cada regime em um modelo de Markov Switching.
        """

        # Validation of inputs
        for regime_name, regime_params in model.num_regimes:
            # check dimensions of Betas vs X
            if X.shape[1] != regime_params["Beta"].shape[0]:
                raise ValueError(f"Número de variáveis exógenas em X não corresponde ao número de betas para {regime_name}.")
            
            # Check dimensions of Omega
            if regime_params["Omega"].ndim == 2 and regime_params["Omega"].shape[0] != regime_params["Omega"].shape[1]:
                raise ValueError(f"Regime {regime_name}: Omega deve ser uma matriz quadrada.")

            if not np.allclose(regime_params["Omega"], regime_params["Omega"].T):
                raise ValueError(f"Regime {regime_name}: Omega deve ser simétrica.")

            # Check dimensions of Omega vs Y
            if Y.shape[1] != regime_params["Omega"].shape[0]:
                raise ValueError(f"Regime {regime_name}: Número de variáveis dependentes em Y não corresponde a dimensão de omega.")

        # ToDo: function should be Multivariate Normal, not univariate
        Betas = np.column_stack([regime["Beta"] for regime in Regimes.values()])
        Omegas = np.column_stack([regime["Omega"] for regime in Regimes.values()])
        
        p = stats.norm.pdf(x = (Y - (X @ Betas)) / Omegas, loc=0, scale=1)
        # scipy.stats.multivariate_normal.pdf(x = Y, mean = X @ Betas, cov = Omegas)

        return p

    def LoadModel(self, filePath) : 
        # Read a CSV file
        df = pd.read_csv(filePath)
        # print(df.head())

        # df.describe(include='all')
        self.model.Y = df['Var1'].values.reshape((-1, 1))
        self.model.X = df[['Var2', 'Var3']].values.reshape((-1, 2))
        print(Y[0:3, :])
        print(X[0:3, :])

        print(X.shape)

    # def Xi_filter(Eta : ArrayLike, transitionMatrix : ArrayLike, unconditional_state_probs : Optional[ArrayLike] = None) -> np.ndarray:
    def Xi_filter(Eta : np.ndarray, transitionMatrix : np.ndarray, unconditional_state_probs  = None) -> np.ndarray:
        """
        Calcula o Xi filtrado (regime probabilities p(S_t | info até t) em um modelo de Markov Switching.
        
        Parâmetros:
        - Eta: array (T x M) com as "quase densidades" por regime no tempo t.
        - transitionMatrix: array (M x M) com as probabilidades de transição entre estados.
        - unconditional_state_probs: array (M,) com as probabilidades incondicionais dos estados (opcional).

        Retorna:
        - Xi: array (T x M) com as probabilidades filtradas por regime.
        """

        Eta = np.asarray(Eta, dtype=float)
        P = np.asarray(transitionMatrix, dtype=float)
        T, M = Eta.shape

        if unconditional_state_probs is None:
            unconditional_state_probs = np.ones(Eta.shape[1])/Eta.shape[1]

        # Validate dimensions
        if transitionMatrix.shape[0] != transitionMatrix.shape[1]:
            raise ValueError("Transition matrix must be square")
            
        if transitionMatrix.shape[0] != Eta.shape[1]:
            raise ValueError(f"Transition matrix dimension ({transitionMatrix.shape[0]}) must match number of regimes in Eta ({Eta.shape[1]})")
            
        if len(unconditional_state_probs) != Eta.shape[1]:
            raise ValueError(f"Unconditional state probabilities length ({len(unconditional_state_probs)}) must match number of regimes in Eta ({Eta.shape[1]})")

        Xi = np.full_like(Eta, 0.0)

        for i in range(Eta.shape[0]):
            if i == 0:
                Xi_t1 = unconditional_state_probs
            else:
                Xi_t1 = transitionMatrix.T @ Xi[i-1]
            
            # print("N=", Eta[i] * Xi_t1)
            # print("D=", Eta[i] @ Xi_t1)
            Xi[i] = (Eta[i] * Xi_t1)/(Eta[i] @ Xi_t1)

        return Xi

    def Xi_smooth(Xi_filtered, transitionMatrix) :
        # unconditional_state_probs = np.array([0.5, 0.5])

        Xi_s = Xi_filtered.copy()

        for i in range(Xi_filtered.shape[0]-2, 0, -1):
            Xi_s[i] = (transitionMatrix @ (Xi_s[i+1] / (transitionMatrix.T @ Xi_filtered[i]))) * Xi_filtered[i]

        return Xi_s

    def EstimateUnconditionalStateProbs(Xi_smoothed) :
        return Xi_smoothed.mean(axis=0)

    def EstimateTransitionMatrix(Xi_smoothed) :

        # get the number of regimes from Xi_smoothed
        M = Xi_smoothed.shape[1]

        # Create a zero matrix for transition counts
        transitionMatrix = np.zeros((M, M))

        #  Estimate the most likely state at each time point
        estimated_state = np.argmax(Xi_smoothed, axis=1)

        # Convert the estimation to a column vector
        estimated_state = estimated_state.reshape(-1, 1)

        # Calculate the unconditional state probabilities. This will be used to estimate the initial state probabilities.
        UnconditionalStateProbs = EstimateUnconditionalStateProbs(Xi_smoothed)

        # Estimate the initial state as the state with the highest unconditional probability
        estimated_initial_state = np.argmax(UnconditionalStateProbs, axis=0)

        # create lagged version of estimated_state, the first value is the estimated initial state
        lagged_state = np.roll(estimated_state, 1, axis=0)
        lagged_state[0] = estimated_initial_state

        #  create a new Xi_smoothed by adding estimated_state and lagged_state as new columns to Xi_smoothed
        Xi_smoothed_helper = np.hstack([
            Xi_smoothed,
            estimated_state,
            lagged_state
        ])

        # Goes through the transition probabilities positions and estimate them based on the smoothed probabilities
        for state_i in range(M):
            for state_j in range(M):

                # Count_ij : P(S_t = j, S_t-1 = i)
                Count_ij = Xi_smoothed_helper[(Xi_smoothed_helper[:, M] == state_j) & (Xi_smoothed_helper[:, M+1] == state_i),].shape[0]

                # Count_i : P(S_t-1 = i)
                Count_i = Xi_smoothed_helper[Xi_smoothed_helper[:, M+1] == state_i, ].shape[0]

                # Update the transition counts matrix
                transitionMatrix[state_i, state_j] = Count_ij / Count_i

        return transitionMatrix

    def EstimateBeta(Xi_smoothed,
                    Y : np.ndarray,
                    X : np.ndarray) -> np.ndarray :
        
        # Initialize Beta matrix to store estimated coefficients for each regime
        # Number of regimes from Xi_smoothed columns
        M = Xi_smoothed.shape[1]

        # Number of variables (columns in X)
        K = X.shape[1]
            
        # Initialize an empty array with K rows and 0 columns
        Betas_estimated = np.zeros((K, 0))
        
        #  For each regime run a OLS to estimate the regression
        for regime_number in range(Xi_smoothed.shape[1]):

            # Check dimensions of Omega
            Weights_Matrix = np.diag(Xi_smoothed[:, regime_number])

            A = (X.T @ Weights_Matrix @ X)

            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(A)

            B = A_inv @ X.T @ Weights_Matrix @ Y

            # para cada regime, empilhar os Betas baseado na estimacao.
            Betas_estimated = np.column_stack([Betas_estimated, B])
            # Betas_estimated = np.column_stack([Betas_estimated, B])
                    # np.kron(B, I)

        return Betas_estimated

    def EstimateResiduals(Y : np.ndarray, X : np.ndarray, Beta : np.ndarray) -> np.ndarray :

        # Number of variables (columns in X)
        K = X.shape[1]
        T = X.shape[0]
        
        residuals_estimated = Y - X @ Beta
        #     # Initialize an empty array with K rows and 0 columns
        #     residuals_estimated = np.zeros((T, 0))
        #     #  For each regime run a OLS to estimate the regression
        #     for regime_number in range(Beta.shape[1]):
        #         residuals = Y - X @ Beta[:, regime_number]
        #         # para cada regime, empilhar os Betas baseado na estimacao.
        #         residuals_estimated = np.column_stack([residuals_estimated, residuals])
        return residuals_estimated

    def EstimateOmega(Xi_smoothed : np.ndarray, residuals : np.ndarray) -> np.ndarray :
        
        # Initialize Beta matrix to store estimated coefficients for each regime
        # Number of regimes from Xi_smoothed columns
        M = Xi_smoothed.shape[1]
        
        Omega_estimated = np.zeros((T, 0))

        #  For each regime run a OLS to estimate the regression
        for regime_number in range(M):

            # Check dimensions of Omega
            Weights_Matrix = np.diag(Xi_smoothed[:, regime_number])

            Omega = (residuals[:, regime_number].T @ Weights_Matrix @ residuals[:, regime_number]) / sum(Xi_smoothed[:, regime_number])

            # para cada regime, empilhar os Betas baseado na estimacao.
            Omega_estimated = np.column_stack([Omega_estimated, Omega])
            # Betas_estimated = np.column_stack([Betas_estimated, B])
                    # np.kron(B, I)

        return Omega_estimated