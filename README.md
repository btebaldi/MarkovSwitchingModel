# MarkovSwitchingModel
## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Overview

This implementation provides tools for estimating and analyzing Markov Switching Vector Autoregressive (VAR) models using Ordinary Least Squares (OLS) methods. The model is based on the theoretical framework presented in "OLS Estimation of Markov switching VAR models: asymptotics and application to energy use" by Maddalena Cavicchioli.

## Features

- Markov regime-switching VAR estimation (currently only univariative is suported)
- OLS-based parameter estimation
- Asymptotic inference and hypothesis testing


## Requirements

- Python 3.8+
- NumPy
- SciPy
- Pandas
- Statsmodels

See `requirements.txt` for specific versions.

## Documentation

For detailed documentation on the Markov Switching VAR model theory and implementation:

- See `docs/` directory for theoretical background
- Examples directory contains the data and results for two numerical implementations. To replicate the results one can execute the file `MarkovSwitchingExample.py`
    - TwoRegimes: DGP and output directory example with 2 regimes.
    - ThreeRegimes:  DGP and output directory example with 3 regimes.


## Getting Started


## 1) What model is implemented

The class `MarkovSwitchingModel` (present in `MarkovSwitchingModel.py`) represents a **Markov-switching regression** (a.k.a. regime-switching linear model):

* There is an unobserved regime/state $S_t \in {0,1,\dots,M-1}$
* The regime evolves as a **Markov chain** with transition matrix $P$
* Conditional on the regime, the observation follows a regression with regime-specific parameters

A typical specification consistent with your code is:
$$Y_t = X_t^\top \beta_{S_t} + \varepsilon_t,\qquad \varepsilon_t \sim \mathcal{N}(0,\omega_{S_t})$$

where:

* $Y_t$ is the dependent variable at time $t$
* $X_t$ is the vector of regressors at time $t$ (intercept, trend, exogenous, lags)
* $\beta_{S_t}$ is the regime-specific coefficient vector (one column per regime)
* $\omega_{S_t}$ is the regime-specific variance

The class `MarkovSwitchingModel` has the following information:

* `Eta`: regime “quasi-densities” $f(Y_t \mid S_t=m)$ for each regime and time
* `Xi_filtered`: filtered probabilities $P(S_t=m \mid \mathcal{I}_t)$
* `Xi_smoothed`: smoothed probabilities $P(S_t=m \mid \mathcal{I}_T)$

* `Y`: A vector (`np.ndarray`) containing the information of the dependent variable. Tipacly a $T \times 1$ vector (The code can be adapted to hande mode dependent variavbles in a VAR maner). Y can not contains missing values.
* `X`: A vector (`np.ndarray`) containing the information of the independent variables. Tipacly a $T \times K$ where $K$ is the total number of regressor includint (intercept, trend, exogenous, lags). X can not contains missing values.

* `NumRegimes`: An integer (`int`) representing the number of regimes ($M$). The number of regimes must be greater than 1.
* `NumObservations` : Number of time observations in the model ($T$)

* `beta`: A vector (`np.ndarray`) of regime-specific coefficients of dimensions $K \times M$. If this variable is set to `None` the model is initialized as a random vector.

* `omega`: A vector (`np.ndarray`) of dimensions $(1 \times M)$ with regime-specific variance parameters (Currently the code uses univariate Y only). Must be strictly positive. If this variable is set to `None`, it is initialized as a random vector.


self, Y: np.ndarray, X: np.ndarray, num_regimes:int,
                 beta: np.ndarray|None =None,
                 omega: np.ndarray|None = None,
                 transitionMatrix: np.ndarray|None = None,
                 unconditional_state_probs: np.ndarray|None = None,
                 param_names: dict | None = None,
                 dates_label: list[datetime] | None = None, model_name=None):

✅ Good:

```python
Y = df["y"].to_numpy().reshape(-1, 1)
```

❌ Bad:

```python
Y = df["y"].to_numpy()           # shape (T,) -> will break later
Y = df[["y1","y2"]].to_numpy()   # shape (T,2) -> explicitly rejected
```

## 3) Optional parameters and how to pass them

---

```python
self.Omega = ones((1, M)) + noise
```

---

* `transitionMatrix: np.ndarray | None`: A Vector () of Markov transition probabilities $P_{i,j} = P(S_t=j \mid S_{t-1}=i)$. This is a matrix of dimensions $M \times M$ where the probabilities for each row must sum to unit. If not provided, it creates a default “sticky” chain with 0.9 on the diagonal and $0.1/(M-1)$ off-diagonal

* `UnconditionalStateProbs: np.ndarray | None`: A vector () of Initial regime Unconditional State Probabilities distribution. A vector of dimension $1 \times M$. If not provided the model initializes it as $1/M$ for each state.

* `param_names: dict | None`: A dictonary with Metadata about the model. This is used only for **reporting, labeling, and prediction logic**.


### param_names
This file representa the Metadata information about the model. It is a dictornary with the following Expected structure:

```python
class TypeOfDependentVariable(Enum):
    INTERCEPT = 1
    TREND = 2
    AUTO_REGRESSIVE = 3
    EXOGENOUS = 4

class TypeOfVariable(Enum):
    DEPENDENT = 0
    INDEPENDENT = 1

class TypeOfTransformation(Enum):
    NONE = 0
    LEVEL = 1
    LOG_DIFF = 2
    STANDARIZE = 3

{
    # information about the dependent variable
  "Y": {0:  { # first dependent variable
        "Name" : "MyFirstYVariable",
        "Type": TypeOfVariable.DEPENDENT,
        "ClassOfRegressor": None,
        "Transformation": TypeOfTransformation.NONE,
        "AR": None},
        # second dependent variable (the program currently only suport univatiave information, however this would be the addaptation needed for a VAR structure )
        # 1:  { 
        # "Name" : "MySeccondYVariable",
        # "Type": TypeOfVariable.DEPENDENT,
        # "ClassOfRegressor": None,
        # "Transformation": TypeOfTransformation.NONE,
        # "AR": None},
        
    },
  "X": {0:  { # first independent variable
        "Name" : "Intercempt",
        "Type": TypeOfVariable.INDEPENDENT,
        "ClassOfRegressor": TypeOfDependentVariable.INTERCEPT,
        "Transformation": TypeOfTransformation.NONE,
        "AR": None},
        0:  { # first dependent variable
        "Name" : "MyFirstYVariable",
        "Type": TypeOfVariable.INDEPENDENT,
        "ClassOfRegressor": None,
        "Transformation": TypeOfTransformation.NONE,
        "AR": None}
        0:  { # first dependent variable
        "Name" : "MyFirstYVariable",
        "Type": TypeOfVariable.INDEPENDENT,
        "ClassOfRegressor": None,
        "Transformation": TypeOfTransformation.NONE,
        "AR": None}
        0:  { # first dependent variable
        "Name" : "MyFirstYVariable",
        "Type": TypeOfVariable.INDEPENDENT,
        "ClassOfRegressor": None,
        "Transformation": TypeOfTransformation.NONE,
        "AR": None}
        0:  { # first dependent variable
        "Name" : "MyFirstYVariable",
        "Type": TypeOfVariable.INDEPENDENT,
        "ClassOfRegressor": None,
        "Transformation": TypeOfTransformation.NONE,
        "AR": None} 1: {...}, ..., K-1: {...}}
}
```

Key requirement:

* Must include `"Y"` and `"X"`
* `"X"` must be a dict with exactly `K` elements (same as number of columns in X)

You construct these entries using:

```python
GetDictRepresentation(
    name="Intercept",
    type=TypeOfVariable.INDEPENDENT,
    classX=TypeOfDependentVariable.INTERCEPT,
    transformation=TypeOfTransformation.NONE,
    ar=None
)
```

How it’s used later:

* `Summary()` prints variable names from `ParamNames`
* Your `Predict()` function (in estimator) uses `ClassOfRegressor` and `AR` to generate future X values (intercept=1, trend increments, AR uses lag of Y, exogenous is not implemented)

✅ Example for X with [Intercept, Lag_1]:

```python
param_names = GetEmptyParamNames()
param_names["Y"][0] = GetDictRepresentation(
    name="Close",
    type=TypeOfVariable.DEPENDENT,
    transformation=TypeOfTransformation.LOG_DIFF
)

param_names["X"][0] = GetDictRepresentation(
    name="Intercept",
    type=TypeOfVariable.INDEPENDENT,
    classX=TypeOfDependentVariable.INTERCEPT
)
param_names["X"][1] = GetDictRepresentation(
    name="Lag_1",
    type=TypeOfVariable.INDEPENDENT,
    classX=TypeOfDependentVariable.AUTO_REGRESSIVE,
    ar=1
)
```

If `param_names=None`, the constructor creates generic names: “Exogenous Variable i”.

---

### `dates_label: list[datetime] | pd.DatetimeIndex | None`

Used for plotting and reporting date range.

Requirements:

* length must equal T (number of observations)
* must be list-like of `date`/`datetime` or a `DatetimeIndex`

If not provided, it uses numeric index `0..T-1`.

✅ Example:

```python
dates = df.index  # DatetimeIndex after setting index_col
model = MarkovSwitchingModel(Y, X, num_regimes=3, dates_label=dates)
```

---

### `model_name`

Only used for printing.

✅ Example:

```python
model = MarkovSwitchingModel(Y, X, num_regimes=3, model_name="NASDAQ MS-AR(1)")
```

---
























### Basic Example

```python
import numpy as np
from markov_switching_model import MSVARModel

# Load your data (multivariate time series)
data = np.array([...])  # Shape: (n_observations, n_variables)

# Create and fit model
model = MSVARModel(data, n_lags=2, n_regimes=2)
model.fit()

# Generate forecasts
forecast = model.predict(steps=10, regime_probabilities=True)

# Estimate hidden states
states = model.viterbi()
smoothed_probs = model.smooth()
```


## License

See LICENSE file for details.





## Installation

```bash
git clone https://github.com/yourusername/MarkovSwitchingModel.git
cd MarkovSwitchingModel
pip install -r requirements.txt
```

## Usage

```python
from markov_switching_model import MSVARModel

# Initialize model with specification
model = MSVARModel(data, n_lags=2, n_regimes=2)

# Estimate parameters
model.fit()

# Predict and infer hidden states
predictions = model.predict(steps=10)
states = model.viterbi()
```

## Structure

- `src/` - Core implementation modules
- `examples/` - Usage examples and case studies
- `tests/` - Unit tests
- `docs/` - Documentation and references

## References

Cavicchioli, M. (Year). "OLS Estimation of Markov switching VAR models: asymptotics and application to energy use."

### Citation

To cite this repository, files, or associated materials, use the format shown below, adapting it as needed for the context of your work.

Barbosa, B. T. (2025). *Markov Switching Model for Python using OLS Estimation*. https://github.com/btebaldi/MarkovSwitchingModel.

> @misc{barbosa2025MarkovSwitchingModelPythonOLS, 
author = {Barbosa, Bruno Tebaldi}, 
title = {Markov Switching Model for Python using OLS Estimation}, 
year = {2025}, 
howpublished = {\url{https://github.com/btebaldi/MarkovSwitchingModel}} 
}


Here’s how your **MarkovSwitchingModel** works, what each variable means, and **exactly how to pass** each argument so the model is valid and the estimator can run.

---


## 4) What happens internally after construction

When you instantiate the class:

1. **Validates input** (no NaNs, correct shapes, num_regimes>1).
2. Stores `Y`, `X`, and counts:

   * `NumObservations = T`
   * `NumXVariables = K`
   * `NumRegimes = M`
3. Initializes parameters:

   * `Beta` (K×M)
   * `Omega` (1×M)
   * `TransitionMatrix` (M×M)
   * `UnconditionalStateProbs` (M,)
4. Allocates arrays that will be filled during estimation:

   * `Eta` (T×M)
   * `Xi_filtered` (T×M)
   * `Xi_t1_filtered` (T×M)
   * `Xi_smoothed` (T×M)

**Important:** the model **does not estimate by itself**.
Estimation happens in your `MarkovSwitchingEstimator`, which updates those arrays.

---

## 5) Methods and what they assume

### `GetResiduals()`

Computes:

```python
self.Y - self.X @ self.Beta
```

⚠️ Shape issue: `X @ Beta` is `(T,K) @ (K,M) = (T,M)`
But `Y` is `(T,1)`. Broadcasting gives `(T,M)` residuals, one per regime, which is what you want.

---

### `GetSSE()`

Weighted SSE across regimes:
[
\sum_{m=0}^{M-1}\sum_{t=1}^{T} \xi_{t,m}^{(smoothed)} \cdot e_{t,m}^2
]
So it assumes:

* `Xi_smoothed` has been computed by the estimator

If you call it before fitting, `Xi_smoothed` is all zeros → SSE becomes 0.

---

### `GetLogLikelihood()`

Computes:
[
\ell = \sum_t \log\left( \sum_m \eta_{t,m} \cdot \xi_{t|t-1,m} \right)
]

Where:

* `Eta[t,m]` must be set (from likelihood evaluation)
* `Xi_t1_filtered` must be set (predicted probs ( \xi_{t|t-1} ))

Again, if you call this before fitting, `Eta`/`Xi_t1_filtered` are zeros → log(0) problems.

---

## 6) Minimal example: “correct” instantiation

```python
T = 100
M = 2

# Example data
Y = np.random.randn(T, 1)

# Regressors: intercept + AR(1) lag (dummy for illustration)
X = np.column_stack([np.ones(T), np.random.randn(T)])
K = X.shape[1]

beta0  = np.zeros((K, M))
omega0 = np.array([[0.5, 1.5]])

param_names = GetEmptyParamNames()
param_names["Y"][0] = GetDictRepresentation("Y", TypeOfVariable.DEPENDENT)
param_names["X"][0] = GetDictRepresentation("Intercept", TypeOfVariable.INDEPENDENT, TypeOfDependentVariable.INTERCEPT)
param_names["X"][1] = GetDictRepresentation("Lag_1", TypeOfVariable.INDEPENDENT, TypeOfDependentVariable.AUTO_REGRESSIVE, ar=1)

model = MarkovSwitchingModel(
    Y=Y,
    X=X,
    num_regimes=M,
    beta=beta0,
    omega=omega0,
    param_names=param_names,
    dates_label=None,
    model_name="Demo MS model"
)
```

---

If you want, I can also explain (in the same level of detail) **how the estimator updates** `Eta`, `Xi_filtered`, `Xi_smoothed`, `Beta`, `Omega`, and `TransitionMatrix` step-by-step, mapping each line of your estimator code to the Hamilton/Kim equations.
