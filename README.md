# MarkovSwitchingModel
## Table of Contents

- [Overview](#overview)
- [Citation](#citation)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Getting Started](#getting-started)
- [Basic Example](#basic-example)
- [License](#license)
- [References](#references)

## Overview

This implementation provides tools for estimating and analyzing Markov Switching Vector Autoregressive (VAR) models using Ordinary Least Squares (OLS) methods. The model is based on the theoretical framework presented in "OLS Estimation of Markov switching VAR models: asymptotics and application to energy use" by Maddalena Cavicchioli.

## Citation

To cite this repository, files, or associated materials, use the format shown below, adapting it as needed for the context of your work.

Barbosa, B. T. (2025). *Markov Switching Model for Python using OLS Estimation*. https://github.com/btebaldi/MarkovSwitchingModel.

> @misc{barbosa2025MarkovSwitchingModelPythonOLS, 
author = {Barbosa, Bruno Tebaldi}, 
title = {Markov Switching Model for Python using OLS Estimation}, 
year = {2025}, 
howpublished = {\url{https://github.com/btebaldi/MarkovSwitchingModel}} 
}


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

## Project Structure

The repository is organized to clearly separate **core logic**, **data**, **examples**, and **outputs**, making the project easier to maintain, extend, and use:

* `scripts/` — Core implementation modules
  Contains the main source code, including model definitions, estimation procedures, and helper utilities.

* `databases/` — Data storage
  Holds raw and processed datasets used by the models and examples.

* `examples/` — Usage examples and case studies
  Provides illustrative scripts and notebooks demonstrating how to configure, estimate, and interpret the models in practice.

* `output/` — Output directory (recommended)
  Intended location for generated results such as estimation summaries, forecasts, and plots.

* `docs/` — Documentation and references
  Includes technical documentation, methodological notes, and relevant academic or applied references.
  

## Documentation

For detailed information on the **theory, implementation, and practical use** of the Markov Switching (VAR) model, please refer to the following resources:

* `docs/` directory
  Contains the theoretical background, methodological notes, and implementation details underlying the Markov Switching framework.

* `examples/` directory
  Includes data, scripts, and outputs for two complete numerical implementations.
  To replicate the reported results, simply run the script:

  ```bash
  MarkovSwitchingExample.py
  ```

  The examples are organized as follows:

  * `TwoRegimes/` — Data-generating process (DGP) and output files for a model with two regimes.
  * `ThreeRegimes/` — Data-generating process (DGP) and output files for a model with three regimes.

These materials are intended to facilitate both conceptual understanding and hands-on replication of the Markov Switching models implemented in this project.

## Getting Started

A **Markov-switching regression** consists in :

* An unobserved regime/state $S_t \in {0,1,\dots,M-1}$
* The regime evolves as a **Markov chain** with transition matrix $P$
* Conditional on the regime, the observation follows a regression with regime-specific parameters

A typical specification consistent with your code is:
$$Y_t = X_t^\top \beta_{S_t} + \varepsilon_t,\qquad \varepsilon_t \sim \mathcal{N}(0,\omega_{S_t})$$

where:

* $Y_t$ is the dependent variable at time $t$
* $X_t$ is the vector of regressors at time $t$ (intercept, trend, exogenous, lags)
* $\beta_{S_t}$ is the regime-specific coefficient vector (one column per regime)
* $\omega_{S_t}$ is the regime-specific variance

### The Model

The codebase is organized into **three main modules**, each responsible for a distinct aspect of the Markov Switching framework: model representation, estimation, and post-estimation analysis.

* `MarkovSwitchingModel.py`
  Defines the **core representation of the Markov Switching model**.
  This module stores all structural elements of the model, including:

  * The dependent variable $Y$ and regressor matrix $X$
  * Regime-specific parameters (coefficients and variances)
  * State probabilities (filtered and smoothed)
  * The regime transition matrix and unconditional state probabilities
    It serves as the central data container used throughout estimation and inference.

* `MarkovSwitchingEstimator.py`
  Implements the **estimation and prediction procedures** for the model.
  This module is responsible for:

  * Computing regime likelihoods and probabilities
  * Performing filtering and smoothing
  * Iteratively estimating model parameters
  * Generating out-of-sample forecasts based on the estimated dynamics

* `MarkovSwitchingModelHelper.py`
  Provides **auxiliary tools for interpretation and visualization**.
  This module includes utilities to:

  * Extract and summarize regime classifications
  * Generate descriptive regime names
  * Produce plots of smoothed regime probabilities and other diagnostic outputs

Together, these three modules form a complete workflow for specifying, estimating, and analyzing Markov Switching models in an applied time-series setting.

### Class MarkovSwitchingModel

The `MarkovSwitchingModel` class stores the core data, parameters, and state-probability objects required for estimation and inference:

* `Eta`: regime “quasi-densities” $f(Y_t \mid S_t = m)$ computed for each time $t$ and regime $m$.

* `Xi_filtered`: filtered regime probabilities $P(S_t = m \mid \mathcal{I}_t)$, i.e., using information available up to time $t$.

* `Xi_smoothed`: smoothed regime probabilities $P(S_t = m \mid \mathcal{I}_T)$, i.e., using the full sample information up to time $T$.

* `Y`: the dependent variable as a NumPy array, typically a $T \times 1$ vector. (The structure could be extended to support multiple dependent variables in a VAR-like setup.) **`Y` must not contain missing values.**

* `X`: the regressor matrix as a NumPy array, typically $T \times K$, where $K$ is the total number of regressors (e.g., intercept, trend, exogenous variables, and lag terms). **`X` must not contain missing values.**

* `NumRegimes`: an integer representing the number of regimes $M$. It must satisfy $M > 1$.

* `NumObservations`: the number of time observations $T$.

* `Beta`: the regime-specific coefficient matrix, with dimensions $K \times M$. If `None`, the model initializes `Beta` with small random values.

* `Omega`: the regime-specific variance parameters, with dimensions $1 \times M$ (currently assuming univariate $Y$). All entries must be strictly positive. If `None`, `Omega` is initialized with random positive values.

* `TransitionMatrix`: The Markov state transition matrix, where each element $P_{i,j} = P(S_t = j \mid S_{t-1} = i)$ represents the probability of moving from regime $i$ at time $t-1$ to regime $j$ at time $t$. This matrix must have dimensions $M \times M$, and **each row must sum to one**, ensuring valid probability transitions. If `None`, the model initializes a default transition structure, assigning probability 0.9 to diagonal elements and distributing the remaining probability evenly across the other $M-1$ regimes.

* `UnconditionalStateProbs`: The vector of unconditional regime probabilities, representing the prior distribution of states before observing any data. It must be a vector of length $1 \times M$. If `None`, the model assumes no prior information and initializes a uniform distribution, assigning probability $1/M$ to each regime.

* `param_names`: A dictionary containing metadata describing the dependent and independent variables in the model. When not provided, the model automatically generates generic parameter names; however, supplying `param_names` is strongly recommended for interpretability and reproducibility.

* `DatesLabel`: Used for plotting and reporting date range. Length must equal $T$ and must be list-like of `date`/`datetime`. If not provided, it uses numeric index `0..T-1`.

* `ModelName`: Only used for printing.

#### The `param_names` representation 

The `param_names` object stores **metadata describing the structure of the model**.
It does **not** affect estimation directly, but it is essential for:

* Clear and interpretable model summaries
* Correct labeling of coefficients and regimes
* Defining variable roles during prediction (e.g., intercept, trend, AR terms)

It is implemented as a **dictionary with a fixed and well-defined structure**, described below.

**Expected structure**

The `param_names` dictionary must contain **two top-level keys**:

* `"Y"`: metadata for the dependent variable(s)
* `"X"`: metadata for the independent variables (regressors)

A fully specified example is shown below:

```python
from MarkovSwitchingModel import TypeOfDependentVariable
from MarkovSwitchingModel import TypeOfVariable

{
    # information about the dependent variable
  "Y": {0:  { # first dependent variable
        "Name" : "MyFirstYVariable",
        "Type": TypeOfVariable.DEPENDENT,
        "ClassOfRegressor": None,
        "AR": None}
         # Example of how a second dependent variable would be added
        # (not currently supported, but indicative of a VAR-style extension):
        #
        # 1: {
        #     "Name": "MySecondYVariable",
        #     "Type": TypeOfVariable.DEPENDENT,
        #     "ClassOfRegressor": None,
        #     "AR": None
        # }
        
    },
  "X": {0:  { # Independent variable 1
        "Name" : "Intercempt",
        "Type": TypeOfVariable.INDEPENDENT,
        "ClassOfRegressor": TypeOfDependentVariable.INTERCEPT,
        "AR": None},
        1:  { # Independent variable 2
        "Name" : "Trend",
        "Type": TypeOfVariable.INDEPENDENT,
        "ClassOfRegressor": TypeOfDependentVariable.TREND,
        "AR": None},
        2:  { # Independent variable 3
        "Name" : "Lag_1",
        "Type": TypeOfVariable.INDEPENDENT,
        "ClassOfRegressor": TypeOfDependentVariable.AUTO_REGRESSIVE,
        "AR": 1},
        3:  { # first dependent variable
        "Name" : "Lag_2",
        "Type": TypeOfVariable.INDEPENDENT,
        "ClassOfRegressor": TypeOfDependentVariable.AUTO_REGRESSIVE,
        "AR": 2},
        4:  { # first dependent variable
        "Name" : "MyFirstYVariable",
        "Type": TypeOfVariable.INDEPENDENT,
        "ClassOfRegressor": TypeOfDependentVariable.EXOGENOUS,
        "AR": None}
    }
}
```

**Key requirements**

* The dictionary **must include both `"Y"` and `"X"` keys**
* `"X"` must be a dictionary with **exactly `K` entries**, where `K` is the number of columns in the regressor matrix `X`
* The integer keys (`0, 1, 2, ...`) must match the **column order** used to build the `X` matrix
* For autoregressive regressors:

  * `"ClassOfRegressor"` must be `AUTO_REGRESSIVE`
  * `"AR"` must specify the lag order (e.g., `1`, `2`, …)
* For non-autoregressive regressors, `"AR"` must be `None`

**Creating entries programmatically**

Each entry in `param_names` should be constructed using the helper function `GetDictRepresentation`, which enforces consistency and validation:

```python
GetDictRepresentation(
    name="Intercept",
    type=TypeOfVariable.INDEPENDENT,
    classX=TypeOfDependentVariable.INTERCEPT,
    transformation=TypeOfTransformation.NONE,
    ar=None
)
```

Using this helper is strongly recommended, as it ensures that:

* Required fields are present
* Regressor classes are correctly assigned
* Autoregressive lags are specified only when appropriate


## Basic Example

Two examples are provided in the file **`MarkovSwitchingExample.py`**. The first example is a "Two-regime Markov Switching model", the seccond example is a "Three-regime Markov Switching model". Both model specifications are provided below. This script demonstrates the full estimation workflow for the two examples.

### **Two-regime Markov Switching model**

The data-generating process (DGP) follows a Markov Switching regression with **two regimes**, each characterized by distinct parameters and governed by a first-order Markov chain.

**Data-Generating Process**

The data are generated according to a regime-dependent model of the form:
$$
y_{t} = \mu(S_t) + \beta_{x}(S_t) \cdot x + \sigma(S_t)\varepsilon
$$
where:

* $x_t$ is an exogenous explanatory variable,
* $\varepsilon_t \sim \mathcal{N}(0,1)$ is an i.i.d. standard normal innovation,
* $S_t$ denotes the latent regime at time $t$, evolving according to a Markov process,
* $\mu(S_t)$, $\beta_{x}(S_t)$, and $\sigma(S_t)$ are **regime-specific parameters**, allowing the conditional mean, slope coefficient, and volatility to vary across states.

This specification captures structural changes in both the level and the dynamics of the process by allowing the model parameters to switch according to the underlying regime $S_t$.


**Transition probabilities**

The regime dynamics are defined by the following transition probabilities:
$$\begin{aligned}
P(S_t = 0 \mid S_{t-1} = 0) &= 0.5, \\
P(S_t = 1 \mid S_{t-1} = 0) &= 0.5, \\
P(S_t = 0 \mid S_{t-1} = 1) &= 0.8, \\
P(S_t = 1 \mid S_{t-1} = 1) &= 0.2.
\end{aligned}$$
  
**Regime-specific parameters**

* **Regime 0**

    * Mean ($\mu$): 100
    * Slope coefficient ($\beta_x$): 2
    * Variance ($\sigma^2 = \Omega$): 4

* **Regime 1**

    * Mean ($\mu$): 0
    * Slope coefficient ($\beta_x$): 1
    * Variance ($\sigma^2 = \Omega$): 1

Together, these specifications generate a process with sharply contrasting regimes in both level and volatility, providing a clear illustration of how Markov Switching models capture structural changes in the data.


### **Three-regime Markov Switching model**

The data-generating process (DGP) follows a Markov Switching regression with **three regimes**, each characterized by distinct parameters and governed by a first-order Markov chain.

**Data-Generating Process**

The data are generated according to a regime-dependent AR(1) model of the form:
$$
Y_{t} = \mu(S_t) + \beta_{lag}(S_t) \cdot Y_{t-1} + \sigma(S_t)\varepsilon
$$
where:

* $\varepsilon_t \sim \mathcal{N}(0,1)$ is an i.i.d. standard normal innovation,
* $S_t$ denotes the latent regime at time $t$, evolving according to a Markov process,
* $\mu(S_t)$, $\beta_{lag}(S_t)$, and $\sigma(S_t)$ are **regime-specific parameters**, allowing the conditional mean, slope coefficient, and volatility to vary across states.

This specification captures structural changes in both the level and the dynamics of the process by allowing the model parameters to switch according to the underlying regime $S_t$.

**Transition probabilities**

The regime dynamics are defined by the following transition probabilities:
$$\begin{aligned}
P(S_t = 0 \mid S_{t-1} = 0) &= 0.40 \\
P(S_t = 1 \mid S_{t-1} = 0) &= 0.50 \\
P(S_t = 2 \mid S_{t-1} = 0) &= 0.10 \\

P(S_t = 0 \mid S_{t-1} = 1) &= 0.01 \\
P(S_t = 1 \mid S_{t-1} = 1) &= 0.18 \\
P(S_t = 2 \mid S_{t-1} = 1) &= 0.81 \\

P(S_t = 0 \mid S_{t-1} = 2) &= 0.09 \\
P(S_t = 1 \mid S_{t-1} = 2) &= 0.82 \\
P(S_t = 2 \mid S_{t-1} = 2) &= 0.09
\end{aligned}
$$

**Regime-specific parameters**

* **Regime 0**

    * Mean ($\mu$): -10
    * Slope coefficient ($\beta_{lag}$): 0.5
    * Variance ($\sigma^2 = \Omega$): 1

* **Regime 1**

    * Mean ($\mu$): 0
    * Slope coefficient ($\beta_{lag}$): 0.2
    * Variance ($\sigma^2 = \Omega$): 4

* **Regime 2**

    * Mean ($\mu$): 10
    * Slope coefficient ($\beta_{lag}$): -0.3
    * Variance ($\sigma^2 = \Omega$): 9

Together, these specifications generate a process with sharply contrasting regimes in both level and volatility, providing a clear illustration of how Markov Switching models capture structural changes in the data.
## License

Please refer to the **LICENSE** file for detailed licensing information.


## References

> Cavicchioli, M. (2021). *OLS Estimation of Markov switching VAR models: asymptotics and application to energy use*. AStA Advances in Statistical Analysis, 105(3), 431-449.

> Hamilton, J. D. (1996). *Specification testing in Markov-switching time-series models*. Journal of econometrics, 70(1), 127-157.

