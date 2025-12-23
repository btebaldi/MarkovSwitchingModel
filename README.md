# MarkovSwitchingModel

## Overview

This implementation provides tools for estimating and analyzing Markov Switching Vector Autoregressive (VAR) models using Ordinary Least Squares (OLS) methods. The model is based on the theoretical framework presented in "OLS Estimation of Markov switching VAR models: asymptotics and application to energy use" by Maddalena Cavicchioli.

## Features

- Markov regime-switching VAR estimation
- OLS-based parameter estimation
- Asymptotic inference and hypothesis testing

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