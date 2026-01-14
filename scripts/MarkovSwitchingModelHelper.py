import MarkovSwitchingModel as MKM
import matplotlib.pyplot as plt
import pandas as pd
import pathlib as plib

def GenerateSmoothProbabilitiesPlot(model: MKM.MarkovSwitchingModel, filePath: plib.Path | None = None) -> None:
    """
    Generate and save a multi-panel plot of smoothed regime probabilities.

    Creates one subplot per regime showing the smoothed probability
    (``model.Xi_smoothed[:, regime]``) over time (``model.DatesLabel``), and saves the
    figure as a PNG file named ``"{fileStem} markov_switching_plot.png"``.

    Parameters
    ----------
    model : MKM.MarkovSwitchingModel
    fileStem : str
        Output file prefix/path used when saving the PNG.

    Returns
    -------
    None
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
        ax.plot(model.DatesLabel, model.Xi_smoothed[:, cur_mod], lw=0.5)
        ax.fill_between(x = model.DatesLabel,  y1 = 1, where=(model.Xi_smoothed[:, cur_mod] >= 0.5), color="k", alpha=0.1)
           
        # Set the y-axis label to indicate which regime this subplot represents
        ax.set_ylabel(f"Regime {cur_mod}")
    
    # Set the title for the entire figure
    fig.suptitle(f'{model.ModelName} Smoothed Probabilities', fontsize=9)

    # Adjust the layout to prevent overlapping labels and titles
    fig.tight_layout()

    # Save the figure to a PNG file with high resolution (300 DPI)
    # bbox_inches='tight' ensures no content is cut off at the figure edges
    if filePath is None:
        filePath =plib.Path(f"{model.ModelName} - markov_switching_plot.png")
    plt.savefig(filePath, dpi=300, bbox_inches='tight')

def GetRegimeClassification (model: MKM.MarkovSwitchingModel) -> pd.DataFrame:
    """
    Extract and classify regime periods from a Markov Switching Model.
    This function analyzes the smoothed state probabilities from a Markov Switching Model
    to identify distinct regime periods, calculate their probabilities, and summarize
    the regime switches over time.
    Parameters
    ----------
    model : MarkovSwitchingModel
        A fitted Markov Switching Model containing:
        - Xi_smoothed: Smoothed state probabilities for each regime
        - NumRegimes: Number of regimes in the model
        - DatesLabel: Time series labels/dates for the observations
    Returns
    -------
    pd.DataFrame
        A DataFrame with MultiIndex (Regime_count, Regime) containing:
        - start_date: Beginning date of the regime period
        - end_date: Ending date of the regime period
        - qtd: Number of observations in the regime period
        - prob: Average probability of the regime during the period
    Examples
    --------
    >>> regime_summary = GetRegimeClassification(fitted_model)
    >>> print(regime_summary)
                        start_date    end_date  qtd      prob
    Regime_count Regime                                      
    1            S_0    2020-01-01  2020-03-15   75  0.923456
    2            S_1    2020-03-16  2020-06-30  107  0.856721
    """

    # Create DataFrame from smoothed state probabilities    
    cols = [f"S_{i}" for i in range(model.NumRegimes)]
    df_States = pd.DataFrame(model.Xi_smoothed, columns=cols)

    # create index column (example: dates)
    df_States.insert(
        loc=0,
        column="date",
        value=model.DatesLabel)

    # set as index
    df_States = df_States.set_index("date")

    # Determine the regime with the highest probability for each observation
    df_States["Regime"] = df_States[cols].idxmax(axis=1)

    # Create a regime change counter
    df_States["Regime_count"] = (df_States["Regime"] != df_States["Regime"].shift()).cumsum()

    # Generate summary DataFrame
    df_States_Resume = df_States.groupby(["Regime_count", "Regime"]).apply(lambda x: pd.Series({
    "start_date": x.index.min(),
    "end_date": x.index.max(),
    "qtd": len(x),
    "prob" : x[x.name[1]].mean()
    }))

    return df_States_Resume

def GetRegimeNames(model: MKM.MarkovSwitchingModel) -> list:
    """
    Generate descriptive regime names based on statistical significance and variance characteristics.
    This function creates a list of regime names that combine variance levels (Low, Mid, High)
    with the statistical significance of the intercept term. The naming convention produces
    labels such as 'S_Low_Variance', 'S_High_Variance', etc.
    The function performs the following steps:
    1. Validates that the model contains no more than 3 regimes
    2. Identifies the index of the intercept regressor
    3. Retrieves beta inference statistics from the model
    4. Classifies each regime by intercept significance (p-value < 0.05) and direction
    5. Ranks regimes by variance (lowest, highest, and mid-range)
    6. Returns regime names combining variance ranking with intercept characteristics
        model (MKM.MarkovSwitchingModel): The Markov Switching Model containing regime information,
                                         inference statistics, and variance estimates (Omega).
    Returns:
        list: A list of strings containing regime names indexed by regime number,
              formatted as 'S_{variance_level}' where variance_level is one of:
              'Low_Variance', 'Mid_Variance', or 'High_Variance'.
    Raises:
        ValueError: If the model contains more than 3 regimes, as these are not supported
                   for automatic naming.
    Note:
        The function assumes the model has been estimated and contains valid inference
        matrices and variance estimates (Omega). Regimes are classified by intercept
        significance and direction when p-value < 0.05.
    Generate regime names based on the number of regimes in the model.
    
    This function creates a list of regime names formatted as 'S_0', 'S_1', ..., 'S_{n-1}',
    where 'n' is the number of regimes specified in the Markov Switching Model.
    
    Parameters:
        model (MarkovSwitchingModel): The Markov Switching Model containing the number of regimes.
    """

    if model.NumRegimes > 3:
        raise ValueError("Models with more than 3 regimes are not supported for naming.")

    # determine the index of the intercept
    indexOfRegressor = 0
    for key, value in model.ParamNames['X'].items():
        if value["ClassOfRegressor"] == MKM.TypeOfDependentVariable.INTERCEPT:
            indexOfRegressor = key
            break

    # get the inference info
    InferenceInfo = model.GetBetaInferenceMatrix()

    regime_names = list()
    for regime in range(model.NumRegimes):
        if InferenceInfo[indexOfRegressor, 3, regime] < 0.1:
            if InferenceInfo[indexOfRegressor, 0, regime] <0:
                regime_names.append(f"Regime de Baixa")
            elif InferenceInfo[indexOfRegressor, 0, regime] >0:
                regime_names.append(f"Regime de Alta")
        else:
            regime_names.append(f"Regime Neutro")


    regime_variances = list()
    for regime in range(model.NumRegimes):
        if regime == model.Omega.argmin():
            # Regime with the lowest variance
            regime_variances.append("baixa variancia")
        elif regime == model.Omega.argmax():
            regime_variances.append("alta variancia")
        else:
            regime_variances.append("variancia media")

    regime_names = [f"{regime_names[r]} com {regime_variances[r]}" for r in range(model.NumRegimes)]
    
    return regime_names