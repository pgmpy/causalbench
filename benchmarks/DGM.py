import numpy as np
import pandas as pd
from scipy.stats import bernoulli, expon, multinomial, norm, uniform


def linear_gaussian(
    n_samples=1000,
    effect_size=1.0,
    n_cond_vars=1,
    seed=None,
):
    """
    Uses Linear model to generate data. The model is defined as:

        ..math:: Z \sim \mathcal{N}(\bm{0}, \bm{1})
        ..math:: \alpha, \beta \sim \textrm{Uniform(0, 1)}
        ..math:: X \sim \mathcal{N}(\alpha * Z) + \mathcal{N}(0, 1)
        ..math:: Y \sim \mathcal{N}(\beta * Z + effect * X) + \mathcal{N}(0, 1)

    When effect_size = 0, the generated data satisfies $ X \ci Y \rvert Z $.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate.

    effect_size : float, optional
        Defines how strongly X and Y are dependent. If 0, X and Y are conditionally
        independent.

    n_cond_vars : int, optional
        Number of conditional variables to generate.

    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['X', 'Y', 'Z1', ...].
    """
    # Step 0: Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Step 1: Sample the Zs.
    Zs = rng.normal(size=(n_samples, n_cond_vars))

    # Step 2: Sample X.
    coef_ZX = rng.uniform(0, 1, size=n_cond_vars)
    X = Zs @ coef_ZX + rng.normal(size=n_samples)

    # Step 3: Sample Y.
    coef_ZY = rng.uniform(0, 1, size=n_cond_vars)
    Y = Zs @ coef_ZY + effect_size * X + rng.normal(size=n_samples)

    # Step 4: Create a dataframe and return.
    data = pd.DataFrame({"X": X, "Y": Y})
    for j in range(n_cond_vars):
        data[f"Z{j+1}"] = Zs[:, j]

    return df


def nonlinear_gaussian(
    n_samples=1000,
    effect_size=1.0,
    noise_std=1.0,
    n_cond_vars=1,
    seed=None,
    dependent=True,
):
    """
    Nonlinear Gaussian DGP:

    Generates data where X and Y are nonlinear functions of Z, optionally (conditionally) dependent or independent.

    Parameters
    ----------
    n_samples : int, optional
    effect_size : float, optional
    noise_std : float, optional
    n_cond_vars : int, optional
    seed : int, optional
    dependent : bool, optional

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['X', 'Y', 'Z1', ...].
        Variable types are in df.attrs['variable_types'].

    Reference: Peters et al (2011), "Causal inference by using invariant prediction"
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_samples, n_cond_vars))
    e1 = rng.normal(scale=noise_std, size=n_samples)
    e2 = rng.normal(scale=noise_std, size=n_samples)
    Z_sum = Z.sum(axis=1)
    if dependent:
        X = np.sin(effect_size * Z_sum) + e1
        Y = np.exp(effect_size * Z_sum * 0.2) + e2
    else:
        X = np.sin(effect_size * Z_sum) + e1
        Y = (
            np.exp(effect_size * rng.normal(size=n_samples) * 0.2) + e2
        )  # Break dependence
    data = {"X": X, "Y": Y}
    for j in range(n_cond_vars):
        data[f"Z{j+1}"] = Z[:, j]
    df = pd.DataFrame(data)
    df.attrs["variable_types"] = {
        "X": "continuous",
        "Y": "continuous",
        **{f"Z{j+1}": "continuous" for j in range(n_cond_vars)},
    }
    return df


# --- NEW DGMs ADDED LATER ---


def discrete_categorical(
    n_samples=1000,
    effect_size=1.0,
    noise_std=1.0,
    n_cond_vars=1,
    n_categories=3,
    noise_prob=0.05,
    seed=None,
    dependent=True,
):
    """
    Discrete (categorical) DGP:

    Parameters
    ----------
    n_samples : int
    effect_size : float
    noise_std : float
        Not used, kept for API consistency.
    n_cond_vars : int
        Number of conditional variables (vector-valued Z).
    n_categories : int
    noise_prob : float
    seed : int
    dependent : bool

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['X', 'Y', 'Z1', ...].
        Variable types are in df.attrs['variable_types'].
    Reference: Scutari, Denis (2021), "Bayesian Networks: With Examples in R"
    """
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, n_categories, size=(n_samples, n_cond_vars))
    if dependent:
        X = Z.sum(axis=1)
        Y = Z.sum(axis=1)
    else:
        X = Z.sum(axis=1)
        Y = rng.integers(0, n_categories * n_cond_vars, size=n_samples)
    for arr in (X, Y):
        flips = rng.random(n_samples) < noise_prob
        arr[flips] = rng.integers(0, n_categories * n_cond_vars, size=flips.sum())
    data = {"X": X, "Y": Y}
    for j in range(n_cond_vars):
        data[f"Z{j+1}"] = Z[:, j]
    df = pd.DataFrame(data)
    df.attrs["variable_types"] = {
        "X": "categorical",
        "Y": "categorical",
        **{f"Z{j+1}": "categorical" for j in range(n_cond_vars)},
    }
    return df


def mixed_data(
    n_samples=1000,
    effect_size=1.0,
    noise_std=1.0,
    n_cond_vars=1,
    n_cat=2,
    seed=None,
    dependent=True,
):
    """
    Mixed continuous and categorical DGP:

    Parameters
    ----------
    n_samples : int
    effect_size : float
    noise_std : float
    n_cond_vars : int
    n_cat : int
    seed : int
    dependent : bool

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['X', 'Y', 'Z1', ...].
        Variable types are in df.attrs['variable_types'].
    Reference: Ghassami et al (2017), "Learning Mixed Graphical Models"
    """
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, n_cat, size=(n_samples, n_cond_vars))
    e1 = rng.normal(scale=noise_std, size=n_samples)
    e2 = rng.normal(scale=noise_std, size=n_samples)
    Z_sum = Z.sum(axis=1)
    if dependent:
        X = effect_size * Z_sum + e1
        Y = 0.5 * effect_size * Z_sum + e2
    else:
        X = effect_size * Z_sum + e1
        Y = (
            0.5 * effect_size * rng.integers(0, n_cat * n_cond_vars, size=n_samples)
            + e2
        )
    data = {"X": X, "Y": Y}
    for j in range(n_cond_vars):
        data[f"Z{j+1}"] = Z[:, j]
    variable_types = {"X": "continuous", "Y": "continuous"}
    variable_types.update({f"Z{j+1}": "categorical" for j in range(n_cond_vars)})
    df = pd.DataFrame(data)
    df.attrs["variable_types"] = variable_types
    return df


def non_gaussian_continuous(
    n_samples=1000,
    effect_size=1.0,
    n_cond_vars=1,
    noise_std=1.0,
    seed=None,
    dependent=True,
):
    """
    Non-Gaussian continuous DGP:

    Parameters
    ----------
    n_samples : int
    effect_size : float
    noise_std : float
        Not used, kept for API consistency.
    n_cond_vars : int
    seed : int
    dependent : bool

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['X', 'Y', 'Z1', ...].
        Variable types are in df.attrs['variable_types'].
    Reference: Shimizu et al (2006), "A Linear Non-Gaussian Acyclic Model for Causal Discovery"
    """
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-2, 2, size=(n_samples, n_cond_vars))
    e1 = rng.exponential(scale=1.0, size=n_samples)
    e2 = rng.exponential(scale=1.0, size=n_samples)
    Z_sum = Z.sum(axis=1)
    if dependent:
        X = effect_size * np.abs(Z_sum) + e1
        Y = effect_size * (Z_sum) ** 2 + e2
    else:
        X = effect_size * np.abs(Z_sum) + e1
        Y = effect_size * rng.uniform(-2, 2, size=n_samples) ** 2 + e2
    data = {"X": X, "Y": Y}
    for j in range(n_cond_vars):
        data[f"Z{j+1}"] = Z[:, j]
    df = pd.DataFrame(data)
    df.attrs["variable_types"] = {
        "X": "continuous",
        "Y": "continuous",
        **{f"Z{j+1}": "continuous" for j in range(n_cond_vars)},
    }
    return df


#  NEW DGMS WILL BE ADDED HERE


# Optionally, a registry for easy reference
DGP_REGISTRY = {
    "linear_gaussian": linear_gaussian,
    "nonlinear_gaussian": nonlinear_gaussian,
    "discrete_categorical": discrete_categorical,
    "mixed_data": mixed_data,
    "non_gaussian_continuous": non_gaussian_continuous,
}

# Example usage for testing:
if __name__ == "__main__":
    for name, func in DGP_REGISTRY.items():
        if name == "user_defined":
            continue
        print(f"Generating data for {name}...")
        df = func(n_samples=100, seed=42)
        print(df.head())
