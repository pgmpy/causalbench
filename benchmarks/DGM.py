import numpy as np
import pandas as pd
from scipy.stats import bernoulli, expon, multinomial, norm, uniform

DGP_REGISTRY = {
    "linear_gaussian": linear_gaussian,
    "nonlinear_gaussian": nonlinear_gaussian,
    "discrete_categorical": discrete_categorical,
    "mixed_data": mixed_data,
    "non_gaussian_continuous": non_gaussian_continuous,
}

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
    df = pd.DataFrame({"X": X, "Y": Y})
    for j in range(n_cond_vars):
        df[f"Z{j+1}"] = Zs[:, j]

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
    Uses a Non-Linear Gaussian model to generate data. The model is defined as:

        ..math:: Z \sim \mathcal{N}(\mathbf{0}, \mathbf{1})
        ..math:: X = \sin(effect\_size \cdot \sum_j Z_j) + \epsilon_1
        ..math:: Y = \exp(effect\_size \cdot \sum_j Z_j \cdot 0.2) + \epsilon_2 \quad \text{if dependent}
        ..math:: Y = \exp(effect\_size \cdot N(0, 1) \cdot 0.2) + \epsilon_2 \quad \text{if independent}

    When dependent=False, the generated data satisfies $ X \ci Y \mid Z $.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate.

    effect_size : float, optional
        Controls the strength of the nonlinear transformation.

    n_cond_vars : int, optional
        Number of conditional variables to generate.

    seed : int, optional
        Seed for the random number generator.

    dependent : bool, optional
        Whether X and Y are dependent (True) or independent (False) given Z.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['X', 'Y', 'Z1', ...].
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
    Uses a discrete categorical model to generate data. The model is defined as:

        ..math:: Z_j \sim \mathrm{DiscreteUniform}(0, n\_categories-1)
        ..math:: X = \sum_j Z_j + \text{noise}
        ..math:: Y = \sum_j Z_j + \text{noise} \quad \text{if dependent}
        ..math:: Y \sim \mathrm{DiscreteUniform}(0, n\_categories \cdot n\_cond\_vars - 1) + \text{noise} \quad \text{if independent}

    Randomly introduces noise to X and Y with probability `noise_prob`.

    When dependent=False, the generated data satisfies $ X \ci Y \mid Z $.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate.

    effect_size : float, optional
        Not used (for API compatibility).

    n_cond_vars : int, optional
        Number of conditional variables to generate.

    n_categories : int, optional
        Number of categories for each Z.

    noise_prob : float, optional
        Probability of flipping X or Y to a random value.

    seed : int, optional
        Seed for the random number generator.

    dependent : bool, optional
        Whether X and Y are dependent (True) or independent (False) given Z.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['X', 'Y', 'Z1', ...].
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
    n_cond_vars=1,
    n_cat=2,
    seed=None,
):
    """
    Uses a Mixed model to generate continuous X/Y and categorical Z. The model is defined as:

        ..math:: Z_j \sim \mathrm{DiscreteUniform}(0, n\_cat-1)
        ..math:: \alpha, \beta \sim \mathrm{Uniform}(0, 1)
        ..math:: X = Z \cdot \alpha + \mathcal{N}(0, 1)
        ..math:: Y = Z \cdot \beta + effect\_size \cdot X + \mathcal{N}(0, 1)

    When effect_size = 0, the generated data satisfies $ X \ci Y \mid Z $.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate.

    effect_size : float, optional
        Strength of dependence from X to Y.

    n_cond_vars : int, optional
        Number of conditional variables to generate.

    n_cat : int, optional
        Number of categories for each Z.

    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['X', 'Y', 'Z1', ...].
    Reference: Ghassami et al (2017), "Learning Mixed Graphical Models"
    """
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, n_cat, size=(n_samples, n_cond_vars))
    coef_ZX = rng.uniform(0, 1, size=n_cond_vars)
    coef_ZY = rng.uniform(0, 1, size=n_cond_vars)
    e1 = rng.normal(size=n_samples)
    e2 = rng.normal(size=n_samples)
    X = Z @ coef_ZX + e1
    Y = (Z @ coef_ZY) + effect_size * X + e2
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
    seed=None,
    dependent=True,
):
    """
    Uses a linear non-Gaussian acyclic model to generate continuous, non-Gaussian data. The model is defined as:

        ..math:: Z_j \sim \mathrm{Uniform}(-2, 2)
        ..math:: \alpha, \beta \sim \mathrm{Uniform}(0, 1)
        ..math:: X = |\ Z \cdot \alpha\ | + e_1,\quad e_1 \sim \mathrm{Exponential}(1.0)
        ..math:: Y = (Z \cdot \beta)^2 + effect\_size \cdot X + e_2, \quad e_2 \sim \mathrm{Exponential}(1.0), \text{if dependent}
        ..math:: Y = (Z \cdot \beta)^2 + e_2, \quad e_2 \sim \mathrm{Exponential}(1.0), \text{if independent}

    When dependent=False or effect_size=0, the generated data satisfies $ X \ci Y \mid Z $.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate.

    effect_size : float, optional
        Strength of dependence from X to Y.

    n_cond_vars : int, optional
        Number of conditional variables to generate.

    seed : int, optional
        Seed for the random number generator.

    dependent : bool, optional
        Whether X and Y are dependent (True) or independent (False) given Z.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['X', 'Y', 'Z1', ...].
    Reference: Shimizu et al (2006), "A Linear Non-Gaussian Acyclic Model for Causal Discovery"
    """
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-2, 2, size=(n_samples, n_cond_vars))
    coef_ZX = rng.uniform(0, 1, size=n_cond_vars)
    coef_ZY = rng.uniform(0, 1, size=n_cond_vars)
    e1 = rng.exponential(scale=1.0, size=n_samples)
    e2 = rng.exponential(scale=1.0, size=n_samples)
    X = np.abs(Z @ coef_ZX) + e1
    Y = (Z @ coef_ZY) ** 2 + effect_size * X + e2
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
