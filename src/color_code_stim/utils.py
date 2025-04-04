import time
from functools import wraps
from pathlib import Path
from typing import (
    Callable,
    ParamSpec,
    Tuple,
    TypeVar,
)

import numpy as np
from statsmodels.stats.proportion import proportion_confint

# Type variables for preserving the signature
P = ParamSpec("P")  # For parameters
R = TypeVar("R")  # For return type


def timeit(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrap(*args: P.args, **kwargs: P.kwargs) -> R:
        if args[0].benchmarking:
            start = time.time()
            res = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"Elapsed time for function '{func.__name__}': {elapsed:.2e} s")
        else:
            res = func(*args, **kwargs)
        return res

    return wrap


def get_pfail(
    shots: int | np.ndarray,
    fails: int | np.ndarray,
    alpha: float = 0.01,
    confint_method: str = "wilson",
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """
    Calculate the failure probability and confidence interval.

    This function computes the estimated failure probability and the half-width
    of its confidence interval based on the number of shots and failures.

    Parameters
    ----------
    shots : int or array-like
        Total number of experimental shots.
    fails : int or array-like
        Number of failures observed.
    alpha : float, default 0.01
        Significance level for the confidence interval (e.g., 0.01 for 99%
        confidence).
    confint_method : str, default "wilson"
        Method to calculate confidence intervals. See
        statsmodels.stats.proportion.proportion_confint for available options.

    Returns
    -------
    pfail : float or array-like
        Estimated failure probability (midpoint of confidence interval).
    delta_pfail : float or array-like
        Half-width of the confidence interval.
    """
    pfail_low, pfail_high = proportion_confint(
        fails, shots, alpha=alpha, method=confint_method
    )
    pfail = (pfail_low + pfail_high) / 2
    delta_pfail = pfail_high - pfail

    return pfail, delta_pfail


def get_project_folder() -> Path:
    project_folder = Path(__file__).resolve().parents[2]
    return project_folder
