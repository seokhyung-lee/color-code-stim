"""
Sampling utilities for color code simulation.

This module provides utility functions for sampling, statistical analysis,
and error handling that support the Simulator class functionality.
"""

from typing import Optional, Tuple
import numpy as np
from statsmodels.stats.proportion import proportion_confint


def calculate_failure_statistics(
    shots: int,
    num_fails: np.ndarray,
    alpha: float = 0.01,
    confint_method: str = "wilson"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate failure rate statistics with confidence intervals.
    
    Parameters
    ----------
    shots : int
        Total number of simulation shots
    num_fails : np.ndarray
        Number of failures for each observable
    alpha : float, default 0.01
        Significance level for confidence interval
    confint_method : str, default 'wilson'
        Method for confidence interval calculation
        
    Returns
    -------
    pfail : np.ndarray
        Estimated failure rates
    delta_pfail : np.ndarray
        Half-width of confidence intervals
    """
    pfail = num_fails / shots
    
    if len(num_fails.shape) == 0:  # scalar case
        ci_low, ci_high = proportion_confint(
            num_fails, shots, alpha=alpha, method=confint_method
        )
        delta_pfail = (ci_high - ci_low) / 2
    else:  # array case
        delta_pfail = np.zeros_like(pfail)
        for i in range(len(num_fails)):
            ci_low, ci_high = proportion_confint(
                num_fails[i], shots, alpha=alpha, method=confint_method
            )
            delta_pfail[i] = (ci_high - ci_low) / 2
    
    return pfail, delta_pfail


def validate_sampling_parameters(
    shots: int,
    seed: Optional[int] = None
) -> None:
    """
    Validate parameters for sampling operations.
    
    Parameters
    ----------
    shots : int
        Number of shots to validate
    seed : int, optional
        Random seed to validate
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots}")
    
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ValueError(f"seed must be a non-negative integer, got {seed}")


def format_simulation_results(
    num_fails: np.ndarray,
    shots: int,
    extra_outputs: dict,
    verbose: bool = False
) -> str:
    """
    Format simulation results for display.
    
    Parameters
    ----------
    num_fails : np.ndarray
        Number of failures for each observable
    shots : int
        Total number of shots
    extra_outputs : dict
        Additional simulation outputs
    verbose : bool, default False
        Whether to include detailed information
        
    Returns
    -------
    str
        Formatted results string
    """
    lines = []
    
    # Basic failure statistics
    pfail = num_fails / shots
    lines.append(f"Simulation Results ({shots} shots):")
    lines.append(f"Failures: {num_fails}")
    lines.append(f"Failure rate: {pfail}")
    
    # Confidence intervals if available
    if "stats" in extra_outputs:
        pfail_ci, delta_pfail = extra_outputs["stats"]
        lines.append(f"95% CI: [{pfail_ci - delta_pfail:.6f}, {pfail_ci + delta_pfail:.6f}]")
    
    # Additional details if verbose
    if verbose and "fails" in extra_outputs:
        fails = extra_outputs["fails"]
        lines.append(f"Failure pattern: {np.sum(fails, axis=1)[:10]}...")  # First 10 samples
    
    return "\n".join(lines)


def analyze_error_patterns(
    errors: np.ndarray,
    det_outcomes: np.ndarray,
    max_analysis_samples: int = 1000
) -> dict:
    """
    Analyze patterns in error data for diagnostic purposes.
    
    Parameters
    ----------
    errors : np.ndarray
        Error patterns from simulation
    det_outcomes : np.ndarray
        Detector outcomes corresponding to errors
    max_analysis_samples : int, default 1000
        Maximum number of samples to analyze (for performance)
        
    Returns
    -------
    dict
        Analysis results including error statistics
    """
    # Limit analysis for performance
    n_samples = min(len(errors), max_analysis_samples)
    sample_errors = errors[:n_samples]
    sample_det = det_outcomes[:n_samples]
    
    analysis = {
        "error_density": np.mean(sample_errors),
        "detector_density": np.mean(sample_det),
        "max_errors_per_sample": np.max(np.sum(sample_errors, axis=1)),
        "avg_errors_per_sample": np.mean(np.sum(sample_errors, axis=1)),
        "error_correlation": np.corrcoef(
            np.sum(sample_errors, axis=1),
            np.sum(sample_det, axis=1)
        )[0, 1] if n_samples > 1 else 0.0
    }
    
    return analysis


def seed_generator(base_seed: Optional[int] = None) -> int:
    """
    Generate reproducible seeds for multi-stage simulations.
    
    Parameters
    ----------
    base_seed : int, optional
        Base seed for reproducible generation
        
    Returns
    -------
    int
        Generated seed value
    """
    if base_seed is None:
        return np.random.randint(0, 2**31 - 1)
    
    # Use a simple hash-based approach for reproducible seed generation
    rng = np.random.RandomState(base_seed)
    return rng.randint(0, 2**31 - 1)