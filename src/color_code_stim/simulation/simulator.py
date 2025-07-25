"""
Simulator module for color code quantum error correction.

This module provides the Simulator class that handles Monte Carlo simulation,
detector/observable sampling, and related utility functions extracted from
the ColorCode class to support the modular architecture.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import stim

from ..utils import get_pfail


class Simulator:
    """
    Simulation and sampling manager for color code circuits.

    This class handles Monte Carlo simulation, detector/observable sampling,
    and utility functions for error analysis. It operates on pre-built circuits
    and integrates with the decoder architecture through dependency injection.
    """

    def __init__(
        self,
        *,
        circuit: stim.Circuit,
        circuit_type: str,
    ):
        """
        Initialize the Simulator with required dependencies.

        Parameters
        ----------
        circuit : stim.Circuit
            The quantum circuit to simulate
        circuit_type : str
            Type of circuit ('tri', 'rec', 'rec_stability', 'growing', 'cult+growing')
        """
        self.circuit = circuit
        self.circuit_type = circuit_type

    def sample(
        self, shots: int, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample detector outcomes and observables from the quantum circuit.

        Parameters
        ----------
        shots : int
            Number of samples to generate
        seed : int, optional
            Seed value to initialize the random number generator

        Returns
        -------
        det : 2D numpy array of bool
            Detector outcomes. det[i,j] is True if and only if the detector
            with id j in the i-th sample has an outcome of −1.
        obs : 1D or 2D numpy array of bool
            Observable outcomes. If there is only one observable, returns a 1D array;
            otherwise returns a 2D array. obs[i] or obs[i,j] is True if and only if
            the j-th observable (j=0 when 1D) of the i-th sample has an outcome of -1.
        """
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        det, obs = sampler.sample(shots, separate_observables=True)
        if obs.shape[1] == 1:
            obs = obs.ravel()
        return det, obs

    def sample_with_errors(
        self,
        shots: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample detector outcomes, observables, and errors from the quantum circuit.

        Parameters
        ----------
        shots : int
            Number of samples to generate
        seed : int, optional
            Seed value to initialize the random number generator

        Returns
        -------
        det : 2D numpy array of bool
            Detector outcomes. det[i,j] is True if and only if the detector
            with id j in the i-th sample has an outcome of −1.
        obs : 1D or 2D numpy array of bool
            Observable outcomes. If there is only one observable, returns a 1D array;
            otherwise returns a 2D array. obs[i] or obs[i,j] is True if and only if
            the j-th observable (j=0 when 1D) of the i-th sample has an outcome of -1.
        errors : 2D numpy array of bool
            Errors sampled from the quantum circuit. errors[i,j] is True if and only if
            the j-th error (in the DEM) of the i-th sample has an outcome of -1.
        """
        dem = self.circuit.detector_error_model()
        sampler = dem.compile_sampler(seed=seed)
        det, obs, err = sampler.sample(shots, return_errors=True)
        if obs.shape[1] == 1:
            obs = obs.ravel()

        return det, obs, err

    def simulate(
        self,
        shots: int,
        *,
        decoder_func,
        colors: Union[List[str], str] = "all",
        alpha: float = 0.01,
        confint_method: str = "wilson",
        full_output: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        """
        Monte-Carlo simulation of quantum error correction decoding.

        Parameters
        ----------
        shots : int
            Number of shots to simulate.
        decoder_func : callable
            Decoder function that takes detector outcomes and returns predictions.
            Should have signature: decoder_func(detector_outcomes, **kwargs) -> predictions
        colors : Union[List[str], str], default 'all'
            Colors of the sub-decoding procedures to consider. Can be 'all', one of {'r', 'g', 'b'},
            or a list containing any combination of {'r', 'g', 'b'}.
        alpha : float, default 0.01
            Significance level for the confidence interval calculation.
        confint_method : str, default 'wilson'
            Method to calculate the confidence interval.
            See statsmodels.stats.proportion.proportion_confint for available options.
        full_output: bool = False,
            If True, return additional information.
        seed : Optional[int], default None
            Seed to initialize the random number generator.
        verbose : bool, default False
            If True, print progress information during simulation.
        **kwargs :
            Additional keyword arguments for the decoder function.

        Returns
        -------
        num_fails : numpy.ndarray
            Number of failures for each observable.
        extra_outputs : dict
            Dictionary containing additional information:
            - 'stats': Tuple of (pfail, delta_pfail) where pfail is the estimated failure rate
              and delta_pfail is the half-width of the confidence interval
            - 'fails': Boolean array indicating which samples failed
            - Additional outputs from the decoder function if full_output=True
        """
        if self.circuit_type == "cult+growing":
            raise NotImplementedError(
                "Cult+growing circuit type is not supported for this method."
            )

        if colors == "all":
            colors = ["r", "g", "b"]

        if verbose:
            print("Sampling...")

        shots = round(shots)
        det, obs = self.sample(shots, seed=seed)

        if verbose:
            print("Decoding...")

        preds = decoder_func(
            det,
            colors=colors,
            verbose=verbose,
            full_output=full_output,
            **kwargs,
        )

        if full_output:
            preds, extra_outputs = preds

        if verbose:
            print("Postprocessing...")

        fails = np.logical_xor(obs, preds)
        num_fails = np.sum(fails, axis=0)

        if full_output:
            pfail, delta_pfail = get_pfail(
                shots, num_fails, alpha=alpha, confint_method=confint_method
            )
            extra_outputs["stats"] = (pfail, delta_pfail)
            extra_outputs["fails"] = fails

            return num_fails, extra_outputs

        else:
            return num_fails
