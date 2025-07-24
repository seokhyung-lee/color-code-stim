"""
Belief Propagation Decoder for Color Code

This module implements belief propagation decoding for color codes using the
optional ldpc library. It provides an alternative decoding strategy that can
be used independently or as pre-decoding for the concatenated matching decoder.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

from .base import BaseDecoder
from ..dem_utils.dem_manager import DemManager
from ..stim_utils import dem_to_parity_check, remove_obs_from_dem


class BPDecoder(BaseDecoder):
    """
    Belief propagation decoder for color codes.

    This decoder uses the LDPC belief propagation algorithm to decode detector
    outcomes. It requires the optional 'ldpc' package and provides an alternative
    to the concatenated matching decoder, particularly useful for pre-decoding
    in hybrid strategies.

    Key Features:
    - LDPC belief propagation with configurable iterations
    - Support for both 1D and 2D detector outcome arrays
    - Optional DEM observable removal for comparative decoding
    - Returns predictions, log-likelihood ratios, and convergence information
    - Graceful handling of missing ldpc dependency

    Attributes
    ----------
    dem_manager : DEMManager
        Manager for detector error models and matrices
    comparative_decoding : bool
        Whether to remove observables from DEM before decoding
    _cached_inputs : dict or None
        Cached parity check matrix and probabilities for efficiency
    """

    def __init__(
        self,
        dem_manager: DemManager,
        comparative_decoding: bool = False,
        cache_inputs: bool = True,
    ):
        """
        Initialize the belief propagation decoder.

        Parameters
        ----------
        dem_manager : DEMManager
            Manager providing access to detector error models
        comparative_decoding : bool, default False
            Whether to remove observables from DEM before decoding
        cache_inputs : bool, default True
            Whether to cache the parity check matrix and probabilities
        """
        self.dem_manager = dem_manager
        self.comparative_decoding = comparative_decoding
        self.cache_inputs = cache_inputs
        self._cached_inputs: Optional[Dict[str, np.ndarray]] = None

    def supports_comparative_decoding(self) -> bool:
        """Return True - this decoder can work with comparative decoding."""
        return True

    def supports_predecoding(self) -> bool:
        """Return False - this decoder is typically used as pre-decoding itself."""
        return False

    def decode(
        self, detector_outcomes: np.ndarray, max_iter: int = 10, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Union[bool, np.ndarray]]:
        """
        Decode detector outcomes using belief propagation.

        This method uses the LDPC belief propagation decoder to decode detector
        outcomes. It handles both single samples (1D) and multiple samples (2D).

        Parameters
        ----------
        detector_outcomes : np.ndarray
            1D or 2D array of detector measurement outcomes to decode.
            For 1D: single sample with detector outcomes.
            For 2D: multiple samples, each row is a sample.
        max_iter : int, default 10
            Maximum number of belief propagation iterations to perform.
        **kwargs
            Additional keyword arguments to pass to the BpDecoder constructor.

        Returns
        -------
        tuple
            (pred, llrs, converge) where:
            - pred: Predicted error pattern (same dimensionality as input)
            - llrs: Log probability ratios for each bit
            - converge: Convergence status (bool for 1D, array for 2D)

        Raises
        ------
        ImportError
            If the 'ldpc' package is not installed.
        ValueError
            If detector_outcomes has invalid dimensions (not 1D or 2D).
        """
        try:
            from ldpc import BpDecoder
        except ImportError:
            raise ImportError(
                "The 'ldpc' package is required for belief propagation decoding. "
                "Please install it using: pip install ldpc"
            )

        # Get or compute parity check matrix and probabilities
        if self.cache_inputs and self._cached_inputs is not None:
            H = self._cached_inputs["H"]
            p = self._cached_inputs["p"]
        else:
            H, p = self._prepare_bp_inputs()
            if self.cache_inputs:
                self._cached_inputs = {"H": H, "p": p}

        detector_outcomes = np.asarray(detector_outcomes)
        
        # Filter detector outcomes to match the DEM dimensions when observables are removed
        if self.comparative_decoding:
            expected_detectors = H.shape[0]
            if detector_outcomes.shape[-1] != expected_detectors:
                # Truncate to match the number of detectors in the filtered DEM
                detector_outcomes = detector_outcomes[..., :expected_detectors]

        # Filter kwargs to only include valid BpDecoder parameters
        bp_kwargs = {k: v for k, v in kwargs.items() if k in ['bp_method', 'schedule', 'ms_scaling_factor', 'bp_method_type']}
        
        if detector_outcomes.ndim == 1:
            # Single sample decoding
            bpd = BpDecoder(H, error_channel=p, max_iter=max_iter, **bp_kwargs)
            pred = bpd.decode(detector_outcomes)
            llrs = bpd.log_prob_ratios
            converge = bpd.converge

        elif detector_outcomes.ndim == 2:
            # Multi-sample decoding
            pred = []
            llrs = []
            converge = []

            for det_sng in detector_outcomes:
                bpd = BpDecoder(H, error_channel=p, max_iter=max_iter, **bp_kwargs)
                pred.append(bpd.decode(det_sng))
                llrs.append(bpd.log_prob_ratios)
                converge.append(bpd.converge)

            pred = np.stack(pred, axis=0)
            llrs = np.stack(llrs, axis=0)
            converge = np.stack(converge, axis=0)

        else:
            raise ValueError(
                f"detector_outcomes must be 1D or 2D, got shape {detector_outcomes.shape}"
            )

        return pred, llrs, converge

    def _prepare_bp_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare parity check matrix and probabilities for BP decoding.

        Returns
        -------
        tuple
            (H, p) where H is the parity check matrix and p is the error probabilities
        """
        dem = self.dem_manager.dem_xz

        if self.comparative_decoding:
            # Remove observables from DEM for comparative decoding
            dem = remove_obs_from_dem(dem)

        # Convert DEM to parity check matrix and probabilities
        H, _, p = dem_to_parity_check(dem)
        
        # Convert H to uint8 as required by ldpc.BpDecoder
        H = H.astype('uint8')

        return H, p

    def clear_cache(self):
        """Clear cached inputs to force recomputation on next decode."""
        self._cached_inputs = None

    def get_parity_check_info(self) -> Dict[str, Union[int, float]]:
        """
        Get information about the parity check matrix.

        Returns
        -------
        dict
            Dictionary containing matrix dimensions, density, and statistics
        """
        if self._cached_inputs is None:
            H, p = self._prepare_bp_inputs()
        else:
            H = self._cached_inputs["H"]
            p = self._cached_inputs["p"]

        return {
            "matrix_shape": H.shape,
            "num_checks": H.shape[0],
            "num_variables": H.shape[1],
            "density": H.nnz / (H.shape[0] * H.shape[1]),
            "avg_check_degree": H.sum(axis=1).mean(),
            "avg_variable_degree": H.sum(axis=0).mean(),
            "min_probability": p.min(),
            "max_probability": p.max(),
            "avg_probability": p.mean(),
        }
