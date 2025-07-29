"""
Belief Propagation + Concatenated Matching Decoder for Color Code

This module implements a composite decoder that combines belief propagation
pre-decoding with concatenated matching decoding. It moves the BP predecoding
logic from ColorCode into a dedicated decoder class for better modularity.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .base import BaseDecoder
from .bp_decoder import BPDecoder
from .concat_matching_decoder import ConcatMatchingDecoder
from ..dem_utils.dem_manager import DemManager
from ..stim_utils import dem_to_parity_check, remove_obs_from_dem


class BeliefConcatMatchingDecoder(BaseDecoder):
    """
    Composite decoder combining belief propagation pre-decoding with concatenated matching.
    
    This decoder implements the sophisticated workflow where belief propagation is used
    as a pre-decoding step to update DEM probabilities, followed by concatenated matching
    decoding using the updated probabilities. This approach can improve decoding performance
    by incorporating soft information from BP.
    
    Key Features:
    - BP pre-decoding with probability updates
    - Seamless integration with ConcatMatchingDecoder
    - Batch processing support
    - Full backward compatibility with existing parameters
    - Numerical stability with probability clipping
    
    Attributes
    ----------
    dem_manager : DemManager
        Manager for detector error models and decompositions
    circuit_type : str
        Type of circuit being decoded
    num_obs : int
        Number of observables
    comparative_decoding : bool
        Whether comparative decoding is enabled
    bp_decoder : BPDecoder
        Internal belief propagation decoder instance
    concat_decoder : ConcatMatchingDecoder
        Internal concatenated matching decoder instance
    """
    
    def __init__(
        self,
        dem_manager: DemManager,
        circuit_type: str,
        num_obs: int,
        comparative_decoding: bool = False,
        bp_cache_inputs: bool = True,
    ):
        """
        Initialize the belief concatenated matching decoder.
        
        Parameters
        ----------
        dem_manager : DemManager
            Manager providing access to decomposed DEMs and matrices
        circuit_type : str
            Type of circuit (tri, rec, rec_stability, growing, cult+growing)
        num_obs : int
            Number of observables in the quantum code
        comparative_decoding : bool, default False
            Whether to enable comparative decoding for logical gap calculation
        bp_cache_inputs : bool, default True
            Whether to cache BP inputs for efficiency
        """
        self.dem_manager = dem_manager
        self.circuit_type = circuit_type
        self.num_obs = num_obs
        self.comparative_decoding = comparative_decoding
        
        # Create internal decoder instances
        self.bp_decoder = BPDecoder(
            dem_manager=dem_manager,
            comparative_decoding=comparative_decoding,
            cache_inputs=bp_cache_inputs,
        )
        
        self.concat_decoder = ConcatMatchingDecoder(
            dem_manager=dem_manager,
            circuit_type=circuit_type,
            num_obs=num_obs,
            comparative_decoding=comparative_decoding,
        )
        
        # Cache for BP inputs (for compatibility with existing code)
        self._bp_inputs: Dict[str, np.ndarray] = {}
    
    def supports_comparative_decoding(self) -> bool:
        """Return True - this decoder supports comparative decoding."""
        return True
    
    def supports_predecoding(self) -> bool:
        """Return True - this decoder supports pre-decoding strategies."""
        return True
    
    def decode(
        self,
        detector_outcomes: np.ndarray,
        colors: Union[str, List[str]] = "all",
        logical_value: Union[bool, Sequence[bool], None] = None,
        bp_prms: Optional[Dict] = None,
        erasure_matcher_predecoding: bool = False,
        partial_correction_by_predecoding: bool = False,
        full_output: bool = False,
        check_validity: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Decode detector outcomes using BP pre-decoding + concatenated MWPM decoding.
        
        This method first runs belief propagation to obtain soft information (log-likelihood
        ratios), converts these to probabilities, updates the DEM probabilities accordingly,
        and then runs concatenated matching decoding with the updated DEMs.
        
        Parameters
        ----------
        detector_outcomes : np.ndarray
            1D or 2D array of detector measurement outcomes.
            If 1D, interpreted as a single sample.
            If 2D, each row is a sample, each column a detector.
        colors : str or list of str, default 'all'
            Colors to use for decoding. Can be 'all', one of {'r', 'g', 'b'},
            or a list containing any combination of {'r', 'g', 'b'}.
        logical_value : bool or sequence of bool, optional
            Logical value(s) to use for decoding. If None and comparative_decoding
            is True, all possible logical value combinations will be tested.
        bp_prms : dict, optional
            Parameters for the belief propagation decoder (e.g., max_iter).
        erasure_matcher_predecoding : bool, default False
            Whether to use erasure matcher as a pre-decoding step.
        partial_correction_by_predecoding : bool, default False
            Whether to apply partial correction from erasure matcher predecoding.
        full_output : bool, default False
            Whether to return extra information about the decoding process.
        check_validity : bool, default False
            Whether to check the validity of predicted error patterns.
        verbose : bool, default False
            Whether to print additional information during decoding.
        **kwargs
            Additional parameters for compatibility.
            
        Returns
        -------
        np.ndarray or tuple
            If full_output is False: predicted observables as bool array.
            If full_output is True: tuple of (predictions, extra_outputs_dict).
        """
        if bp_prms is None:
            bp_prms = {}
        
        # Ensure detector_outcomes is 2D for batch processing
        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        original_shape_1d = detector_outcomes.ndim == 1
        if original_shape_1d:
            detector_outcomes = detector_outcomes.reshape(1, -1)
        
        # Process color selection
        if colors == "all":
            colors = ["r", "g", "b"]
        elif colors in ["r", "g", "b"]:
            colors = [colors]
        
        if verbose:
            print("Running BP pre-decoding...")
        
        # Step 1: Run BP pre-decoding to get log-likelihood ratios
        _, llrs, _ = self.bp_decoder.decode(detector_outcomes, **bp_prms)
        
        # Step 2: Convert LLRs to probabilities with numerical stability
        bp_probs = 1 / (1 + np.exp(llrs))
        eps = 1e-14
        bp_probs = bp_probs.clip(eps, 1 - eps)
        
        # Update BP inputs cache (for compatibility with existing code)
        self._update_bp_inputs_cache()
        
        # Step 3: Process each sample with updated DEM probabilities
        results = []
        extra_outputs_list = []
        
        for i, det_outcomes_single in enumerate(detector_outcomes):
            if verbose:
                print(f"Processing sample {i+1}/{len(detector_outcomes)} with updated DEMs...")
            
            # Step 4: Create updated DEMs using BP probabilities
            updated_dems = self._create_updated_dems(colors, bp_probs[i] if bp_probs.ndim > 1 else bp_probs)
            
            # Step 5: Run concatenated matching with updated DEMs
            sample_result = self._decode_with_updated_dems(
                det_outcomes_single.reshape(1, -1),
                updated_dems,
                colors=colors,
                logical_value=logical_value,
                erasure_matcher_predecoding=erasure_matcher_predecoding,
                partial_correction_by_predecoding=partial_correction_by_predecoding,
                full_output=full_output,
                check_validity=check_validity,
                verbose=verbose,
            )
            
            if full_output:
                obs_pred, extra_output = sample_result
                results.append(obs_pred)
                extra_outputs_list.append(extra_output)
            else:
                results.append(sample_result)
        
        # Step 6: Aggregate results
        final_results = np.concatenate(results, axis=0)
        
        if original_shape_1d and final_results.ndim > 1 and final_results.shape[0] == 1:
            final_results = final_results.ravel()
        
        if full_output:
            # Aggregate extra outputs
            aggregated_extra = {}
            for key in extra_outputs_list[0].keys():
                try:
                    aggregated_extra[key] = np.concatenate([eo[key] for eo in extra_outputs_list], axis=0)
                except (ValueError, np.AxisError):
                    # Handle cases where concatenation fails (e.g., different shapes)
                    aggregated_extra[key] = [eo[key] for eo in extra_outputs_list]
            
            return final_results, aggregated_extra
        else:
            return final_results
    
    def _update_bp_inputs_cache(self):
        """Update BP inputs cache for compatibility with existing code."""
        if not self._bp_inputs:
            if self.comparative_decoding:
                dem = remove_obs_from_dem(self.dem_manager.dem_xz)
            else:
                dem = self.dem_manager.dem_xz
            H, _, p = dem_to_parity_check(dem)
            self._bp_inputs["H"] = H
            self._bp_inputs["p"] = p
    
    def _create_updated_dems(self, colors: List[str], bp_probs: np.ndarray) -> Dict[str, Tuple[Tuple, Tuple]]:
        """
        Create updated DEM matrices and probabilities using BP probabilities for each color.
        
        Parameters
        ----------
        colors : list of str
            Colors to create updated DEMs for
        bp_probs : np.ndarray
            BP probabilities to use for DEM updates
            
        Returns
        -------
        dict
            Dictionary mapping color to ((H1, p1), (H2, p2)) tuples where:
            - H1, H2 are parity check matrices for stages 1 and 2
            - p1, p2 are probability arrays for stages 1 and 2
        """
        updated_dem_data = {}
        
        for c in colors:
            dem1_sym, dem2_sym = self.dem_manager.dems_decomposed[c].dems_symbolic
            
            # Create updated DEMs using BP probabilities  
            dem1, _ = dem1_sym.to_dem(bp_probs)
            dem2, _ = dem2_sym.to_dem(bp_probs, sort=True)
            
            # Extract parity check matrices and probabilities
            H1, _, p1 = dem_to_parity_check(dem1)
            H2, _, p2 = dem_to_parity_check(dem2)
            
            # Store in the format expected by ConcatMatchingDecoder
            updated_dem_data[c] = ((H1, p1), (H2, p2))
        
        return updated_dem_data
    
    def _decode_with_updated_dems(
        self,
        detector_outcomes: np.ndarray,
        updated_dem_data: Dict[str, Tuple[Tuple, Tuple]],
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Run concatenated matching decoding with updated DEM data.
        
        This method passes custom DEM matrices and probabilities to the
        ConcatMatchingDecoder, avoiding any modification of shared state.
        
        Parameters
        ----------
        detector_outcomes : np.ndarray
            Single sample detector outcomes (2D with shape (1, n_detectors))
        updated_dem_data : dict
            Dictionary mapping color to ((H1, p1), (H2, p2)) tuples
        **kwargs
            Additional arguments to pass to the concatenated decoder
            
        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, dict]]
            Decoding results from concatenated matching decoder
        """
        # Pass custom DEM data to the concatenated decoder
        return self.concat_decoder.decode(
            detector_outcomes, 
            custom_dem_data=updated_dem_data,
            **kwargs
        )
    
    def get_bp_decoder(self) -> BPDecoder:
        """Get access to the internal BP decoder."""
        return self.bp_decoder
    
    def get_concat_decoder(self) -> ConcatMatchingDecoder:
        """Get access to the internal concatenated matching decoder."""
        return self.concat_decoder