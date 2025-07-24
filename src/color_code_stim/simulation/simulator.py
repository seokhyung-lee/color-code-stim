"""
Simulator module for color code quantum error correction.

This module provides the Simulator class that handles Monte Carlo simulation,
detector/observable sampling, and related utility functions extracted from
the ColorCode class to support the modular architecture.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import stim
import igraph as ig
from scipy.sparse import csc_matrix
from statsmodels.stats.proportion import proportion_confint

from ..config import PAULI_LABEL, COLOR_LABEL, color_val_to_color
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
        num_obs: int,
        detectors_checks_map: List[Tuple[ig.Vertex, int]],
        obs_paulis: List[PAULI_LABEL],
        tanner_graph: ig.Graph,
        physical_probs: Dict[str, float],
        rounds: int,
    ):
        """
        Initialize the Simulator with required dependencies.
        
        Parameters
        ----------
        circuit : stim.Circuit
            The quantum circuit to simulate
        circuit_type : str
            Type of circuit ('tri', 'rec', 'rec_stability', 'growing', 'cult+growing')
        num_obs : int
            Number of observables in the circuit
        detectors_checks_map : List[Tuple[ig.Vertex, int]]
            Mapping from detector IDs to (vertex, round) pairs
        obs_paulis : List[PAULI_LABEL]
            Pauli types for each observable
        tanner_graph : ig.Graph
            The Tanner graph representation
        physical_probs : Dict[str, float]
            Physical error probabilities by type
        rounds : int
            Number of syndrome extraction rounds
        """
        self.circuit = circuit
        self.circuit_type = circuit_type
        self.num_obs = num_obs
        self.detectors_checks_map = detectors_checks_map
        self.obs_paulis = obs_paulis
        self.tanner_graph = tanner_graph
        self.physical_probs = physical_probs
        self.rounds = rounds
    
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
    
    def get_detector_type(self, detector_id: int) -> Tuple[PAULI_LABEL, COLOR_LABEL]:
        """
        Get the Pauli and color type of a detector.
        
        Parameters
        ----------
        detector_id : int
            Detector ID to query
            
        Returns
        -------
        pauli : PAULI_LABEL
            Pauli type of the detector ('X', 'Y', or 'Z')
        color : COLOR_LABEL
            Color of the detector ('r', 'g', or 'b')
        """
        coords = self.circuit.get_detector_coordinates(only=[detector_id])[detector_id]
        pauli = coords[3]
        if pauli == 0:
            pauli = "X"
        elif pauli == 1:
            pauli = "Y"
        elif pauli == 2:
            pauli = "Z"
        else:
            raise ValueError(f"Invalid pauli: {pauli}")
        color = color_val_to_color(coords[4])

        return pauli, color

    def get_observable_pauli(self, observable_id: int) -> PAULI_LABEL:
        """
        Get the Pauli type of an observable.
        
        Parameters
        ----------
        observable_id : int
            Observable ID to query
            
        Returns
        -------
        PAULI_LABEL
            Pauli type of the observable
        """
        return self.obs_paulis[observable_id]

    def get_detector(self, detector_id: int) -> Tuple[ig.Vertex, int]:
        """
        Get the ancillary qubit and round corresponding to a detector from a
        given detector ID.

        Parameters
        ----------
        detector_id : int
            Detector ID.

        Returns
        -------
        anc : ig.Vertex
            Ancillary qubit involved in the detector.
        round : int
            Round that the detector belongs to.
        """
        try:
            return self.detectors_checks_map[detector_id]
        except IndexError:
            raise ValueError(f"Detector ID {detector_id} not found.")
    
    def errors_to_qubits(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        """
        Convert errors (generated by `sample_with_errors`) or error predictions
        (generated by decoders) into the corresponding data qubit indices.

        Available only for `tri` and `rec` circuit types with `rounds=1` under
        bit-flip noise (i.e., probabilities besides `p_bitflip` are 0).

        Note: Errors and error predictions from `sample_with_errors` or decoders
        follow the ordering of error mechanisms in the circuit's detector error model
        (`circuit.detector_error_model()`). This function is necessary because this 
        ordering differs from the data qubit ordering in the tanner graph 
        (`tanner_graph.vs.select(pauli=None)`). This conversion is especially helpful 
        when visualizing errors or error predictions on the lattice.

        Parameters
        ----------
        errors : 2D numpy array of bool
            Errors following the ordering of error mechanisms in the DEM of the circuit
            `circuit.detector_error_model()`.

        Returns
        -------
        errors_qubits : 2D numpy array of bool
            Errors following the ordering of data qubits in the tanner graph
            `tanner_graph.vs.select(pauli=None)`.
        """

        if self.circuit_type not in {"tri", "rec"}:
            raise NotImplementedError(
                f'errors_to_qubits is not available for "{self.circuit_type}" circuit type.'
            )

        if self.rounds != 1:
            raise NotImplementedError(
                "errors_to_qubits is only available when rounds = 1."
            )

        if any(
            prob > 0 for key, prob in self.physical_probs.items() if key != "bitflip"
        ):
            raise NotImplementedError(
                "errors_to_qubits is only available under bit-flip noise "
                "(only p_bitflip is nonzero)."
            )

        # set of ancillary qubits - data qubit mapping
        anc_qids_to_data_qubit_idx = {}
        data_qubits = self.tanner_graph.vs.select(pauli=None)
        for i_dq, data_qubit in enumerate(data_qubits):
            data_qubit: ig.Vertex
            connected_anc_qubits = data_qubit.neighbors()
            connected_anc_qubits = [
                q for q in connected_anc_qubits if q["pauli"] == "Z"
            ]
            key = frozenset(q.index for q in connected_anc_qubits)
            assert key not in anc_qids_to_data_qubit_idx
            anc_qids_to_data_qubit_idx[key] = i_dq

        # data qubit mapping - error mechanism in DEM
        dem = self.circuit.detector_error_model()
        data_qubit_idx_to_em = np.full(len(data_qubits), -1, dtype="int32")
        for i_em, em in enumerate(dem):
            if em.type == "error":
                det_ids = [
                    int(str(target)[1:])
                    for target in em.targets_copy()
                    if target.is_relative_detector_id()
                ]
                anc_qids = [
                    self.get_detector(det_id)[0].index
                    for det_id in det_ids
                    if det_id < len(self.detectors_checks_map)
                ]
                anc_qids = frozenset(anc_qids)
                data_qubit_idx = anc_qids_to_data_qubit_idx[anc_qids]
                if data_qubit_idx_to_em[data_qubit_idx] != -1:
                    raise ValueError(
                        f"Data qubit {data_qubit_idx} is mapped to multiple error mechanisms: {data_qubit_idx_to_em[data_qubit_idx]} and {i_em}"
                    )
                data_qubit_idx_to_em[data_qubit_idx] = i_em
        assert np.all(data_qubit_idx_to_em != -1)

        return errors[..., data_qubit_idx_to_em]

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