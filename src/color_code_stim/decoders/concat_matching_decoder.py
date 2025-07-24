"""
Concatenated Matching Decoder for Color Code

This module implements the concatenated minimum-weight perfect matching (MWPM) decoder
for color codes. It performs two-stage decoding with color-based decomposition,
supporting comparative decoding and advanced pre-decoding strategies.
"""

import itertools
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pymatching

from .base import BaseDecoder
from ..config import COLOR_LABEL, color_to_color_val
from ..dem_utils.dem_manager import DemManager
from ..utils import _get_final_predictions


class ConcatMatchingDecoder(BaseDecoder):
    """
    Concatenated minimum-weight perfect matching decoder for color codes.

    This decoder implements the sophisticated concatenated decoding strategy where
    each color is decoded in two stages, and the results are combined to find the
    minimum-weight error correction. It supports comparative decoding for magic
    state distillation and various pre-decoding strategies.

    Key Features:
    - Two-stage MWPM decoding per color (stage 1: local errors, stage 2: global errors)
    - Comparative decoding: test all logical classes, return minimum weight
    - Logical gap calculation for post-selection in magic state distillation
    - Erasure matcher pre-decoding for improved performance
    - BP pre-decoding integration (when available)
    - Color-specific decoding with flexible color selection

    Attributes
    ----------
    dem_manager : DEMManager
        Manager for detector error models and decompositions
    circuit_type : str
        Type of circuit being decoded
    num_obs : int
        Number of observables
    comparative_decoding : bool
        Whether comparative decoding is enabled
    """

    def __init__(
        self,
        dem_manager: DemManager,
    ):
        """
        Initialize the concatenated matching decoder.

        Parameters
        ----------
        dem_manager : DEMManager
            Manager providing access to decomposed DEMs and matrices
        """
        self.dem_manager = dem_manager
        self.circuit_type = dem_manager.circuit_type
        self.num_obs = dem_manager.circuit.num_observables
        self.comparative_decoding = dem_manager.comparative_decoding

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
        erasure_matcher_predecoding: bool = False,
        partial_correction_by_predecoding: bool = False,
        full_output: bool = False,
        check_validity: bool = False,
        verbose: bool = False,
        custom_dem_data: Optional[Dict[str, Tuple[Tuple, Tuple]]] = None,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Decode detector outcomes using concatenated MWPM decoding.

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
        custom_dem_data : dict, optional
            Custom DEM matrices and probabilities for BP predecoding.
            Format: {color: ((H1, p1), (H2, p2))} where H1,H2 are parity check
            matrices and p1,p2 are probability arrays for stages 1 and 2.
        **kwargs
            Additional parameters (for compatibility).

        Returns
        -------
        np.ndarray or tuple
            If full_output is False: predicted observables as bool array.
            If full_output is True: tuple of (predictions, extra_outputs_dict).
        """
        if erasure_matcher_predecoding:
            if not self.comparative_decoding:
                raise ValueError(
                    "Erasure matcher predecoding requires comparative_decoding=True"
                )

        # Ensure detector_outcomes is 2D
        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim == 1:
            detector_outcomes = detector_outcomes.reshape(1, -1)

        # Process color selection
        if colors == "all":
            colors = ["r", "g", "b"]
        elif colors in ["r", "g", "b"]:
            colors = [colors]

        # Generate all logical value combinations for comparative decoding
        all_logical_values = np.array(
            list(itertools.product([False, True], repeat=self.num_obs))
        )

        if logical_value is not None:
            logical_value = np.asarray(logical_value, dtype=bool).ravel()
            if len(logical_value) != self.num_obs:
                raise ValueError(f"logical_value must have length {self.num_obs}")

        # Handle cultivation circuit post-selection
        if self.circuit_type == "cult+growing":
            cult_interface_det_ids = (
                self.dem_manager.cult_detector_ids
                + self.dem_manager.interface_detector_ids
            )
            cult_success = ~np.any(detector_outcomes[:, cult_interface_det_ids], axis=1)
            detector_outcomes = detector_outcomes[cult_success, :]

        # Determine number of logical classes to test
        num_logical_classes = (
            len(all_logical_values)
            if self.comparative_decoding and logical_value is None
            else 1
        )

        # Stage 1 decoding for all logical classes and colors
        error_preds_stage1_all = []
        if verbose:
            print("First-round decoding:")

        for i in range(num_logical_classes):
            error_preds_stage1_all.append({})
            for c in colors:
                if verbose:
                    print(f"    > logical class {i}, color {c}...")

                if self.comparative_decoding:
                    detector_outcomes_copy = detector_outcomes.copy()
                    if logical_value is not None:
                        detector_outcomes_copy[:, -self.num_obs :] = logical_value
                    else:
                        detector_outcomes_copy[:, -self.num_obs :] = all_logical_values[
                            i
                        ]
                    error_preds_stage1_all[i][c] = self._decode_stage1(
                        detector_outcomes_copy, c, custom_dem_data
                    )
                else:
                    error_preds_stage1_all[i][c] = self._decode_stage1(
                        detector_outcomes, c, custom_dem_data
                    )

        # Erasure matcher predecoding
        if erasure_matcher_predecoding:
            if len(error_preds_stage1_all) <= 1:
                raise ValueError(
                    "Erasure matcher predecoding requires multiple logical classes"
                )

            if verbose:
                print("Erasure matcher predecoding:")

            (
                predecoding_obs_preds,
                predecoding_error_preds,
                predecoding_weights,
                predecoding_success,
            ) = self._erasure_matcher_predecoding(
                error_preds_stage1_all, detector_outcomes
            )

            predecoding_failure = ~predecoding_success
            detector_outcomes_left = detector_outcomes[predecoding_failure, :]
            error_preds_stage1_left = [
                {
                    c: arr[predecoding_failure, :]
                    for c, arr in error_preds_stage1_all[i].items()
                }
                for i in range(len(error_preds_stage1_all))
            ]

            if verbose:
                print(
                    f"    > # of samples with successful predecoding: {predecoding_success.sum()}"
                )
        else:
            detector_outcomes_left = detector_outcomes
            error_preds_stage1_left = error_preds_stage1_all

        # Stage 2 decoding
        if verbose:
            print("Second-round decoding:")

        num_left_samples = detector_outcomes_left.shape[0]

        if num_left_samples > 0 and not (
            erasure_matcher_predecoding and partial_correction_by_predecoding
        ):
            num_errors = self.dem_manager.H.shape[1]

            error_preds = np.empty(
                (num_logical_classes, len(colors), num_left_samples, num_errors),
                dtype=bool,
            )
            weights = np.empty(
                (num_logical_classes, len(colors), num_left_samples), dtype=float
            )

            for i in range(len(error_preds_stage1_left)):
                for i_c, c in enumerate(colors):
                    if verbose:
                        print(f"    > logical class {i}, color {c}...")

                    if self.comparative_decoding:
                        detector_outcomes_copy = detector_outcomes_left.copy()
                        if logical_value is not None:
                            detector_outcomes_copy[:, -self.num_obs :] = logical_value
                        else:
                            detector_outcomes_copy[:, -self.num_obs :] = (
                                all_logical_values[i]
                            )
                        error_preds_new, weights_new = self._decode_stage2(
                            detector_outcomes_copy,
                            error_preds_stage1_left[i][c],
                            c,
                            custom_dem_data,
                        )
                    else:
                        error_preds_new, weights_new = self._decode_stage2(
                            detector_outcomes_left,
                            error_preds_stage1_left[i][c],
                            c,
                            custom_dem_data,
                        )

                    # Map errors back to original DEM ordering
                    error_preds_new = self.dem_manager.dems_decomposed[
                        c
                    ].map_errors_to_org_dem(error_preds_new, stage=2)

                    error_preds[i, i_c, :, :] = error_preds_new
                    weights[i, i_c, :] = weights_new

            # Find best predictions across logical classes and colors
            best_logical_classes, best_color_inds, weights_final, logical_gaps = (
                _get_final_predictions(weights)
            )

            error_preds_final = error_preds[
                best_logical_classes, best_color_inds, np.arange(num_left_samples), :
            ]

            # Calculate observable predictions
            if self.comparative_decoding:
                if logical_value is None:
                    obs_preds_final = all_logical_values[best_logical_classes]
                    if obs_preds_final.shape != (num_left_samples, self.num_obs):
                        raise RuntimeError("Observable prediction shape mismatch")
                else:
                    obs_preds_final = np.tile(logical_value, (num_left_samples, 1))
            else:
                obs_preds_final = np.empty((num_left_samples, self.num_obs), dtype=bool)
                for i_c, c in enumerate(colors):
                    obs_matrix = self.dem_manager.obs_matrix
                    mask = best_color_inds == i_c
                    obs_preds_final[mask, :] = (
                        (error_preds_final[mask, :].astype("uint8") @ obs_matrix.T) % 2
                    ).astype(bool)

            # Adjust color indices for non-standard color selections
            if colors == ["r", "g", "b"]:
                best_colors = best_color_inds
            else:
                best_colors = np.array([color_to_color_val(c) for c in colors])[
                    best_color_inds
                ]

        elif (
            num_left_samples > 0
            and erasure_matcher_predecoding
            and partial_correction_by_predecoding
        ):
            # Partial correction strategy
            predecoding_error_preds_failed = predecoding_error_preds[
                predecoding_failure, :
            ].astype("uint8")

            def get_partial_corr(matrix):
                corr = (predecoding_error_preds_failed @ matrix.T) % 2
                return corr.astype(bool)

            obs_partial_corr = get_partial_corr(self.dem_manager.obs_matrix)
            det_partial_corr = get_partial_corr(self.dem_manager.H)
            detector_outcomes_left ^= det_partial_corr

            # Recursive call with partial correction
            obs_preds_final = self.decode(
                detector_outcomes_left,
                colors=colors,
                full_output=full_output,
            )
            if full_output:
                obs_preds_final, extra_outputs = obs_preds_final
            else:
                extra_outputs = {}

            if obs_preds_final.ndim == 1:
                obs_preds_final = obs_preds_final[:, np.newaxis]

            if full_output:
                error_preds_final = extra_outputs["error_preds"]
                best_colors = extra_outputs["best_colors"]
                weights_final = extra_outputs["weights"]
                logical_gaps = extra_outputs["logical_gaps"]

        else:
            # No samples to decode
            error_preds_final = np.array([[]], dtype=bool)
            obs_preds_final = np.array([[]], dtype=bool)
            best_colors = np.array([], dtype=np.uint8)
            weights_final = np.array([], dtype=float)
            logical_gaps = np.array([], dtype=float)

        # Merge predecoding and second-round results
        if erasure_matcher_predecoding and np.any(predecoding_success):
            if verbose:
                print("Merging predecoding & second-round decoding outcomes")

            full_obs_preds_final = predecoding_obs_preds.copy()
            if full_output:
                full_best_colors = np.full(detector_outcomes.shape[0], "P")
                full_weights_final = predecoding_weights.copy()
                full_logical_gaps = np.full(detector_outcomes.shape[0], -1)
                full_error_preds_final = predecoding_error_preds.copy()

            if detector_outcomes_left.shape[0] > 0:
                if partial_correction_by_predecoding:
                    obs_preds_final ^= obs_partial_corr
                    if full_output:
                        error_preds_final ^= predecoding_error_preds_failed.astype(bool)

                full_obs_preds_final[predecoding_failure, :] = obs_preds_final

                if full_output:
                    full_best_colors[predecoding_failure] = best_colors
                    full_weights_final[predecoding_failure] = weights_final
                    full_logical_gaps[predecoding_failure] = logical_gaps
                    full_error_preds_final[predecoding_failure, :] = error_preds_final

            obs_preds_final = full_obs_preds_final
            if full_output:
                best_colors = full_best_colors
                weights_final = full_weights_final
                logical_gaps = full_logical_gaps
                error_preds_final = full_error_preds_final

        # Validity checking
        if check_validity:
            det_preds = (
                error_preds_final.astype("uint8") @ self.dem_manager.H.T % 2
            ).astype(bool)
            validity = np.all(det_preds == detector_outcomes, axis=1)
            if verbose:
                if np.all(validity):
                    print("All predictions are valid")
                else:
                    print(f"{np.sum(~validity)} invalid predictions found!")

        # Format output
        if obs_preds_final.shape[1] == 1:
            obs_preds_final = obs_preds_final.ravel()

        if full_output:
            extra_outputs = {
                "best_colors": best_colors,
                "weights": weights_final,
                "error_preds": error_preds_final,
            }

            if len(error_preds_stage1_all) > 1:
                extra_outputs["logical_gaps"] = logical_gaps
                extra_outputs["logical_values"] = all_logical_values
                if erasure_matcher_predecoding:
                    extra_outputs["erasure_matcher_success"] = predecoding_success
                    extra_outputs["predecoding_error_preds"] = predecoding_error_preds
                    extra_outputs["predecoding_obs_preds"] = predecoding_obs_preds

            if self.circuit_type == "cult+growing":
                extra_outputs["cult_success"] = cult_success

            if check_validity:
                extra_outputs["validity"] = validity

            return obs_preds_final, extra_outputs
        else:
            return obs_preds_final

    def _decode_stage1(
        self,
        detector_outcomes: np.ndarray,
        color: str,
        custom_dem_data: Optional[Dict[str, Tuple[Tuple, Tuple]]] = None,
    ) -> np.ndarray:
        """
        Perform stage 1 decoding for a specific color.

        Stage 1 focuses on local error patterns within each color's subspace.

        Parameters
        ----------
        detector_outcomes : np.ndarray
            2D array of detector outcomes
        color : str
            Color to decode ('r', 'g', or 'b')

        Returns
        -------
        np.ndarray
            Stage 1 error predictions
        """
        det_outcomes_dem1 = detector_outcomes.copy()

        # Use custom DEM data if provided, otherwise use DEM manager data
        if custom_dem_data and color in custom_dem_data:
            H, p = custom_dem_data[color][0]  # Stage 1 data (H1, p1)
        else:
            H, p = (
                self.dem_manager.dems_decomposed[color].Hs[0],
                self.dem_manager.dems_decomposed[color].probs[0],
            )

        # Remove empty checks
        checks_to_keep = H.tocsr().getnnz(axis=1) > 0
        det_outcomes_dem1 = det_outcomes_dem1[:, checks_to_keep]
        H = H[checks_to_keep, :]

        # MWPM decoding
        weights = np.log((1 - p) / p)
        matching = pymatching.Matching.from_check_matrix(H, weights=weights)
        preds_dem1 = matching.decode_batch(det_outcomes_dem1)

        del det_outcomes_dem1, matching
        return preds_dem1

    def _decode_stage2(
        self,
        detector_outcomes: np.ndarray,
        preds_dem1: np.ndarray,
        color: COLOR_LABEL,
        custom_dem_data: Optional[Dict[str, Tuple[Tuple, Tuple]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform stage 2 decoding for a specific color.

        Stage 2 combines stage 1 predictions with remaining detector information
        to find global error patterns.

        Parameters
        ----------
        detector_outcomes : np.ndarray
            2D array of detector outcomes
        preds_dem1 : np.ndarray
            Stage 1 error predictions
        color : COLOR_LABEL
            Color to decode ('r', 'g', or 'b')

        Returns
        -------
        tuple
            (error_predictions, weights) from stage 2 decoding
        """
        det_outcome_dem2 = detector_outcomes.copy()

        # Mask out detectors not belonging to this color
        mask = np.full_like(det_outcome_dem2, True)
        mask[:, self.dem_manager.detector_ids_by_color[color]] = False
        det_outcome_dem2[mask] = False
        del mask

        # Combine with stage 1 predictions
        det_outcome_dem2 = np.concatenate([det_outcome_dem2, preds_dem1], axis=1)

        # Stage 2 MWPM decoding
        # Use custom DEM data if provided, otherwise use DEM manager data
        if custom_dem_data and color in custom_dem_data:
            H, p = custom_dem_data[color][1]  # Stage 2 data (H2, p2)
        else:
            H, p = (
                self.dem_manager.dems_decomposed[color].Hs[1],
                self.dem_manager.dems_decomposed[color].probs[1],
            )
        weights = np.log((1 - p) / p)
        matching = pymatching.Matching.from_check_matrix(H, weights=weights)
        preds, weights_new = matching.decode_batch(
            det_outcome_dem2, return_weights=True
        )

        return preds, weights_new

    def _find_error_set_intersection(
        self,
        preds_dem1: Dict[COLOR_LABEL, np.ndarray],
    ) -> np.ndarray:
        """
        Find the intersection of error sets from different colors.

        This method identifies errors that are consistent across all color
        predictions, used in erasure matcher predecoding.

        Parameters
        ----------
        preds_dem1 : dict
            Stage 1 predictions for each color

        Returns
        -------
        np.ndarray
            Boolean array indicating error set intersection
        """
        possible_errors = []
        for c in ["r", "g", "b"]:
            preds_dem1_c = preds_dem1[c]
            error_map_matrix = (
                self.dem_manager.dems_decomposed[c].dems_symbolic[0].error_map_matrix
            )
            possible_errors_c = (preds_dem1_c.astype("uint8") @ error_map_matrix) > 0
            possible_errors.append(possible_errors_c)

        possible_errors = np.stack(possible_errors, axis=-1)
        error_set_intersection = np.all(possible_errors, axis=-1).astype(bool)

        return error_set_intersection

    def _erasure_matcher_predecoding(
        self,
        preds_dem1_all: List[Dict[COLOR_LABEL, np.ndarray]],
        detector_outcomes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform erasure matcher predecoding.

        This advanced predecoding strategy finds error predictions that are
        consistent across all colors and logical classes, providing high-confidence
        corrections before the main decoding stage.

        Parameters
        ----------
        preds_dem1_all : list
            Stage 1 predictions for all logical classes and colors
        detector_outcomes : np.ndarray
            Original detector outcomes

        Returns
        -------
        tuple
            (obs_preds, error_preds, weights, validity) from predecoding
        """
        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)

        # Generate all logical value combinations
        all_logical_values = list(itertools.product([False, True], repeat=self.num_obs))
        all_logical_values = np.array(all_logical_values)

        # Calculate error set intersection and weights for each logical class
        error_preds_all = []
        weights_all = []
        for preds_dem1 in preds_dem1_all:
            error_preds = self._find_error_set_intersection(preds_dem1)
            llrs_all = np.log(
                (1 - self.dem_manager.probs_xz) / self.dem_manager.probs_xz
            )
            llrs = np.zeros_like(error_preds, dtype=float)
            llrs[error_preds] = llrs_all[np.where(error_preds)[1]]
            weights = llrs.sum(axis=1)
            error_preds_all.append(error_preds)
            weights_all.append(weights)

        # Stack results
        error_preds_all = np.stack(error_preds_all, axis=1)
        weights_all = np.stack(weights_all, axis=1)
        num_samples = error_preds_all.shape[0]

        # Sort logical classes by prediction weight
        inds_logical_class_sorted = np.argsort(weights_all, axis=1)

        # Sort error predictions and weights by weight
        error_preds_all_sorted = error_preds_all[
            np.arange(num_samples)[:, np.newaxis], inds_logical_class_sorted
        ].astype("uint8")

        weights_all_sorted = np.take_along_axis(
            weights_all, inds_logical_class_sorted, axis=1
        )

        # Check validity (match with detectors and observables)
        match_with_dets = np.all(
            ((error_preds_all_sorted @ self.dem_manager.H.T.toarray()) % 2).astype(bool)
            == detector_outcomes[:, np.newaxis, :],
            axis=-1,
        )

        logical_classes_sorted = all_logical_values[inds_logical_class_sorted]
        match_with_obss = np.all(
            (
                (error_preds_all_sorted @ self.dem_manager.obs_matrix.T.toarray()) % 2
            ).astype(bool)
            == logical_classes_sorted,
            axis=-1,
        )

        validity_full = match_with_dets & match_with_obss

        # Find first valid prediction for each sample
        inds_first_valid_logical_classes = np.argmax(validity_full, axis=1)
        obs_preds = logical_classes_sorted[
            np.arange(num_samples), inds_first_valid_logical_classes, :
        ]
        validity = np.any(validity_full, axis=1)

        # Extract weights and error predictions
        weights = weights_all_sorted[
            np.arange(num_samples), inds_first_valid_logical_classes
        ]
        weights[~validity] = np.inf

        error_preds = error_preds_all_sorted[
            np.arange(num_samples), inds_first_valid_logical_classes, :
        ]

        return obs_preds, error_preds, weights, validity
