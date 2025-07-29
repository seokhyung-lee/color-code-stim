from typing import List, Literal, Tuple, Iterator

import numpy as np
import stim
from scipy.sparse import csc_matrix, csr_matrix, spmatrix

from ..stim_symbolic import _DemSymbolic
from ..stim_utils import dem_to_parity_check

COLOR_LABEL = Literal["r", "g", "b"]


class DemDecomp:
    """
    Decomposition of a detector error model (DEM) into two stages for concatenated color
    code decoding.

    This class decomposes a detector error model into a restricted DEM (stage 1) and
    a monochromatic DEM (stage 2) for a specific color in the color code.

    Attributes
    ----------
    color : one of {"r", "g", "b"}
        The color for which the DEM is decomposed.
    _dems : 2-tuple of stim.DetectorErrorModel
        The decomposed detector error models for stages 1 and 2.
        Can be accessed simply by `self[0]` and `self[1]`.
    dems_symbolic : 2-tuple of _DemSymbolic
        Symbolic representations of the decomposed DEMs.
    Hs : 2-tuple of csc_matrix (bool)
        Parity check matrices for stages 1 and 2.
    probs : 2-tuple of 1D numpy array (float)
        Error probabilities for stages 1 and 2.
    org_dem : stim.DetectorErrorModel
        The original detector error model.
    org_prob : 1D numpy array (float)
        Error probabilities of the original DEM.
    error_map_matrices : 2-tuple of csr_matrix (bool)
        Matrices mapping errors in decomposed DEMs to errors in original DEM.
    """

    color: COLOR_LABEL
    _dems: Tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]
    dems_symbolic: Tuple[_DemSymbolic, _DemSymbolic]
    Hs: Tuple[csc_matrix, csc_matrix]
    probs: Tuple[np.ndarray, np.ndarray]
    obs_matrix_stage2: csc_matrix
    org_dem: stim.DetectorErrorModel
    org_prob: np.ndarray
    error_map_matrices: Tuple[
        csr_matrix, csr_matrix
    ]  # Each: (# of errors in decomp DEM, # of errors in org DEM), bool
    # _best_org_error_map: Tuple[np.ndarray, np.ndarray]

    def __init__(
        self,
        *,
        org_dem: stim.DetectorErrorModel,
        color: COLOR_LABEL,
        remove_non_edge_like_errors: bool = True,
    ):
        """
        Initialize a DemDecomp object for decomposing a detector error model.

        Parameters
        ----------
        org_dem : stim.DetectorErrorModel
            The original detector error model to decompose.
        color : one of {"r", "g", "b"}
            The color ('r', 'g', or 'b') for which to decompose the DEM.
        remove_non_edge_like_errors : bool, default True
            Whether to remove error mechanisms that are not edge-like when decomposing
            the original DEM into two stages.
        """
        self.color = color
        self.org_dem = org_dem

        dem1_sym, dem2_sym = self.decompose_org_dem(
            remove_non_edge_like_errors=remove_non_edge_like_errors
        )
        self.dems_symbolic = dem1_sym, dem2_sym

        _, _, org_prob = dem_to_parity_check(org_dem)
        dem1, _ = dem1_sym.to_dem(org_prob)
        dem2, inds_dem2 = dem2_sym.to_dem(org_prob, sort=True)
        self._dems = dem1, dem2
        self.org_prob = org_prob

        error_map_matrix1 = dem1_sym.error_map_matrix
        error_map_matrix2 = csr_matrix(dem2_sym.error_map_matrix[inds_dem2, :])
        self.error_map_matrices = (error_map_matrix1, error_map_matrix2)

        # error_map_matrix2 should have at most one element per row / column
        # Check rows
        rows_with_multiple = np.diff(error_map_matrix2.indptr) > 1
        if np.any(rows_with_multiple):
            raise ValueError(
                f"error_map_matrix2 has {np.sum(rows_with_multiple)} rows with multiple non-zero elements"
            )

        # Check columns
        cols_with_multiple = np.bincount(error_map_matrix2.indices) > 1
        if np.any(cols_with_multiple):
            raise ValueError(
                f"error_map_matrix2 has {np.sum(cols_with_multiple)} columns with multiple non-zero elements"
            )

        H1, _, prob1 = dem_to_parity_check(dem1)
        H2, obs_matrix_stage2, prob2 = dem_to_parity_check(dem2)
        self.Hs = (H1, H2)
        self.probs = (prob1, prob2)
        self.obs_matrix_stage2 = obs_matrix_stage2

        # self._best_org_error_map = self._precompute_best_org_error_map()

    def __repr__(self) -> str:
        """Return the string representation of this object."""

        return (
            f"<DemDecomp object with color='{self.color}', "
            f"Hs[0].shape={self.Hs[0].shape}, Hs[1].shape={self.Hs[1].shape}>"
        )

    def decompose_org_dem(
        self,
        remove_non_edge_like_errors: bool = True,
        _decompose_pauli: bool = True,
    ) -> Tuple[_DemSymbolic, _DemSymbolic]:
        """
        Decompose the original DEM into the restricted and monochromatic symbolic DEMs.

        Parameters
        ----------
        remove_non_edge_like_errors : bool, default True
            Whether to remove error mechanisms that are not edge-like.
        _decompose_pauli : bool, default True
            Internal parameter to control whether to decompose Pauli X and Z errors.

        Returns
        -------
        dem1, dem2: _DemSymbolic
            Restricted and monochromatic DEMs of the given color, respectively, in
            symbolic form.
        """
        # Original XZ-decomposed DEM
        org_dem = self.org_dem

        # Set of detector ids to be reduced
        det_ids_to_reduce = []
        detector_coords = org_dem.get_detector_coordinates()
        for det_id, coord in detector_coords.items():
            if coord[4] == {"r": 0, "g": 1, "b": 2}[self.color]:
                det_ids_to_reduce.append(det_id)
        det_ids_to_reduce = set(det_ids_to_reduce)

        num_org_error_sources = org_dem.num_errors
        num_detectors = org_dem.num_detectors
        org_dem_dets = org_dem[num_org_error_sources:]
        org_dem_errors = org_dem[:num_org_error_sources]

        # Decompose into X and Z errors (normally not needed, but for cult+growing).
        # (i.e., ignore correlations between X and Z errors)
        pauli_decomposed_targets_dict = {}
        pauli_decomposed_probs_dict = {}
        for i_em, em in enumerate(org_dem_errors):
            targets = em.targets_copy()
            target_labels = []
            detids = []
            for target in targets:
                if target.is_relative_detector_id():
                    detid = int(str(target)[1:])
                    target_labels.append(detid)
                    detids.append(detid)
                elif target.is_logical_observable_id():
                    target_labels.append(f"L{str(target)[1:]}")
                else:
                    raise ValueError(f"Unknown target: {target}")
            det_paulis = [detector_coords[detid][3] for detid in detids]
            if 0 in det_paulis and 2 in det_paulis and _decompose_pauli:
                target_labels_X = []
                targets_X = []
                target_labels_Z = []
                targets_Z = []
                for i_label, (target, label) in enumerate(zip(targets, target_labels)):
                    if not isinstance(label, str) and det_paulis[i_label] == 0:
                        target_labels_X.append(label)
                        targets_X.append(target)
                    else:
                        target_labels_Z.append(label)
                        targets_Z.append(target)
                target_labels_X = frozenset(target_labels_X)
                pauli_decomposed_targets_dict[target_labels_X] = targets_X
                pauli_decomposed_probs_dict[target_labels_X] = [i_em]
                target_labels_Z = frozenset(target_labels_Z)
                pauli_decomposed_targets_dict[target_labels_Z] = targets_Z
                pauli_decomposed_probs_dict[target_labels_Z] = [i_em]
            else:
                target_labels = frozenset(target_labels)
                pauli_decomposed_targets_dict[target_labels] = targets
                pauli_decomposed_probs_dict[target_labels] = [i_em]

        # Obtain targets list for the two steps
        dem1_probs_dict = {}
        dem1_dets_dict = {}
        dem1_obss_dict = {}
        dem1_virtual_obs_dict = {}

        # dem2_targets_list = []
        dem2_probs = []
        dem2_dets = []
        dem2_obss = []

        for target_ids in pauli_decomposed_targets_dict:
            targets = pauli_decomposed_targets_dict[target_ids]
            prob = pauli_decomposed_probs_dict[target_ids]

            dem1_dets_sng = []
            dem1_obss_sng = []
            dem2_dets_sng = []
            dem2_obss_sng = []
            dem1_det_ids = set()

            for target in targets:
                if target.is_logical_observable_id():
                    dem2_obss_sng.append(target)
                else:
                    det_id = int(str(target)[1:])
                    if det_id in det_ids_to_reduce:
                        dem2_dets_sng.append(target)
                    else:
                        dem1_dets_sng.append(target)
                        dem1_det_ids.add(det_id)

            if remove_non_edge_like_errors:
                if dem1_dets_sng:
                    if len(dem1_dets_sng) >= 3 or len(dem2_dets_sng) >= 2:
                        continue
                else:
                    if len(dem2_dets_sng) >= 3:
                        continue

            if dem1_det_ids:
                dem1_det_ids = frozenset(dem1_det_ids)
                try:
                    dem1_probs_dict[dem1_det_ids].extend(prob.copy())
                    virtual_obs = dem1_virtual_obs_dict[dem1_det_ids]
                except KeyError:
                    virtual_obs = len(dem1_probs_dict)
                    # dem1_obss_sng.append(stim.target_logical_observable_id(virtual_obs))
                    dem1_probs_dict[dem1_det_ids] = prob.copy()
                    dem1_dets_dict[dem1_det_ids] = dem1_dets_sng
                    dem1_obss_dict[dem1_det_ids] = dem1_obss_sng
                    dem1_virtual_obs_dict[dem1_det_ids] = virtual_obs

                virtual_det_id = num_detectors + virtual_obs
                dem2_dets_sng.append(stim.target_relative_detector_id(virtual_det_id))

            # Add a virtual observable to dem2 for distinguishing error sources
            # L0: real observable. L1, L2, ...: virtual observables.
            dem2_dets.append(dem2_dets_sng)
            dem2_obss.append(dem2_obss_sng)
            dem2_probs.append(prob)

        # Convert dem1 information to lists
        dem1_probs = list(dem1_probs_dict.values())
        dem1_dets = [dem1_dets_dict[key] for key in dem1_probs_dict]
        dem1_obss = [dem1_obss_dict[key] for key in dem1_probs_dict]

        # Convert to _DemSymbolic objects
        dem1_sym = _DemSymbolic(
            dem1_probs, dem1_dets, dem1_obss, org_dem_dets, org_dem.num_errors
        )
        dem2_sym = _DemSymbolic(
            dem2_probs, dem2_dets, dem2_obss, org_dem_dets, org_dem.num_errors
        )

        return dem1_sym, dem2_sym

    def map_errors_to_org_dem(
        self, errors: List[bool | int] | np.ndarray | spmatrix, *, stage: int
    ) -> np.ndarray:
        """
        Map errors from the decomposed DEM back to the original DEM format.

        Parameters
        ----------
        errors : list, numpy array, or scipy sparse matrix of bool/int
            Errors in the decomposed DEM for the specified stage. If it has more than
            one dimension, the last dimension is assumed to index the errors.
        stage : int
            Which stage's errors to map (1 or 2).

        Returns
        -------
        errors_org : numpy array of bool
            Error array mapped to the original DEM format.
        """
        if isinstance(errors, spmatrix):
            errors = errors.astype("uint8")
        else:
            errors = np.asarray(errors, dtype=np.uint8)

        error_map_matrix = self.error_map_matrices[stage - 1]
        errors_org = errors @ error_map_matrix
        return errors_org

    def __iter__(self) -> Iterator[stim.DetectorErrorModel]:
        """
        Return an iterator over the decomposed detector error models.

        Returns
        -------
        Iterator of stim.DetectorErrorModel
            Yields the restricted and monochromatic detector error models.
        """

        return iter(self._dems)

    def __getitem__(self, idx: int) -> stim.DetectorErrorModel:
        """
        Access one of the decomposed detector error models by index.

        Parameters
        ----------
        idx : int
            ``0`` for the restricted model or ``1`` for the monochromatic model.

        Returns
        -------
        stim.DetectorErrorModel
            The selected detector error model.
        """

        return self._dems[idx]
