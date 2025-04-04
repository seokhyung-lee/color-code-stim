from typing import List, Sequence, Set

import numpy as np
import stim
from scipy.sparse import csr_matrix


class _ErrorMechanismSymbolic:
    prob_vars: np.ndarray
    prob_muls: np.ndarray
    dets: Set[stim.DemTarget]
    obss: Set[stim.DemTarget]

    def __init__(
        self,
        prob_vars: Sequence[int],
        dets: Sequence[stim.DemTarget],
        obss: Sequence[stim.DemTarget],
        prob_muls: Sequence[float] | int | float = 1,
    ):
        prob_vars = np.asarray(prob_vars, dtype="int32")
        prob_muls = np.asarray(prob_muls)
        if prob_muls.ndim == 0:
            prob_muls = np.full_like(prob_vars, prob_muls)

        self.prob_vars = prob_vars
        self.prob_muls = prob_muls
        self.dets = set(dets)
        self.obss = set(obss)


class _DemSymbolic:
    ems: List[_ErrorMechanismSymbolic]
    dets_org: stim.DetectorErrorModel
    error_mapping_arr: csr_matrix

    def __init__(
        self,
        prob_vars: Sequence[Sequence[int]],
        dets: Sequence[Sequence[stim.DemTarget]],
        obss: Sequence[Sequence[stim.DemTarget]],
        dets_org: stim.DetectorErrorModel,
        num_org_errors: int,
    ):
        self.ems = [
            _ErrorMechanismSymbolic(*prms) for prms in zip(prob_vars, dets, obss)
        ]
        self.dets_org = dets_org

        # Prepare data for efficient CSR matrix creation
        data = []
        rows = []
        cols = []

        # Collect the coordinates of all True values
        for i, pv in enumerate(prob_vars):
            for col in pv:
                rows.append(i)
                cols.append(col)
                data.append(True)

        # Create the CSR matrix directly from the coordinate data
        self.error_mapping_arr = csr_matrix(
            (data, (rows, cols)), shape=(len(self.ems), num_org_errors), dtype=bool
        )

    def to_dem(
        self, prob_vals: Sequence[float], sort: bool = False
    ) -> stim.DetectorErrorModel:
        prob_vals = np.asarray(prob_vals, dtype="float64")

        probs = [
            (1 - np.prod(1 - 2 * em.prob_muls * prob_vals[em.prob_vars])) / 2
            for em in self.ems
        ]

        if sort:
            inds = np.argsort(probs)[::-1]
        else:
            inds = range(len(probs))

        dem = stim.DetectorErrorModel()
        for i in inds:
            em = self.ems[i]
            targets = em.dets | em.obss
            dem.append("error", probs[i], list(targets))

        dem += self.dets_org

        return dem

    def non_edge_like_errors_exist(self) -> bool:
        for e in self.ems:
            if len(e.dets) > 2:
                return True
        return False

    def decompose_complex_error_mechanisms(self):
        """
        For each error mechanism `e` in `dem` that involves more than two detectors,
        searches for candidate pairs (e1, e2) among the other error mechanisms (with e1, e2 disjoint)
        such that:
        - e1.dets ∪ e2.dets equals e.dets, and e1.dets ∩ e2.dets is empty.
        - e1.obss ∪ e2.obss equals e.obss, and e1.obss ∩ e2.obss is empty.

        For each valid candidate pair, updates both e1 and e2 by concatenating e's
        probability variable and multiplier arrays. If there are multiple candidate pairs,
        the probability multipliers from e are split equally among the pairs.

        Finally, removes the complex error mechanism `e` from `dem.ems`.

        Raises:
            ValueError: If a complex error mechanism cannot be decomposed.
        """
        # Iterate over a copy of the error mechanisms list
        em_inds_to_remove = []
        for i_e, e in enumerate(self.ems):
            # Process only error mechanisms that involve more than 2 detectors.
            if len(e.dets) > 2:
                candidate_pairs = []
                # Search for candidate pairs among the other error mechanisms.
                for i, e1 in enumerate(self.ems):
                    if i == i_e or i in em_inds_to_remove:
                        continue
                    for j in range(i + 1, len(self.ems)):
                        if j == i_e or j in em_inds_to_remove:
                            continue
                        e2 = self.ems[j]
                        # Check that e1 and e2 have disjoint detectors and observables.
                        if e1.dets & e2.dets:
                            continue
                        if e1.obss & e2.obss:
                            continue
                        # Check that the union of their detectors and observables equals e's.
                        if (e1.dets | e2.dets == e.dets) and (
                            e1.obss | e2.obss == e.obss
                        ):
                            candidate_pairs.append((e1, e2))
                if not candidate_pairs:
                    raise ValueError(
                        f"No valid decomposition found for error mechanism with dets {e.dets} and obss {e.obss}."
                    )
                # If there are multiple decompositions, split the probability equally.
                fraction = 1 / len(candidate_pairs)
                for e1, e2 in candidate_pairs:
                    # Append the probability variable arrays.
                    e1.prob_vars = np.concatenate([e1.prob_vars, e.prob_vars])
                    e2.prob_vars = np.concatenate([e2.prob_vars, e.prob_vars])
                    # Append the probability multiplier arrays, scaling by the fraction.
                    e1.prob_muls = np.concatenate(
                        [e1.prob_muls, e.prob_muls * fraction]
                    )
                    e2.prob_muls = np.concatenate(
                        [e2.prob_muls, e.prob_muls * fraction]
                    )
                # Remove the complex error mechanism from the model.
                em_inds_to_remove.append(i_e)

            for i_e in em_inds_to_remove[::-1]:
                self.ems.pop(i_e)
