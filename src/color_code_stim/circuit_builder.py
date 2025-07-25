"""
Circuit builder module for color code quantum error correction.

This module provides the CircuitBuilder class which handles the generation
of quantum circuits for different color code topologies and configurations.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import stim
import igraph as ig

from .config import COLOR_LABEL, CIRCUIT_TYPE, color_to_color_val, get_qubit_coords
from .cultivation import _reformat_cultivation_circuit
from .noise_model import NoiseModel


class CircuitBuilder:
    """
    Builder class for constructing color code quantum circuits.

    This class extracts circuit generation logic from the monolithic ColorCode class
    to provide modular, testable circuit construction for different topologies.
    """

    def __init__(
        self,
        d: int,
        d2: Optional[int],
        rounds: int,
        circuit_type: CIRCUIT_TYPE,
        cnot_schedule: List[int],
        temp_bdry_type: str,
        noise_model: Union[NoiseModel, Dict[str, float]],
        perfect_init_final: bool,
        perfect_logical_initialization: bool,
        perfect_logical_measurement: bool,
        perfect_first_syndrome_extraction: bool,
        tanner_graph: ig.Graph,
        qubit_groups: Dict[str, ig.VertexSeq],
        exclude_non_essential_pauli_detectors: bool = False,
        cultivation_circuit: Optional[stim.Circuit] = None,
        comparative_decoding: bool = False,
    ):
        """
        Initialize the circuit builder.

        Parameters
        ----------
        d : int
            Code distance.
        d2 : Optional[int]
            Second code distance (required for some circuit types).
        rounds : int
            Number of syndrome extraction rounds.
        circuit_type : CIRCUIT_TYPE
            Type of circuit to build.
        cnot_schedule : List[int]
            CNOT gate schedule.
        temp_bdry_type : str
            Temporal boundary type.
        noise_model : NoiseModel or Dict[str, float]
            Noise model specifying error rates for different operations.
        perfect_init_final : bool
            Whether to use perfect initialization and final measurement (backward compatibility).
        perfect_logical_initialization : bool
            Whether logical initialization operations (data qubit reset) are noiseless.
        perfect_logical_measurement : bool
            Whether logical final measurement operations are noiseless.
        perfect_first_syndrome_extraction : bool
            Whether the first syndrome extraction round is noiseless.
        tanner_graph : ig.Graph
            The Tanner graph representing the color code.
        qubit_groups : Dict[str, ig.VertexSeq]
            Grouped qubits by type (data, anc, anc_Z, anc_X).
        exclude_non_essential_pauli_detectors : bool, default False
            Whether to exclude non-essential Pauli detectors.
        cultivation_circuit : Optional[stim.Circuit], default None
            Cultivation circuit for cult+growing.
        comparative_decoding : bool, default False
            Whether to use comparative decoding.
        """
        self.d = d
        self.d2 = d2
        self.rounds = rounds
        self.circuit_type = circuit_type
        self.cnot_schedule = cnot_schedule
        self.temp_bdry_type = temp_bdry_type
        self.noise_model = noise_model
        self.perfect_init_final = perfect_init_final
        self.perfect_logical_initialization = perfect_logical_initialization
        self.perfect_logical_measurement = perfect_logical_measurement
        self.perfect_first_syndrome_extraction = perfect_first_syndrome_extraction
        self.tanner_graph = tanner_graph
        self.qubit_groups = qubit_groups
        self.exclude_non_essential_pauli_detectors = (
            exclude_non_essential_pauli_detectors
        )
        self.cultivation_circuit = cultivation_circuit
        self.comparative_decoding = comparative_decoding

        # Extract physical error rates
        self.p_bitflip = noise_model["bitflip"]
        self.p_depol = noise_model["depol"]
        self.p_reset = noise_model["reset"]
        self.p_meas = noise_model["meas"]
        self.p_cnot = noise_model["cnot"]
        self.p_idle = noise_model["idle"]
        self.p_initial_data_qubit_depol = noise_model["initial_data_qubit_depol"]
        self.p_depol1_after_cnot = noise_model["depol1_after_cnot"]
        self.p_idle_during_cnot = noise_model["idle_during_cnot"]
        self.p_idle_during_meas = noise_model["idle_during_meas"]

        # Extract qubit groups
        self.data_qubits = qubit_groups["data"]
        self.anc_qubits = qubit_groups["anc"]
        self.anc_Z_qubits = qubit_groups["anc_Z"]
        self.anc_X_qubits = qubit_groups["anc_X"]

        # Extract qubit IDs
        self.data_qids = self.data_qubits["qid"]
        self.anc_qids = self.anc_qubits["qid"]
        self.anc_Z_qids = self.anc_Z_qubits["qid"]
        self.anc_X_qids = self.anc_X_qubits["qid"]

        # Calculate counts
        self.num_data_qubits = len(self.data_qids)
        self.num_anc_Z_qubits = len(self.anc_Z_qubits)
        self.num_anc_X_qubits = len(self.anc_X_qubits)
        self.num_anc_qubits = self.num_anc_X_qubits + self.num_anc_Z_qubits

        self.num_qubits = tanner_graph.vcount()
        self.all_qids = list(range(self.num_qubits))
        self.all_qids_set = set(self.all_qids)

    def build(self) -> stim.Circuit:
        """
        Build the complete quantum circuit.

        Returns
        -------
        stim.Circuit
            The constructed quantum circuit.
        """
        # Identify red linkes (only for 'rec_stability', 'growing', and 'cult+growing')
        red_links, data_q1s, data_q2s = self._identify_red_links()

        # Initialize main circuit with qubit coordinates
        circuit = self._initialize_circuit_with_coordinates()

        # Add cultivation circuit if needed
        interface_detectors_info = self._add_cultivation_circuit(circuit)

        # Build syndrome extraction circuits
        synd_extr_circuits, obs_included_lookbacks = (
            self._build_syndrome_extraction_circuits(interface_detectors_info)
        )

        # Add data qubit initialization
        self._add_data_qubit_initialization(circuit, red_links, data_q1s, data_q2s)

        # Add initial data qubit depolarizing noise if perfect_first_syndrome_extraction=False
        if not self.perfect_first_syndrome_extraction:
            self._add_initial_data_qubit_depol(circuit)

        # Add ancilla qubit initialization
        self._add_ancilla_initialization(circuit)

        # Add main syndrome extraction rounds
        circuit += synd_extr_circuits[0]

        # Add initial data qubit depolarizing noise if perfect_first_syndrome_extraction=True
        if self.perfect_first_syndrome_extraction:
            self._add_initial_data_qubit_depol(circuit)

        circuit += synd_extr_circuits[1] * (self.rounds - 1)

        # Add final measurements and detectors
        self._add_final_measurements_and_detectors(
            circuit, red_links, data_q1s, data_q2s
        )

        # Add logical observables
        self._add_logical_observables(circuit, obs_included_lookbacks)

        return circuit

    def _add_initial_data_qubit_depol(self, circuit: stim.Circuit) -> None:
        """
        Add initial data qubit depolarizing noise.

        Timing depends on perfect_first_syndrome_extraction:
        - If True: Applied after first syndrome extraction round
        - If False: Applied after data qubit initialization
        """
        if self.p_initial_data_qubit_depol > 0:
            circuit.append(
                "DEPOLARIZE1", self.data_qids, self.p_initial_data_qubit_depol
            )

    def _identify_red_links(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Identify red links for the circuit."""
        if self.circuit_type == "rec_stability":
            red_links = [
                [link.source, link.target]
                for link in self.tanner_graph.es.select(color="r")
            ]
            red_links = np.array(red_links)
            data_q1s = red_links[:, 0]
            data_q2s = red_links[:, 1]
        elif self.circuit_type in {"tri", "rec"}:
            red_links = data_q1s = data_q2s = None
        elif self.circuit_type in {"growing", "cult+growing"}:
            x_offset_init_patch = 6 * round((self.d2 - self.d) / 2)
            y_offset_init_patch = 3 * round((self.d2 - self.d) / 2)
            self.x_offset_init_patch = x_offset_init_patch
            self.y_offset_init_patch = y_offset_init_patch

            data_qubits_outside_init_patch = self.data_qubits.select(
                y_lt=y_offset_init_patch
            )
            red_links = [
                [link.source, link.target]
                for link in self.tanner_graph.es.select(
                    color="r",
                    _within=data_qubits_outside_init_patch,
                )
            ]
            red_links = np.array(red_links)
            data_q1s = red_links[:, 0]
            data_q2s = red_links[:, 1]
        else:
            raise NotImplementedError(
                f"Circuit type {self.circuit_type} not implemented"
            )

        return red_links, data_q1s, data_q2s

    def _initialize_circuit_with_coordinates(self) -> stim.Circuit:
        """Initialize circuit and add qubit coordinates."""
        circuit = stim.Circuit()
        for qubit in self.tanner_graph.vs:
            coords = get_qubit_coords(qubit)
            circuit.append("QUBIT_COORDS", qubit.index, coords)
        return circuit

    def _add_cultivation_circuit(self, circuit: stim.Circuit) -> Optional[Dict]:
        """Add cultivation circuit for cult+growing type."""
        if self.circuit_type == "cult+growing":
            qubit_coords = {}
            for qubit in self.tanner_graph.vs:
                coords = get_qubit_coords(qubit)
                qubit_coords[qubit.index] = coords

            cult_circuit, interface_detectors_info = _reformat_cultivation_circuit(
                self.cultivation_circuit,
                self.d,
                qubit_coords,
                x_offset=self.x_offset_init_patch,
                y_offset=self.y_offset_init_patch,
            )
            circuit += cult_circuit
            return interface_detectors_info
        return None

    def _build_syndrome_extraction_circuits(
        self,
        interface_detectors_info: Optional[Dict],
    ) -> Tuple[List[stim.Circuit], Set]:
        """Build syndrome extraction circuits with and without detectors."""
        # Build circuits with measurements and detectors
        synd_extr_circuits = []
        obs_included_lookbacks = set()

        for first in [True, False]:
            # Check if this is the first round and perfect_first_syndrome_extraction is enabled
            skip_noise = first and self.perfect_first_syndrome_extraction

            # Build base syndrome extraction circuit (with or without noise)
            synd_extr_circuit = self._build_base_syndrome_extraction(
                perfect_round=skip_noise
            )

            # Add bit-flip errors (skip if perfect first round)
            if self.p_bitflip > 0 and not skip_noise:
                synd_extr_circuit.insert(
                    0,
                    stim.CircuitInstruction(
                        "X_ERROR", self.data_qids, [self.p_bitflip]
                    ),
                )

            # Add depolarizing errors (skip if perfect first round)
            if self.p_depol > 0 and not skip_noise:
                synd_extr_circuit.insert(
                    0,
                    stim.CircuitInstruction(
                        "DEPOLARIZE1", self.data_qids, [self.p_depol]
                    ),
                )

            # Add measurements (use p_meas=0 for perfect first round)
            p_meas_round = 0 if skip_noise else self.p_meas
            synd_extr_circuit.append("MRZ", self.anc_Z_qids, p_meas_round)
            synd_extr_circuit.append("MRX", self.anc_X_qids, p_meas_round)

            # Apply idle noise to data qubits during measurement operations
            if not skip_noise:
                idle_rate_meas = self._get_idle_rate_for_context("meas")
                if idle_rate_meas > 0:
                    synd_extr_circuit.append(
                        "DEPOLARIZE1", self.data_qids, idle_rate_meas
                    )

            # Add detectors
            obs_included_lookbacks = self._add_detectors(
                synd_extr_circuit,
                first,
                interface_detectors_info,
                obs_included_lookbacks,
            )

            # Add reset errors and idle errors (these are for the next round, so always include)
            if self.p_reset > 0:
                synd_extr_circuit.append("X_ERROR", self.anc_Z_qids, self.p_reset)
                synd_extr_circuit.append("Z_ERROR", self.anc_X_qids, self.p_reset)

            synd_extr_circuit.append("TICK")
            synd_extr_circuit.append("SHIFT_COORDS", (), (0, 0, 1))

            synd_extr_circuits.append(synd_extr_circuit)

        return synd_extr_circuits, obs_included_lookbacks

    def _build_base_syndrome_extraction(
        self, perfect_round: bool = False
    ) -> stim.Circuit:
        """Build the base syndrome extraction circuit without SPAM operations.

        Parameters
        ----------
        perfect_round : bool, default False
            If True, skip CNOT and idle errors for a perfect syndrome extraction round.
        """
        synd_extr_circuit = stim.Circuit()

        for timeslice in range(1, max(self.cnot_schedule) + 1):
            targets = [
                i for i, val in enumerate(self.cnot_schedule) if val == timeslice
            ]
            operated_qids = set()

            CX_targets = []
            for target in targets:
                # Define offset based on target
                if target in {0, 6}:
                    offset = (-2, 1)
                elif target in {1, 7}:
                    offset = (2, 1)
                elif target in {2, 8}:
                    offset = (4, 0)
                elif target in {3, 9}:
                    offset = (2, -1)
                elif target in {4, 10}:
                    offset = (-2, -1)
                else:
                    offset = (-4, 0)

                target_anc_qubits = (
                    self.anc_Z_qubits if target < 6 else self.anc_X_qubits
                )
                for anc_qubit in target_anc_qubits:
                    data_qubit_x = anc_qubit["face_x"] + offset[0]
                    data_qubit_y = anc_qubit["face_y"] + offset[1]
                    data_qubit_name = f"{data_qubit_x}-{data_qubit_y}"
                    try:
                        data_qubit = self.tanner_graph.vs.find(name=data_qubit_name)
                    except ValueError:
                        continue
                    anc_qid = anc_qubit.index
                    data_qid = data_qubit.index
                    operated_qids.update({anc_qid, data_qid})

                    CX_target = (
                        [data_qid, anc_qid] if target < 6 else [anc_qid, data_qid]
                    )
                    CX_targets.extend(CX_target)

            synd_extr_circuit.append("CX", CX_targets)
            if self.p_cnot > 0 and not perfect_round:
                synd_extr_circuit.append("DEPOLARIZE2", CX_targets, self.p_cnot)

            # Apply single-qubit depolarizing noise to each qubit involved in CNOT gates
            self._apply_depol1_after_cnot(synd_extr_circuit, CX_targets, perfect_round)

            idle_rate = self._get_idle_rate_for_context("cnot")
            if idle_rate > 0 and not perfect_round:
                idling_qids = list(self.all_qids_set - operated_qids)
                synd_extr_circuit.append("DEPOLARIZE1", idling_qids, idle_rate)

            synd_extr_circuit.append("TICK")

        return synd_extr_circuit

    def _add_detectors(
        self,
        circuit: stim.Circuit,
        first: bool,
        interface_detectors_info: Optional[Dict],
        obs_included_lookbacks: Set,
    ) -> Set:
        """Add Z-type, X-type, and Y-type detectors to the circuit."""
        # Z- and X-type detectors
        for pauli in ["Z", "X"]:
            if self.exclude_non_essential_pauli_detectors:
                if self.temp_bdry_type in {"X", "Z"} and pauli != self.temp_bdry_type:
                    continue

            anc_qubits_now = self.anc_Z_qubits if pauli == "Z" else self.anc_X_qubits
            init_lookback = (
                -self.num_anc_qubits if pauli == "Z" else -self.num_anc_X_qubits
            )

            for j, anc_qubit in enumerate(anc_qubits_now):
                pauli_val = 0 if pauli == "X" else 2
                color = anc_qubit["color"]
                color_val = color_to_color_val(color)
                coords = get_qubit_coords(anc_qubit)
                det_coords = coords + (0, pauli_val, color_val)

                if not first:
                    lookback = init_lookback + j
                    targets = [
                        stim.target_rec(lookback),
                        stim.target_rec(lookback - self.num_anc_qubits),
                    ]
                    circuit.append("DETECTOR", targets, det_coords)
                else:
                    detector_exists = self._check_detector_exists(coords, color, pauli)

                    if detector_exists:
                        targets = [stim.target_rec(init_lookback + j)]

                        # Special handling for cult+growing
                        if (
                            self.circuit_type == "cult+growing"
                            and coords[1] >= self.y_offset_init_patch
                        ):
                            obs_included_lookbacks = self._handle_cultivation_detectors(
                                circuit,
                                targets,
                                det_coords,
                                coords,
                                color,
                                pauli,
                                anc_qubit,
                                interface_detectors_info,
                                obs_included_lookbacks,
                            )
                        else:
                            circuit.append("DETECTOR", targets, det_coords)

        # Y-type detectors
        self._add_y_type_detectors(circuit, first)

        return obs_included_lookbacks

    def _check_detector_exists(
        self, coords: Tuple[int, int], color: str, pauli: str
    ) -> bool:
        """Check if a detector should exist based on circuit type and position."""
        if self.circuit_type in {"tri", "rec"}:
            return self.temp_bdry_type == pauli
        elif self.circuit_type == "rec_stability":
            return color != "r"
        elif self.circuit_type == "growing":
            if coords[1] >= self.y_offset_init_patch:
                return self.temp_bdry_type == pauli
            else:
                return color != "r"
        elif self.circuit_type == "cult+growing":
            return coords[1] >= self.y_offset_init_patch or color != "r"
        else:
            raise NotImplementedError

    def _handle_cultivation_detectors(
        self,
        circuit: stim.Circuit,
        targets: List,
        det_coords: Tuple,
        coords: Tuple[int, int],
        color: str,
        pauli: str,
        anc_qubit: ig.Vertex,
        interface_detectors_info: Dict,
        obs_included_lookbacks: Set,
    ) -> Set:
        """Handle special detector logic for cultivation + growing circuits."""
        det_coords += (-1,)
        adj_data_qubits = frozenset(
            qubit.index
            for qubit in anc_qubit.neighbors()
            if qubit["y"] >= self.y_offset_init_patch - 1e-6
        )
        paulis = [pauli]
        if pauli == "X":
            paulis.append("Z")
            det_coords = list(det_coords)
            det_coords[3] = 1
            det_coords = tuple(det_coords)

            anc_Z_name = f"{coords[0]}-{coords[1]}-Z"
            anc_Z_qid = self.tanner_graph.vs.find(name=anc_Z_name).index
            j_Z = self.anc_Z_qids.index(anc_Z_qid)
            targets.append(stim.target_rec(-self.num_anc_qubits + j_Z))

        targets_cult_all = []
        lookbacks = []
        for pauli_now in paulis:
            key = (pauli_now, adj_data_qubits)
            targets_cult = interface_detectors_info[key]
            lookbacks.extend(targets_cult)
            targets_cult = [
                stim.target_rec(-self.num_anc_qubits + cult_lookback)
                for cult_lookback in targets_cult
            ]
            targets_cult_all.extend(targets_cult)
        targets.extend(targets_cult_all)

        if pauli == "X" and color == "g":
            obs_included_lookbacks ^= set(lookbacks)

        circuit.append("DETECTOR", targets, det_coords)
        return obs_included_lookbacks

    def _add_y_type_detectors(self, circuit: stim.Circuit, first: bool) -> None:
        """Add Y-type detectors for Y temporal boundary."""
        if first and self.temp_bdry_type == "Y" and self.circuit_type != "cult+growing":
            for j_Z, anc_qubit_Z in enumerate(self.anc_Z_qubits):
                color = anc_qubit_Z["color"]
                coords = get_qubit_coords(anc_qubit_Z)

                detector_exists = self._check_y_detector_exists(coords, color)

                if detector_exists:
                    j_X = self.anc_X_qubits["name"].index(
                        f"{anc_qubit_Z['x']}-{anc_qubit_Z['y']}-X"
                    )
                    det_coords = coords + (0, 1, color_to_color_val(color))
                    targets = [
                        stim.target_rec(-self.num_anc_qubits + j_Z),
                        stim.target_rec(-self.num_anc_X_qubits + j_X),
                    ]
                    circuit.append("DETECTOR", targets, det_coords)

    def _check_y_detector_exists(self, coords: Tuple[int, int], color: str) -> bool:
        """Check if Y-type detector should exist."""
        if self.circuit_type in {"tri", "rec"}:
            return True
        elif self.circuit_type == "rec_stability":
            return color != "r"
        elif self.circuit_type == "growing":
            return coords[1] >= self.y_offset_init_patch
        else:
            raise NotImplementedError

    def _add_data_qubit_initialization(
        self,
        circuit: stim.Circuit,
        red_links: Optional[np.ndarray],
        data_q1s: Optional[np.ndarray],
        data_q2s: Optional[np.ndarray],
    ) -> None:
        """Add data qubit initialization based on circuit type."""
        if self.circuit_type in {"tri", "rec"}:
            circuit.append(f"R{self.temp_bdry_type}", self.data_qids)
            if self.p_reset > 0 and not self.perfect_logical_initialization:
                error_type = "Z_ERROR" if self.temp_bdry_type == "X" else "X_ERROR"
                circuit.append(error_type, self.data_qids, self.p_reset)

        elif self.circuit_type == "rec_stability":
            circuit.append("RX", data_q1s)
            circuit.append("RZ", data_q2s)
            if self.p_reset > 0 and not self.perfect_logical_initialization:
                circuit.append("Z_ERROR", data_q1s, self.p_reset)
                circuit.append("X_ERROR", data_q2s, self.p_reset)
            circuit.append("TICK")
            circuit.append("CX", red_links.ravel())
            if self.p_cnot > 0:
                circuit.append("DEPOLARIZE2", red_links.ravel(), self.p_cnot)
            # Apply single-qubit depolarizing noise to each qubit involved in CNOT gates
            self._apply_depol1_after_cnot(circuit, red_links.ravel())

        elif self.circuit_type == "growing":
            # Data qubits inside the initial patch
            data_qids_init_patch = self.data_qubits.select(
                y_ge=self.y_offset_init_patch
            )["qid"]
            circuit.append(f"R{self.temp_bdry_type}", data_qids_init_patch)
            if self.p_reset > 0 and not self.perfect_logical_initialization:
                error_type = "Z_ERROR" if self.temp_bdry_type == "X" else "X_ERROR"
                circuit.append(error_type, data_qids_init_patch, self.p_reset)

            # Data qubits outside the initial patch
            circuit.append("RX", data_q1s)
            circuit.append("RZ", data_q2s)
            if self.p_reset > 0:
                circuit.append("Z_ERROR", data_q1s, self.p_reset)
                circuit.append("X_ERROR", data_q2s, self.p_reset)
            circuit.append("TICK")
            circuit.append("CX", red_links.ravel())
            if self.p_cnot > 0:
                circuit.append("DEPOLARIZE2", red_links.ravel(), self.p_cnot)
            # Apply single-qubit depolarizing noise to each qubit involved in CNOT gates
            self._apply_depol1_after_cnot(circuit, red_links.ravel())

        elif self.circuit_type == "cult+growing":
            # Find last tick position
            for i in range(len(circuit) - 1, -1, -1):
                instruction = circuit[i]
                if (
                    isinstance(instruction, stim.CircuitInstruction)
                    and instruction.name == "TICK"
                ):
                    last_tick_pos = i
                    break

            # Data qubits outside the initial patch (inserted before the last tick)
            circuit.insert(last_tick_pos, stim.CircuitInstruction("RX", data_q1s))
            circuit.insert(last_tick_pos + 1, stim.CircuitInstruction("RZ", data_q2s))
            if self.p_reset > 0:
                circuit.insert(
                    last_tick_pos + 2,
                    stim.CircuitInstruction("Z_ERROR", data_q1s, [self.p_reset]),
                )
                circuit.insert(
                    last_tick_pos + 3,
                    stim.CircuitInstruction("X_ERROR", data_q2s, [self.p_reset]),
                )

            # CX gate (inserted after the last tick)
            circuit.append("CX", red_links.ravel())
            if self.p_cnot > 0:
                circuit.append("DEPOLARIZE2", red_links.ravel(), self.p_cnot)
            # Apply single-qubit depolarizing noise to each qubit involved in CNOT gates
            self._apply_depol1_after_cnot(circuit, red_links.ravel())

        else:
            raise NotImplementedError

    def _add_ancilla_initialization(self, circuit: stim.Circuit) -> None:
        """Add ancilla qubit initialization."""
        circuit.append("RZ", self.anc_Z_qids)
        circuit.append("RX", self.anc_X_qids)

        if self.p_reset > 0 and not self.perfect_logical_initialization:
            circuit.append("X_ERROR", self.anc_Z_qids, self.p_reset)
            circuit.append("Z_ERROR", self.anc_X_qids, self.p_reset)

        circuit.append("TICK")

    def _add_final_measurements_and_detectors(
        self,
        circuit: stim.Circuit,
        red_links: Optional[np.ndarray],
        data_q1s: Optional[np.ndarray],
        data_q2s: Optional[np.ndarray],
    ) -> None:
        """Add final data qubit measurements and last detectors."""
        use_last_detectors = True
        p_meas_final = 0 if self.perfect_logical_measurement else self.p_meas

        if self.circuit_type in {"tri", "rec", "growing", "cult+growing"}:
            circuit.append(f"M{self.temp_bdry_type}", self.data_qids, p_meas_final)
            if use_last_detectors:
                self._add_last_detectors(circuit)

        elif self.circuit_type == "rec_stability":
            if not use_last_detectors:
                raise NotImplementedError
            self._add_stability_final_measurements(
                circuit, red_links, data_q1s, data_q2s, p_meas_final
            )

        else:
            raise NotImplementedError

    def _add_last_detectors(self, circuit: stim.Circuit) -> None:
        """Add last detectors for tri/rec/growing/cult+growing circuits."""
        if self.temp_bdry_type == "X":
            anc_qubits_now = self.anc_X_qubits
            init_lookback = -self.num_data_qubits - self.num_anc_X_qubits
            pauli_val = 0
        else:
            anc_qubits_now = self.anc_Z_qubits
            init_lookback = -self.num_data_qubits - self.num_anc_qubits
            pauli_val = 2 if self.temp_bdry_type == "Z" else 1

        for j_anc, anc_qubit in enumerate(anc_qubits_now):
            ngh_data_qubits = anc_qubit.neighbors()
            lookback_inds = [
                -self.num_data_qubits + self.data_qids.index(q.index)
                for q in ngh_data_qubits
            ]
            lookback_inds.append(init_lookback + j_anc)
            if self.temp_bdry_type == "Y":
                anc_X_qubit = self.tanner_graph.vs.find(
                    name=f"{anc_qubit['x']}-{anc_qubit['y']}-X"
                )
                j_anc_X = self.anc_X_qids.index(anc_X_qubit.index)
                lookback_inds.append(
                    -self.num_data_qubits - self.num_anc_X_qubits + j_anc_X
                )

            target = [stim.target_rec(ind) for ind in lookback_inds]
            color_val = color_to_color_val(anc_qubit["color"])
            coords = get_qubit_coords(anc_qubit) + (0, pauli_val, color_val)
            circuit.append("DETECTOR", target, coords)

    def _add_stability_final_measurements(
        self,
        circuit: stim.Circuit,
        red_links: np.ndarray,
        data_q1s: np.ndarray,
        data_q2s: np.ndarray,
        p_meas_final: float,
    ) -> None:
        """Add final measurements for rec_stability circuits."""
        circuit.append("CX", red_links.ravel())
        if self.p_cnot > 0 and not self.perfect_logical_measurement:
            circuit.append("DEPOLARIZE2", red_links.ravel(), self.p_cnot)
        # Apply single-qubit depolarizing noise to each qubit involved in CNOT gates
        # Use perfect_logical_measurement flag to determine if noise should be skipped
        self._apply_depol1_after_cnot(
            circuit, red_links.ravel(), self.perfect_logical_measurement
        )

        circuit.append("TICK")

        # ZZ measurement outcomes
        circuit.append("MZ", data_q2s, p_meas_final)

        # Apply idle noise to ancilla qubits during data qubit measurements
        if not self.perfect_logical_measurement:
            idle_rate_meas = self._get_idle_rate_for_context("meas")
            if idle_rate_meas > 0:
                circuit.append("DEPOLARIZE1", self.anc_qids, idle_rate_meas)

        num_data_q2s = data_q2s.size
        lookback_inds_anc = {}
        for j, data_q2 in enumerate(data_q2s):
            for anc_Z_qubit in self.tanner_graph.vs[data_q2].neighbors():
                if anc_Z_qubit["pauli"] == "Z" and anc_Z_qubit["color"] != "r":
                    anc_Z_qid = anc_Z_qubit.index
                    lookback_ind = j - num_data_q2s
                    try:
                        lookback_inds_anc[anc_Z_qid].append(lookback_ind)
                    except KeyError:
                        lookback_inds_anc[anc_Z_qid] = [lookback_ind]

        obs_Z_lookback_inds = []
        for j_anc_Z, anc_Z_qubit in enumerate(self.anc_Z_qubits):
            check_meas_lookback_ind = j_anc_Z - num_data_q2s - self.num_anc_qubits
            if anc_Z_qubit["color"] != "g":
                obs_Z_lookback_inds.append(check_meas_lookback_ind)
            try:
                lookback_inds = lookback_inds_anc[anc_Z_qubit.index]
            except KeyError:
                continue
            lookback_inds.append(check_meas_lookback_ind)
            target = [stim.target_rec(ind) for ind in lookback_inds]
            color_val = color_to_color_val(anc_Z_qubit["color"])
            coords = get_qubit_coords(anc_Z_qubit) + (0, 2, color_val)
            circuit.append("DETECTOR", target, coords)

        target = [stim.target_rec(ind) for ind in obs_Z_lookback_inds]
        circuit.append("OBSERVABLE_INCLUDE", target, 0)
        if self.comparative_decoding:
            raise NotImplementedError

        # XX measurement outcomes
        circuit.append("MX", data_q1s, p_meas_final)

        # Apply idle noise to ancilla qubits during data qubit measurements
        if not self.perfect_logical_measurement:
            idle_rate_meas = self._get_idle_rate_for_context("meas")
            if idle_rate_meas > 0:
                circuit.append("DEPOLARIZE1", self.anc_qids, idle_rate_meas)

        num_data_q1s = data_q1s.size
        lookback_inds_anc = {}
        for j, data_q1 in enumerate(data_q1s):
            for anc_X_qubit in self.tanner_graph.vs[data_q1].neighbors():
                if anc_X_qubit["pauli"] == "X" and anc_X_qubit["color"] != "r":
                    anc_X_qid = anc_X_qubit.index
                    lookback_ind = j - num_data_q1s
                    try:
                        lookback_inds_anc[anc_X_qid].append(lookback_ind)
                    except KeyError:
                        lookback_inds_anc[anc_X_qid] = [lookback_ind]

        obs_X_lookback_inds = []
        for j_anc_X, anc_X_qubit in enumerate(self.anc_X_qubits):
            check_meas_lookback_ind = (
                j_anc_X - num_data_q1s - num_data_q2s - self.num_anc_X_qubits
            )
            color = anc_X_qubit["color"]
            if color != "g":
                obs_X_lookback_inds.append(check_meas_lookback_ind)

            try:
                lookback_inds = lookback_inds_anc[anc_X_qubit.index]
            except KeyError:
                continue

            lookback_inds.append(check_meas_lookback_ind)
            target = [stim.target_rec(ind) for ind in lookback_inds]
            color_val = color_to_color_val(color)
            coords = get_qubit_coords(anc_X_qubit) + (0, 0, color_val)
            circuit.append("DETECTOR", target, coords)

        target = [stim.target_rec(ind) for ind in obs_X_lookback_inds]
        circuit.append("OBSERVABLE_INCLUDE", target, 1)
        if self.comparative_decoding:
            raise NotImplementedError

    def _apply_depol1_after_cnot(
        self, circuit: stim.Circuit, CX_targets: list, perfect_round: bool = False
    ) -> None:
        """
        Apply single-qubit depolarizing noise to each qubit involved in CNOT gates.

        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to add noise to.
        CX_targets : list
            List of CNOT targets in the format [control, target, control2, target2, ...].
        perfect_round : bool, default False
            If True, skip noise application for perfect syndrome extraction rounds.
        """
        if self.p_depol1_after_cnot == 0 or perfect_round:
            return

        # Extract unique qubits from CX_targets
        # CX_targets format: [control1, target1, control2, target2, ...]
        unique_qubits = list(set(CX_targets))

        if unique_qubits:
            circuit.append("DEPOLARIZE1", unique_qubits, self.p_depol1_after_cnot)

    def _get_idle_rate_for_context(self, context: str) -> float:
        """
        Get the appropriate idle noise rate for a given context.

        Parameters
        ----------
        context : str
            Context for which to get idle rate. Must be one of:
            - "cnot": During CNOT operations
            - "meas": During measurement operations
            - "general": General idle periods (mixed CNOT and measurement)

        Returns
        -------
        float
            Appropriate idle noise rate based on context and parameter overrides.
            For general context, uses maximum between idle_during_cnot and idle_during_meas
            if both are specified, otherwise falls back to individual overrides or base idle.
        """
        if context == "cnot" and self.p_idle_during_cnot is not None:
            return self.p_idle_during_cnot
        elif context == "meas" and self.p_idle_during_meas is not None:
            return self.p_idle_during_meas
        elif context == "general":
            # For general idle periods (mixed operations), use maximum of context-specific rates
            rates_to_consider = []

            if self.p_idle_during_cnot is not None:
                rates_to_consider.append(self.p_idle_during_cnot)
            if self.p_idle_during_meas is not None:
                rates_to_consider.append(self.p_idle_during_meas)

            if rates_to_consider:
                # Use maximum of specified context-specific rates
                return max(rates_to_consider)
            else:
                # Fall back to base idle rate if no context-specific rates specified
                return self.p_idle
        else:
            return self.p_idle

    def _add_logical_observables(
        self, circuit: stim.Circuit, obs_included_lookbacks: Set
    ) -> None:
        """Add logical observables based on circuit type."""
        if self.circuit_type not in {"tri", "rec", "growing", "cult+growing"}:
            return

        if self.circuit_type in {"tri", "growing", "cult+growing"}:
            qubits_logs = [self.tanner_graph.vs.select(obs=True)]
            if self.circuit_type == "tri":
                bdry_colors = [0]
            elif self.circuit_type == "growing":
                bdry_colors = [1]
            elif self.circuit_type == "cult+growing":
                bdry_colors = [1]
        elif self.circuit_type == "rec":
            qubits_log_r = self.tanner_graph.vs.select(obs_r=True)
            qubits_log_g = self.tanner_graph.vs.select(obs_g=True)
            qubits_logs = [qubits_log_r, qubits_log_g]
            bdry_colors = [1, 0]

        for obs_id, qubits_log in enumerate(qubits_logs):
            lookback_inds = [
                -self.num_data_qubits + self.data_qids.index(q.index)
                for q in qubits_log
            ]
            if obs_included_lookbacks:
                num_meas_after_cult = (
                    self.num_anc_qubits * self.rounds + self.num_data_qubits
                )
                lookback_inds.extend(
                    lb - num_meas_after_cult for lb in obs_included_lookbacks
                )

            target = [stim.target_rec(ind) for ind in lookback_inds]
            circuit.append("OBSERVABLE_INCLUDE", target, obs_id)
            if self.comparative_decoding:
                color_val = bdry_colors[obs_id]
                if self.temp_bdry_type == "X":
                    pauli_val = 0
                elif self.temp_bdry_type == "Y":
                    pauli_val = 1
                elif self.temp_bdry_type == "Z":
                    pauli_val = 2
                else:
                    raise ValueError(f"Invalid temp_bdry_type: {self.temp_bdry_type}")
                coords = (-1, -1, -1, pauli_val, color_val, obs_id)
                circuit.append("DETECTOR", target, coords)
