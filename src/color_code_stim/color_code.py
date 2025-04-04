import itertools
import math
import stat
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim
from ldpc import BpDecoder
from matplotlib.ticker import AutoLocator
from scipy.sparse import csc_matrix
from statsmodels.stats.proportion import proportion_confint

from .cultivation import _load_cultivation_circuit, _reformat_cultivation_circuit
from .stim_symbolic import _DemSymbolic
from .stim_utils import (
    dem_to_parity_check,
    get_observable_matrix_from_dem,
    remove_obs_from_dem,
)
from .utils import (
    get_pfail,
    get_project_folder,
    timeit,
)

PAULI_LABEL = Literal["X", "Y", "Z"]
COLOR_LABEL = Literal["r", "g", "b"]


class ColorCode:
    tanner_graph: ig.Graph
    circuit: stim.Circuit
    d: int
    rounds: int
    temp_bdry_type: Literal["X", "Y", "Z", "r", "g", "b"]
    qubit_groups: Dict[str, ig.VertexSeq]
    obs_paulis: List[PAULI_LABEL]
    org_dem: stim.DetectorErrorModel
    H: csc_matrix
    org_probs: np.ndarray
    obs_matrix: csc_matrix
    _bp_inputs: Dict[str, Any]
    detectors: List[Tuple[ig.Vertex, int]]
    detector_ids: Dict[COLOR_LABEL, List[int]]
    cult_detector_ids: List[int]
    interface_detector_ids: List[int]
    shape: str
    num_obs: int
    d2: Optional[int]
    cnot_schedule: Union[str, Tuple[int, ...]]
    perfect_init_final: bool
    probs: Dict[str, float]
    comparative_decoding: bool
    use_last_detectors: bool
    cultivation_circuit: Optional[stim.Circuit]
    benchmarking: bool
    dems_sym_decomposed: Dict[COLOR_LABEL, Tuple[_DemSymbolic, _DemSymbolic]]
    dems_decomposed: Dict[
        COLOR_LABEL, Tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]
    ]

    def __init__(
        self,
        *,
        d: int,
        rounds: int,
        shape: str = "tri",
        d2: int = None,
        cnot_schedule: Union[str, Sequence[int]] = "tri_optimal",
        temp_bdry_type: Optional[Literal["X", "Y", "Z", "x", "y", "z"]] = None,
        p_bitflip: float = 0.0,
        p_reset: float = 0.0,
        p_meas: float = 0.0,
        p_cnot: float = 0.0,
        p_idle: float = 0.0,
        p_circuit: Optional[float] = None,
        cultivation_circuit: Optional[stim.Circuit] = None,
        perfect_init_final: bool = False,
        comparative_decoding: bool = False,
        use_last_detectors: bool = True,
        generate_dem: bool = True,
        decompose_dem: bool = True,
        benchmarking: bool = False,
    ):
        """
        Class for constructing a color code circuit and simulating the
        concatenated MWPM decoder.

        Parameters
        ----------
        d : int >= 3
            Code distance. Should be an odd number of 3 or more.
        rounds : int >= 1
            Number of syndrome extraction rounds.
        shape : {'triangle', 'tri', 'rectangle', 'rec', 'rec_stability', 'growing',
                'cult+growing'}, default 'tri'
            Circuit type.
            - 'triangle'/'tri': memory experiment of a triangular patch with distance
              `d`.
            - 'rectangle'/'rec': memory experiment of a rectangular patch with distance
              `d` and `d2`.
            - 'rec_stability': stability experiment of a rectangle-like patch with
              single-type boundaries. `d` and `d2` indicate the size of the patch,
              although they are no longer code distances.
            - 'growing': growing operation from a triangular patch with distance `d` to
              a larger triangular patch with distance `d2`. Must be `d2 > d`.
            - 'cult+growing': cultivation on a triangular patch with distance `d`,
              followed by a growing operation to distance `d2`. Must be `d2 > d`.
        d2 : int >= 3, optional
            Second code distance required for several circuit types.
            If not provided, `d2 = d`.
        cnot_schedule : {12-tuple of integers, 'tri_optimal', 'tri_optimal_reversed'},
                        default 'tri_optimal'
            CNOT schedule.
            If this is a 12-tuple of integers, it indicates (a, b, ... l) specifying
            the CNOT schedule.
            If this is 'tri_optimal', it is (2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2), which
            is the optimal schedule for the triangular color code.
            If this is 'tri_optimal_reversed', it is (3, 4, 7, 6, 5, 2, 2, 3, 6, 5, 4, 1),
            which has the X- and Z-part reversed from 'tri_optimal'.
        temp_bdry_type : {'X', 'Y', 'Z', 'x', 'y', 'z'}, optional
            Type of the temporal boundaries, i.e., the reset/measurement basis of
            data qubits at the beginning and end of the circuit.
            Not supported for `rec_stability` and `cult+growing` circuits: the types of
            the temporal boundaries are fixed to red for `rec_stability` and `Y` for
            `cult+growing`. For the other circuit types, it is `Z` by default.
        p_bitflip : float, default 0
            Bit-flip probability at the start of each round.
        p_reset : float, default 0
            Probability of a wrong qubit reset (i.e., producing an
            orthogonal state).
        p_meas : float, default 0
            Probability of a flipped measurement outcome.
        p_cnot : float, default 0
            Strength of a two-qubit depolarizing noise channel following
            each CNOT gate.
        p_idle : float, default 0
            Strength of a single-qubit depolarizing noise channel following
            each idle gate.
        p_circuit : float, optional
            If given, p_reset = p_meas = p_cnot = p_idle = p_circuit.
        cultivation_circuit: stim.Circuit, optional
            If given, it is used as the cultivation circuit for cultivation + growing circuit (`shape == 'cult+growing'`). WARNING: Its validity is not checked internally.
        perfect_init_final : bool, default False
            Whether to use perfect initialization and final measurement.
        comparative_decoding : bool, default False
            Whether to use the comparative decoding technique. If True, observables are included as additional detectors and decoding can be done by running the decoder for each logical class and choosing the lowest-weight one. This also provides the logical gap information, which quantifies the reliability of decoding.
        use_last_detectors : bool, default True
            Whether to use detectors from the last round.
        generate_dem : bool, default True
            Whether to generate the detector error model in advance.
        decompose_dem: bool, default True
            Whether to decompose the detector error model in advance.
        benchmarking : bool, default False
            Whether to measure execution time of each step.
        """
        if isinstance(cnot_schedule, str):
            if cnot_schedule in ["tri_optimal", "LLB"]:
                cnot_schedule = (2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2)
            elif cnot_schedule == "tri_optimal_reversed":
                cnot_schedule = (3, 4, 7, 6, 5, 2, 2, 3, 6, 5, 4, 1)
            else:
                raise ValueError
        else:
            assert len(cnot_schedule) == 12

        assert d > 1 and rounds >= 1

        if p_circuit is not None:
            p_reset = p_meas = p_cnot = p_idle = p_circuit

        self.d = d
        d2 = self.d2 = d if d2 is None else d2
        self.rounds = rounds

        if shape in {"triangle", "tri"}:
            assert d % 2 == 1
            self.shape = "tri"
            self.num_obs = 1

        elif shape in {"rectangle", "rec"}:
            assert d2 is not None
            assert d % 2 == 0 and d2 % 2 == 0
            self.shape = "rec"
            self.num_obs = 2

        elif shape == "rec_stability":
            assert d2 is not None
            assert d % 2 == 0 and d2 % 2 == 0
            self.shape = "rec_stability"
            self.num_obs = 2

        elif shape == "growing":
            assert d2 is not None
            assert d % 2 == 1 and d2 % 2 == 1 and d2 > d
            self.shape = "growing"
            self.num_obs = 1

        elif shape in {"cultivation+growing", "cult+growing"}:
            assert p_circuit is not None and p_bitflip == 0
            assert d2 is not None
            assert d % 2 == 1 and d2 % 2 == 1 and d2 > d
            self.shape = "cult+growing"
            self.num_obs = 1

        else:
            raise ValueError("Invalid shape")

        if temp_bdry_type is None:
            if shape == "rec_stability":
                temp_bdry_type = "r"
            elif shape == "cult+growing":
                temp_bdry_type = "Y"
            else:
                temp_bdry_type = "Z"
        else:
            assert temp_bdry_type in {"X", "Y", "Z", "x", "y", "z"}
            assert shape not in {"rec_stability", "cult+growing"}
            temp_bdry_type = temp_bdry_type.upper()

        self.temp_bdry_type = temp_bdry_type

        if shape == "rec_stability":
            self.obs_paulis = ["Z", "X"]
        else:
            self.obs_paulis = [temp_bdry_type] * self.num_obs

        self.cnot_schedule = cnot_schedule
        self.perfect_init_final = perfect_init_final
        self.probs = {
            "bitflip": p_bitflip,
            "reset": p_reset,
            "meas": p_meas,
            "cnot": p_cnot,
            "idle": p_idle,
        }
        self.comparative_decoding = comparative_decoding

        if self.comparative_decoding and self.shape == "rec_stability":
            raise NotImplementedError

        self.use_last_detectors = use_last_detectors

        if self.shape == "cult+growing":
            if cultivation_circuit is None:
                project_folder = get_project_folder()
                try:
                    path = (
                        project_folder
                        / "assets"
                        / "cultivation_circuits"
                        / f"d{d}_p{p_circuit}.stim"
                    )
                except FileNotFoundError:
                    raise NotImplementedError(
                        f"Not supported for d = {d}, p = {p_circuit}"
                    )
                cultivation_circuit = stim.Circuit.from_file(path)
        else:
            cultivation_circuit = None
        self.cultivation_circuit = cultivation_circuit

        self.tanner_graph = ig.Graph()

        self.benchmarking = benchmarking

        # Mapping between detector ids and ancillary qubits
        # self.detectors[detector_id] = (anc_qubit, time_coord)
        self.detectors = []

        # Detector ids grouped by colors
        self.detector_ids = {"r": [], "g": [], "b": []}

        # Various qubit groups
        self.qubit_groups = {}

        # Decomposed detector error models
        # It is generated when required.
        # if dem_decomposed is None:
        #     dem_decomposed = {}
        self.dems_sym_decomposed = {}
        self.dems_decomposed = {}

        tanner_graph = self.tanner_graph

        self._create_tanner_graph()

        self.circuit = self._generate_circuit()

        if generate_dem:
            self.org_dem = self.circuit.detector_error_model(flatten_loops=True)

            self.H, self.org_probs = dem_to_parity_check(self.org_dem)
            self.obs_matrix = get_observable_matrix_from_dem(self.org_dem)
        else:
            self.org_dem = None
            self.H = None
            self.obs_matrix = None
            self.org_probs = None

        self._bp_inputs = {}

        # Detector coordinates: (x, y, t, pauli, color) OR (x, y, t, pauli, color, flag)
        # pauli = 0, 1, 2 -> X, Y, Z
        # color = 0, 1, 2 -> r, g, b
        # flag does not exist for ordinary detectors.
        # flag >= 0: detector corresponding to an observable (id=flag)
        # (only when comparatitive_decoding is True)
        # flag = -2, -1: cultivation / interface between cultivation and growing
        # (only for `cult+growing` circuits)
        detector_coords_dict = self.circuit.get_detector_coordinates()
        self.cult_detector_ids = []
        self.interface_detector_ids = []

        for detector_id in range(self.circuit.num_detectors):
            coords = detector_coords_dict[detector_id]
            if self.shape == "cult+growing" and len(coords) == 6:
                # The detector is in the cultivation circuit or the interface region
                flag = coords[-1]
                if flag == -1:
                    self.interface_detector_ids.append(detector_id)
                elif flag == -2:
                    self.cult_detector_ids.append(detector_id)
                    continue

            x = round(coords[0])
            y = round(coords[1])
            t = round(coords[2])
            pauli = round(coords[3])
            color = self.color_val_to_color(round(coords[4]))
            is_obs = len(coords) == 6 and round(coords[-1]) >= 0

            if not is_obs:
                # Ordinary X/Z detectors
                if pauli == 0:
                    name = f"{x-1}-{y}-X"
                elif pauli in [1, 2]:
                    name = f"{x+1}-{y}-Z"
                else:
                    print(coords)
                    raise ValueError(f"Invalid pauli: {pauli}")

                qubit = tanner_graph.vs.find(name=name)
                self.detectors.append((qubit, t))
                color = qubit["color"]

            self.detector_ids[color].append(detector_id)

        if generate_dem and decompose_dem:
            for c in ["r", "g", "b"]:
                self.decompose_detector_error_model(c)

    def get_detector_type(self, detector_id: int) -> Tuple[PAULI_LABEL, COLOR_LABEL]:
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
        color = self.color_val_to_color(coords[4])

        return pauli, color

    @timeit
    def _create_tanner_graph(self):
        shape = self.shape
        tanner_graph = self.tanner_graph

        if shape in {"tri", "growing", "cult+growing"}:
            if shape == "tri":
                d = self.d
            else:
                d = self.d2

            assert d % 2 == 1

            detid = 0
            L = round(3 * (d - 1) / 2)
            for y in range(L + 1):
                if y % 3 == 0:
                    anc_qubit_color = "g"
                    anc_qubit_pos = 2
                elif y % 3 == 1:
                    anc_qubit_color = "b"
                    anc_qubit_pos = 0
                else:
                    anc_qubit_color = "r"
                    anc_qubit_pos = 1

                for x in range(2 * y, 4 * L - 2 * y + 1, 4):
                    boundary = []
                    if y == 0:
                        boundary.append("r")
                    if x == 2 * y:
                        boundary.append("g")
                    if x == 4 * L - 2 * y:
                        boundary.append("b")
                    boundary = "".join(boundary)
                    if not boundary:
                        boundary = None

                    if shape == "tri":
                        obs = boundary in ["r", "rg", "rb"]
                    elif shape in {"growing", "cult+growing"}:
                        obs = boundary in ["g", "gb", "rg"]
                    else:
                        obs = False

                    if round((x / 2 - y) / 2) % 3 != anc_qubit_pos:
                        tanner_graph.add_vertex(
                            name=f"{x}-{y}",
                            x=x,
                            y=y,
                            qid=tanner_graph.vcount(),
                            pauli=None,
                            color=None,
                            obs=obs,
                            boundary=boundary,
                        )
                    else:
                        for pauli in ["Z", "X"]:
                            tanner_graph.add_vertex(
                                name=f"{x}-{y}-{pauli}",
                                x=x,
                                y=y,
                                qid=tanner_graph.vcount(),
                                pauli=pauli,
                                color=anc_qubit_color,
                                obs=False,
                                boundary=boundary,
                            )
                            detid += 1

        elif shape == "rec":
            d, d2 = self.d, self.d2
            assert d % 2 == 0
            assert d2 % 2 == 0

            detid = 0
            L1 = round(3 * d / 2 - 2)
            L2 = round(3 * d2 / 2 - 2)
            for y in range(L2 + 1):
                if y % 3 == 0:
                    anc_qubit_color = "g"
                    anc_qubit_pos = 2
                elif y % 3 == 1:
                    anc_qubit_color = "b"
                    anc_qubit_pos = 0
                else:
                    anc_qubit_color = "r"
                    anc_qubit_pos = 1

                for x in range(2 * y, 2 * y + 4 * L1 + 1, 4):
                    boundary = []
                    if y == 0 or y == L2:
                        boundary.append("r")
                    if 2 * y == x or 2 * y == x - 4 * L1:
                        boundary.append("g")
                    boundary = "".join(boundary)
                    if not boundary:
                        boundary = None

                    if round((x / 2 - y) / 2) % 3 != anc_qubit_pos:
                        obs_g = y == 0
                        obs_r = x == 2 * y + 4 * L1

                        tanner_graph.add_vertex(
                            name=f"{x}-{y}",
                            x=x,
                            y=y,
                            qid=tanner_graph.vcount(),
                            pauli=None,
                            color=None,
                            obs_r=obs_r,
                            obs_g=obs_g,
                            boundary=boundary,
                        )
                    else:
                        for pauli in ["Z", "X"]:
                            tanner_graph.add_vertex(
                                name=f"{x}-{y}-{pauli}",
                                x=x,
                                y=y,
                                qid=tanner_graph.vcount(),
                                pauli=pauli,
                                color=anc_qubit_color,
                                obs_r=False,
                                obs_g=False,
                                boundary=boundary,
                            )
                            detid += 1

            # Additional corner vertex
            x = 2 * L2 + 2
            y = L2 + 1
            # x, y = L2 - 2, L2
            tanner_graph.add_vertex(
                name=f"{x}-{y}",
                x=x,
                y=y,
                qid=tanner_graph.vcount(),
                pauli=None,
                color=None,
                obs_r=False,
                obs_g=False,
                boundary="rg",
            )

        elif shape == "rec_stability":
            d = self.d
            d2 = self.d2
            assert d % 2 == 0
            assert d2 % 2 == 0

            detid = 0
            L1 = round(3 * d / 2 - 2)
            L2 = round(3 * d2 / 2 - 2)
            for y in range(L2 + 1):
                if y % 3 == 0:
                    anc_qubit_color = "r"
                    anc_qubit_pos = 0
                elif y % 3 == 1:
                    anc_qubit_color = "b"
                    anc_qubit_pos = 1
                else:
                    anc_qubit_color = "g"
                    anc_qubit_pos = 2

                if y == 0:
                    x_init_adj = 8
                elif y == 1:
                    x_init_adj = 4
                else:
                    x_init_adj = 0

                if y == L2:
                    x_fin_adj = 8
                elif y == L2 - 1:
                    x_fin_adj = 4
                else:
                    x_fin_adj = 0

                for x in range(2 * y + x_init_adj, 2 * y + 4 * L1 + 1 - x_fin_adj, 4):
                    if (
                        y == 0
                        or y == L2
                        or x == y * 2
                        or x == 2 * y + 4 * L1
                        or (x, y) == (6, 1)
                        or (x, y) == (2 * L2 + 4 * L1 - 6, L2 - 1)
                    ):
                        boundary = "g"
                    else:
                        boundary = None

                    if round((x / 2 - y) / 2) % 3 != anc_qubit_pos:
                        tanner_graph.add_vertex(
                            name=f"{x}-{y}",
                            x=x,
                            y=y,
                            qid=tanner_graph.vcount(),
                            pauli=None,
                            color=None,
                            boundary=boundary,
                        )
                    else:
                        for pauli in ["Z", "X"]:
                            tanner_graph.add_vertex(
                                name=f"{x}-{y}-{pauli}",
                                x=x,
                                y=y,
                                qid=tanner_graph.vcount(),
                                pauli=pauli,
                                color=anc_qubit_color,
                                boundary=boundary,
                            )
                            detid += 1

        else:
            raise ValueError("Invalid shape")

        # Update qubit_groups
        data_qubits = tanner_graph.vs.select(pauli=None)
        anc_qubits = tanner_graph.vs.select(pauli_ne=None)
        anc_Z_qubits = anc_qubits.select(pauli="Z")
        anc_X_qubits = anc_qubits.select(pauli="X")
        anc_red_qubits = anc_qubits.select(color="r")
        anc_green_qubits = anc_qubits.select(color="g")
        anc_blue_qubits = anc_qubits.select(color="b")

        self.qubit_groups.update(
            {
                "data": data_qubits,
                "anc": anc_qubits,
                "anc_Z": anc_Z_qubits,
                "anc_X": anc_X_qubits,
                "anc_red": anc_red_qubits,
                "anc_green": anc_green_qubits,
                "anc_blue": anc_blue_qubits,
            }
        )

        # Add edges
        links = []
        offsets = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)]
        for anc_qubit in self.qubit_groups["anc"]:
            data_qubits = []
            for offset in offsets:
                data_qubit_x = anc_qubit["x"] + offset[0]
                data_qubit_y = anc_qubit["y"] + offset[1]
                data_qubit_name = f"{data_qubit_x}-{data_qubit_y}"
                try:
                    data_qubit = tanner_graph.vs.find(name=data_qubit_name)
                except ValueError:
                    continue
                data_qubits.append(data_qubit)
                tanner_graph.add_edge(anc_qubit, data_qubit, kind="tanner", color=None)

            if anc_qubit["pauli"] == "Z":
                weight = len(data_qubits)
                for i in range(weight):
                    qubit = data_qubits[i]
                    next_qubit = data_qubits[(i + 1) % weight]
                    if not tanner_graph.are_connected(qubit, next_qubit):
                        link = tanner_graph.add_edge(
                            qubit, next_qubit, kind="lattice", color=None
                        )
                        links.append(link)

        # Assign colors to links
        link: ig.Edge
        for link in links:
            v1: ig.Vertex
            v2: ig.Vertex
            v1, v2 = link.target_vertex, link.source_vertex
            ngh_ancs_1 = {anc.index for anc in v1.neighbors() if anc["pauli"] == "Z"}
            ngh_ancs_2 = {anc.index for anc in v2.neighbors() if anc["pauli"] == "Z"}
            color = tanner_graph.vs[(ngh_ancs_1 ^ ngh_ancs_2).pop()]["color"]
            link["color"] = color

    @staticmethod
    def get_qubit_coords(qubit: ig.Vertex) -> Tuple[int, int]:
        coords = [qubit["x"], qubit["y"]]
        if qubit["pauli"] == "Z":
            coords[0] -= 1
        elif qubit["pauli"] == "X":
            coords[0] += 1

        return tuple(coords)

    @staticmethod
    def color_to_color_val(color: Literal["r", "g", "b"]) -> int:
        if color == "r":
            return 0
        elif color == "g":
            return 1
        elif color == "b":
            return 2
        else:
            raise ValueError(f"Invalid color: {color}")

    @staticmethod
    def color_val_to_color(color_val: int) -> Literal["r", "g", "b"]:
        if color_val == 0:
            return "r"
        elif color_val == 1:
            return "g"
        elif color_val == 2:
            return "b"
        else:
            raise ValueError(f"Invalid color value: {color_val}")

    @timeit
    def _generate_circuit(self) -> stim.Circuit:
        qubit_groups = self.qubit_groups
        cnot_schedule = self.cnot_schedule
        tanner_graph = self.tanner_graph
        rounds = self.rounds
        shape = self.shape
        d = self.d
        d2 = self.d2
        temp_bdry_type = self.temp_bdry_type
        probs = self.probs
        p_bitflip = probs["bitflip"]
        p_reset = probs["reset"]
        p_meas = probs["meas"]
        p_cnot = probs["cnot"]
        p_idle = probs["idle"]

        perfect_init_final = self.perfect_init_final
        use_last_detectors = self.use_last_detectors

        data_qubits = qubit_groups["data"]
        anc_qubits = qubit_groups["anc"]
        anc_Z_qubits = qubit_groups["anc_Z"]
        anc_X_qubits = qubit_groups["anc_X"]

        data_qids = data_qubits["qid"]
        anc_qids = anc_qubits["qid"]
        anc_Z_qids = anc_Z_qubits["qid"]
        anc_X_qids = anc_X_qubits["qid"]

        num_data_qubits = len(data_qids)
        num_anc_Z_qubits = len(anc_Z_qubits)
        num_anc_X_qubits = len(anc_X_qubits)
        num_anc_qubits = num_anc_X_qubits + num_anc_Z_qubits

        num_qubits = tanner_graph.vcount()
        all_qids = list(range(num_qubits))
        all_qids_set = set(all_qids)

        if shape == "rec_stability":
            red_links = [
                [link.source, link.target] for link in tanner_graph.es.select(color="r")
            ]
            red_links = np.array(red_links)
            data_q1s = red_links[:, 0]
            data_q2s = red_links[:, 1]
        elif shape in {"tri", "rec"}:
            red_links = data_q1s = data_q2s = None
        elif shape in {"growing", "cult+growing"}:
            x_offset_init_patch = 6 * round((d2 - d) / 2)
            y_offset_init_patch = 3 * round((d2 - d) / 2)
            data_qubits_outside_init_patch = data_qubits.select(
                y_lt=y_offset_init_patch
            )
            red_links = [
                [link.source, link.target]
                for link in tanner_graph.es.select(
                    color="r",
                    _within=data_qubits_outside_init_patch,
                )
            ]
            red_links = np.array(red_links)
            data_q1s = red_links[:, 0]
            data_q2s = red_links[:, 1]
        else:
            raise NotImplementedError

        # Main circuit
        circuit = stim.Circuit()
        for qubit in tanner_graph.vs:
            coords = self.get_qubit_coords(qubit)
            circuit.append("QUBIT_COORDS", qubit.index, coords)

        if shape == "cult+growing":
            qubit_coords = {}
            for qubit in tanner_graph.vs:
                coords = self.get_qubit_coords(qubit)
                qubit_coords[qubit.index] = coords
            cult_circuit = _load_cultivation_circuit(self.d, self.probs["idle"])
            cult_circuit, interface_detectors_info = _reformat_cultivation_circuit(
                cult_circuit,
                self.d,
                qubit_coords,
                x_offset=x_offset_init_patch,
                y_offset=y_offset_init_patch,
            )
            circuit += cult_circuit

        # Syndrome extraction circuit without SPAM
        synd_extr_circuit_without_spam = stim.Circuit()
        for timeslice in range(1, max(cnot_schedule) + 1):
            targets = [i for i, val in enumerate(cnot_schedule) if val == timeslice]
            operated_qids = set()

            CX_targets = []
            for target in targets:
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

                target_anc_qubits = anc_Z_qubits if target < 6 else anc_X_qubits
                for anc_Z_qubit in target_anc_qubits:
                    data_qubit_x = anc_Z_qubit["x"] + offset[0]
                    data_qubit_y = anc_Z_qubit["y"] + offset[1]
                    data_qubit_name = f"{data_qubit_x}-{data_qubit_y}"
                    try:
                        data_qubit = tanner_graph.vs.find(name=data_qubit_name)
                    except ValueError:
                        continue
                    anc_qid = anc_Z_qubit.index
                    data_qid = data_qubit.index
                    operated_qids.update({anc_qid, data_qid})

                    CX_target = (
                        [data_qid, anc_qid] if target < 6 else [anc_qid, data_qid]
                    )
                    CX_targets.extend(CX_target)

            synd_extr_circuit_without_spam.append("CX", CX_targets)
            if p_cnot > 0:
                synd_extr_circuit_without_spam.append("DEPOLARIZE2", CX_targets, p_cnot)

            if p_idle > 0:
                idling_qids = list(all_qids_set - operated_qids)
                synd_extr_circuit_without_spam.append(
                    "DEPOLARIZE1", idling_qids, p_idle
                )

            synd_extr_circuit_without_spam.append("TICK")

        # Syndrome extraction circuit with measurement & detector
        def get_synd_extr_circuit(first=False):
            synd_extr_circuit = synd_extr_circuit_without_spam.copy()

            synd_extr_circuit.append("MRZ", anc_Z_qids, p_meas)
            synd_extr_circuit.append("MRX", anc_X_qids, p_meas)

            ## If first is True:
            # tri/rec: detectors have the same type as the temporal boundary
            # rec_stability: X- and Z-type detectors exist except for red faces
            # growing: same as tri/rec if anc_qubit['y'] >= y_offset_init_patch,
            #          same as rec_stability otherwise.
            # cult+growing: X- and Z-type detectors exist if anc_qubit['y'] >= y_offset_init_patch,
            #               same as rec_stability otherwise.

            ## Z- and X-type detectors
            for pauli in ["Z", "X"]:
                anc_qubits_now = anc_Z_qubits if pauli == "Z" else anc_X_qubits
                init_lookback = -num_anc_qubits if pauli == "Z" else -num_anc_X_qubits

                for j, anc_qubit in enumerate(anc_qubits_now):
                    anc_qubit: ig.Vertex
                    pauli_val = 0 if pauli == "X" else 2
                    color = anc_qubit["color"]
                    color_val = self.color_to_color_val(color)
                    coords = self.get_qubit_coords(anc_qubit)
                    det_coords = coords + (0, pauli_val, color_val)

                    if not first:
                        lookback = init_lookback + j
                        targets = [
                            stim.target_rec(lookback),
                            stim.target_rec(lookback - num_anc_qubits),
                        ]
                        synd_extr_circuit.append("DETECTOR", targets, det_coords)

                    else:
                        if shape in {"tri", "rec"}:
                            detector_exists = temp_bdry_type == pauli
                        elif shape == "rec_stability":
                            detector_exists = color != "r"
                        elif shape == "growing":
                            if coords[1] >= y_offset_init_patch:
                                detector_exists = temp_bdry_type == pauli
                            else:
                                detector_exists = color != "r"
                        elif shape == "cult+growing":
                            detector_exists = (
                                coords[1] >= y_offset_init_patch or color != "r"
                            )
                        else:
                            raise NotImplementedError

                        if detector_exists:
                            targets = [stim.target_rec(init_lookback + j)]

                            # For cult+growing, need to add targets during cultivation
                            # if anc_qubit is inside the initial patch.
                            if (
                                shape == "cult+growing"
                                and coords[1] >= y_offset_init_patch
                            ):
                                # Add detector targets during cultivation
                                det_coords += (-1,)
                                adj_data_qubits = frozenset(
                                    qubit.index
                                    for qubit in anc_qubit.neighbors()
                                    if qubit["y"] >= y_offset_init_patch
                                )
                                key = (pauli, adj_data_qubits)
                                targets_cult = interface_detectors_info[key]
                                targets_cult = [
                                    stim.target_rec(init_lookback + cult_lookback)
                                    for cult_lookback in targets_cult
                                ]
                                targets.extend(targets_cult)

                            # Add detector
                            synd_extr_circuit.append("DETECTOR", targets, det_coords)

            ## Y-type detectors
            if first and temp_bdry_type == "Y" and shape != "cult+growing":
                for j_Z, anc_qubit_Z in enumerate(anc_Z_qubits):
                    anc_qubit_Z: ig.Vertex
                    color = anc_qubit_Z["color"]
                    coords = self.get_qubit_coords(anc_qubit_Z)

                    if shape in {"tri", "rec"}:
                        detector_exists = True
                    elif shape == "rec_stability":
                        detector_exists = color != "r"
                    elif shape == "growing":
                        detector_exists = (
                            coords[1] >= y_offset_init_patch or color != "r"
                        )
                    else:
                        raise NotImplementedError

                    if detector_exists:
                        j_X = anc_X_qubits["name"].index(
                            f"{anc_qubit_Z['x']}-{anc_qubit_Z['y']}-X"
                        )
                        det_coords = coords + (0, 1, self.color_to_color_val(color))
                        targets = [
                            stim.target_rec(-num_anc_qubits + j_Z),
                            stim.target_rec(-num_anc_X_qubits + j_X),
                        ]
                        synd_extr_circuit.append("DETECTOR", targets, det_coords)

            if p_reset > 0:
                synd_extr_circuit.append("X_ERROR", anc_Z_qids, p_reset)
                synd_extr_circuit.append("Z_ERROR", anc_X_qids, p_reset)
            if p_idle > 0:
                synd_extr_circuit.append("DEPOLARIZE1", data_qids, p_idle)
            if p_bitflip > 0:
                synd_extr_circuit.append("X_ERROR", data_qids, p_bitflip)

            # if custom_noise_channel is not None:
            #     synd_extr_circuit.append(custom_noise_channel[0],
            #                              data_qids,
            #                              custom_noise_channel[1])

            synd_extr_circuit.append("TICK")
            synd_extr_circuit.append("SHIFT_COORDS", (), (0, 0, 1))

            return synd_extr_circuit

        # Initialize qubits
        if temp_bdry_type in {"X", "Y", "Z"} and shape not in {
            "growing",
            "cult+growing",
        }:
            circuit.append(f"R{temp_bdry_type}", data_qids)
            if p_reset > 0 and not perfect_init_final:
                error_type = "Z_ERROR" if temp_bdry_type == "X" else "X_ERROR"
                circuit.append(error_type, data_qids, p_reset)

        elif temp_bdry_type == "r":
            circuit.append("RX", data_q1s)
            circuit.append("RZ", data_q2s)
            if p_reset > 0 and not perfect_init_final:
                circuit.append("Z_ERROR", data_q1s, p_reset)
                circuit.append("X_ERROR", data_q2s, p_reset)

            circuit.append("TICK")

            circuit.append("CX", red_links.ravel())
            if p_cnot > 0:
                circuit.append("DEPOLARIZE2", red_links.ravel(), p_cnot)

        elif shape == "growing":
            # Data qubits inside the initial patch
            data_qids_init_patch = data_qubits.select(y_ge=y_offset_init_patch)["qid"]
            circuit.append(f"R{temp_bdry_type}", data_qids_init_patch)
            if p_reset > 0 and not perfect_init_final:
                error_type = "Z_ERROR" if temp_bdry_type == "X" else "X_ERROR"
                circuit.append(error_type, data_qids_init_patch, p_reset)

            # Data qubits outside the initial patch
            circuit.append("RX", data_q1s)
            circuit.append("RZ", data_q2s)
            if p_reset > 0:
                circuit.append("Z_ERROR", data_q1s, p_reset)
                circuit.append("X_ERROR", data_q2s, p_reset)

            circuit.append("TICK")

            circuit.append("CX", red_links.ravel())
            if p_cnot > 0:
                circuit.append("DEPOLARIZE2", red_links.ravel(), p_cnot)

        elif shape == "cult+growing":
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
            if p_reset > 0:
                circuit.insert(
                    last_tick_pos + 2,
                    stim.CircuitInstruction("Z_ERROR", data_q1s, [p_reset]),
                )
                circuit.insert(
                    last_tick_pos + 3,
                    stim.CircuitInstruction("X_ERROR", data_q2s, [p_reset]),
                )

            # CX gate (inserted after the last tick)
            circuit.append("CX", red_links.ravel())
            if p_cnot > 0:
                circuit.append("DEPOLARIZE2", red_links.ravel(), p_cnot)

        else:
            raise NotImplementedError

        circuit.append("RZ", anc_Z_qids)
        circuit.append("RX", anc_X_qids)

        if p_reset > 0:
            circuit.append("X_ERROR", anc_Z_qids, p_reset)
            circuit.append("Z_ERROR", anc_X_qids, p_reset)

        if p_bitflip > 0:
            circuit.append("X_ERROR", data_qids, p_bitflip)

        # if custom_noise_channel is not None:
        #     circuit.append(custom_noise_channel[0],
        #                    data_qids,
        #                    custom_noise_channel[1])

        circuit.append("TICK")

        circuit += get_synd_extr_circuit(first=True)
        circuit += get_synd_extr_circuit() * (rounds - 1)

        # Final data qubit measurements (& observables for red boundaries)
        p_meas_final = 0 if perfect_init_final else p_meas
        if temp_bdry_type in {"X", "Y", "Z"}:
            circuit.append(f"M{temp_bdry_type}", data_qids, p_meas_final)
            if use_last_detectors:
                if temp_bdry_type == "X":
                    anc_qubits_now = anc_X_qubits
                    init_lookback = -num_data_qubits - num_anc_X_qubits
                    pauli_val = 0
                else:
                    anc_qubits_now = anc_Z_qubits
                    init_lookback = -num_data_qubits - num_anc_qubits
                    pauli_val = 2 if temp_bdry_type == "Z" else 1

                for j_anc, anc_qubit in enumerate(anc_qubits_now):
                    anc_qubit: ig.Vertex
                    ngh_data_qubits = anc_qubit.neighbors()
                    lookback_inds = [
                        -num_data_qubits + data_qids.index(q.index)
                        for q in ngh_data_qubits
                    ]
                    lookback_inds.append(init_lookback + j_anc)
                    if temp_bdry_type == "Y":
                        anc_X_qubit = tanner_graph.vs.find(
                            name=f"{anc_qubit['x']}-{anc_qubit['y']}-X"
                        )
                        j_anc_X = anc_X_qids.index(anc_X_qubit.index)
                        lookback_inds.append(
                            -num_data_qubits - num_anc_X_qubits + j_anc_X
                        )
                    target = [stim.target_rec(ind) for ind in lookback_inds]
                    color_val = self.color_to_color_val(anc_qubit["color"])
                    coords = self.get_qubit_coords(anc_qubit) + (
                        0,
                        pauli_val,
                        color_val,
                    )
                    circuit.append("DETECTOR", target, coords)

        elif temp_bdry_type == "r":
            if not use_last_detectors:
                raise NotImplementedError

            circuit.append("CX", red_links.ravel())
            if p_cnot > 0 and not perfect_init_final:
                circuit.append("DEPOLARIZE2", red_links.ravel(), p_cnot)

            circuit.append("TICK")

            circuit.append("MZ", data_q2s, p_meas_final)  # ZZ measurement outcomes

            num_data_q2s = data_q2s.size
            lookback_inds_anc = {}
            for j, data_q2 in enumerate(data_q2s):
                for anc_Z_qubit in tanner_graph.vs[data_q2].neighbors():
                    if anc_Z_qubit["pauli"] == "Z" and anc_Z_qubit["color"] != "r":
                        anc_Z_qid = anc_Z_qubit.index
                        lookback_ind = j - num_data_q2s
                        try:
                            lookback_inds_anc[anc_Z_qid].append(lookback_ind)
                        except KeyError:
                            lookback_inds_anc[anc_Z_qid] = [lookback_ind]

            obs_Z_lookback_inds = []
            for j_anc_Z, anc_Z_qubit in enumerate(anc_Z_qubits):
                check_meas_lookback_ind = j_anc_Z - num_data_q2s - num_anc_qubits
                if anc_Z_qubit["color"] != "g":
                    obs_Z_lookback_inds.append(check_meas_lookback_ind)
                try:
                    lookback_inds = lookback_inds_anc[anc_Z_qubit.index]
                except KeyError:
                    continue
                lookback_inds.append(check_meas_lookback_ind)
                target = [stim.target_rec(ind) for ind in lookback_inds]
                color_val = self.color_to_color_val(anc_Z_qubit["color"])
                coords = self.get_qubit_coords(anc_Z_qubit) + (0, 2, color_val)
                circuit.append("DETECTOR", target, coords)

            target = [stim.target_rec(ind) for ind in obs_Z_lookback_inds]
            circuit.append("OBSERVABLE_INCLUDE", target, 0)
            if self.comparative_decoding:
                raise NotImplementedError

            circuit.append("MX", data_q1s, p_meas_final)  # XX measurement outcomes

            num_data_q1s = data_q1s.size
            lookback_inds_anc = {}
            for j, data_q1 in enumerate(data_q1s):
                for anc_X_qubit in tanner_graph.vs[data_q1].neighbors():
                    if anc_X_qubit["pauli"] == "X" and anc_X_qubit["color"] != "r":
                        anc_X_qid = anc_X_qubit.index
                        lookback_ind = j - num_data_q1s
                        try:
                            lookback_inds_anc[anc_X_qid].append(lookback_ind)
                        except KeyError:
                            lookback_inds_anc[anc_X_qid] = [lookback_ind]

            obs_X_lookback_inds = []
            for j_anc_X, anc_X_qubit in enumerate(anc_X_qubits):
                check_meas_lookback_ind = (
                    j_anc_X - num_data_q1s - num_data_q2s - num_anc_X_qubits
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
                color_val = self.color_to_color_val(color)
                coords = self.get_qubit_coords(anc_X_qubit) + (0, 0, color_val)
                circuit.append("DETECTOR", target, coords)

            target = [stim.target_rec(ind) for ind in obs_X_lookback_inds]
            circuit.append("OBSERVABLE_INCLUDE", target, 1)
            if self.comparative_decoding:
                raise NotImplementedError

        else:
            raise NotImplementedError

        # Logical observables
        if temp_bdry_type in {"X", "Y", "Z"}:
            if shape in {"tri", "growing"}:
                qubits_logs = [tanner_graph.vs.select(obs=True)]
                bdry_colors = [0] if self.shape == "tri" else [1]

            elif shape == "cult+growing":
                qubits_logs = [tanner_graph.vs.select(pauli=None)]
                bdry_colors = [1]

            elif shape == "rec":
                qubits_log_r = tanner_graph.vs.select(obs_r=True)
                qubits_log_g = tanner_graph.vs.select(obs_g=True)
                qubits_logs = [qubits_log_r, qubits_log_g]
                bdry_colors = [1, 0]

            for obs_id, qubits_log in enumerate(qubits_logs):
                lookback_inds = [
                    -num_data_qubits + data_qids.index(q.index) for q in qubits_log
                ]
                target = [stim.target_rec(ind) for ind in lookback_inds]
                circuit.append("OBSERVABLE_INCLUDE", target, obs_id)
                if self.comparative_decoding:
                    color_val = bdry_colors[obs_id]
                    if temp_bdry_type == "X":
                        pauli_val = 0
                    elif temp_bdry_type == "Y":
                        pauli_val = 1
                    elif temp_bdry_type == "Z":
                        pauli_val = 2
                    else:
                        raise ValueError(f"Invalid temp_bdry_type: {temp_bdry_type}")
                    coords = (-1, -1, -1, pauli_val, color_val, obs_id)
                    circuit.append("DETECTOR", target, coords)

        return circuit

    def draw_tanner_graph(
        self,
        ax: Optional[plt.Axes] = None,
        show_axes: bool = False,
        show_lattice: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """
        Draw the tanner graph of the code.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis on which to draw the graph. If None, a new figure and axis will be created.
        show_axes : bool, default False
            Whether to show the x- and y-axis.
        show_lattice : bool, default False
            Whether to show the lattice edges in addition to the tanner graph edges.
        **kwargs : dict
            Additional keyword arguments to pass to igraph.plot.

        Returns
        -------
        matplotlib.axes.Axes
            The axis containing the drawn graph.
        """
        if ax is None:
            _, ax = plt.subplots()
        tanner_graph = self.tanner_graph
        g: ig.Graph
        g = tanner_graph.subgraph(tanner_graph.vs.select(pauli_ne="X"))
        if not show_lattice:
            g = g.subgraph_edges(g.es.select(kind="tanner"))

        color_dict = {"r": "red", "g": "green", "b": "blue"}
        g.vs["color"] = ["black" if c is None else color_dict[c] for c in g.vs["color"]]
        if show_lattice:
            links = g.es.select(kind="lattice")
            links["color"] = [color_dict[c] for c in links["color"]]

        ig.plot(g, target=ax, **kwargs)
        if show_axes:
            ax.spines["top"].set_visible(True)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.xaxis.set_major_locator(AutoLocator())  # solution
            ax.yaxis.set_major_locator(AutoLocator())

        return ax

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
        return self.detectors[detector_id]

    def decompose_detector_error_model(
        self,
        color: COLOR_LABEL,
        decompose_non_edge_like_errors: bool = True,
        symbolic: bool = False,
        benchmarking: bool = False,
    ) -> Tuple[
        _DemSymbolic | stim.DetectorErrorModel, _DemSymbolic | stim.DetectorErrorModel
    ]:
        """
        Decompose the detector error model (DEM) of the circuit into the
        restricted and monochromatic DEMs of a given color.

        Parameters
        ----------
        color : {'r', 'g', 'b'}
            Color of the decomposed DEMs.
        remove_non_edge_like_errors : bool, default True
            Whether to remove error mechanisms that are not edge-like.
        benchmarking : bool, default False

        Returns
        -------
        dem1, dem2: stim.DetectorErrorModel
            Restricted and monochromatic DEMs of the given color, respectively.
        """
        # assert method_handling_correlated_errors in ['ignore', 'other']

        try:
            if symbolic:
                return self.dems_sym_decomposed[color]
            else:
                return self.dems_decomposed[color]
        except KeyError:
            pass

        stability_exp = self.shape == "rec_stability"

        # Set of detector ids to be reduced
        det_ids_to_reduce = set(self.detector_ids[color])

        t0 = time.time()

        # Original DEM
        org_dem = self.org_dem
        org_probs = self.org_probs
        num_org_error_sources = org_dem.num_errors
        num_detectors = org_dem.num_detectors
        org_dem_dets = org_dem[num_org_error_sources:]
        org_dem_errors = org_dem[:num_org_error_sources]

        obs_paulis = self.obs_paulis

        if benchmarking:
            print(time.time() - t0)
            t0 = time.time()

        # Decompose into X and Z errors
        # (i.e., ignore correlations between X and Z errors)
        # from here (how to handle Y observables? - Use LLM)
        pauli_decomposed_targets_dict = {}
        pauli_decomposed_probs_dict = {}

        for i_inst, inst in enumerate(org_dem_errors):
            targets = inst.targets_copy()
            new_targets = {"Z": [], "X": []}
            new_target_ids = {"Z": set(), "X": set()}

            for target in targets:
                if target.is_logical_observable_id():
                    obsid = int(str(target)[1:])
                    obs_pauli = obs_paulis[obsid]
                    new_targets[obs_pauli].append(target)
                    new_target_ids[obs_pauli].add(f"L{obsid}")

                else:
                    detid = int(str(target)[1:])
                    try:
                        pauli = self.detectors[detid][0]["pauli"]
                    except IndexError:
                        pauli = "X" if stability_exp and detid else "Z"
                    new_targets[pauli].append(target)
                    new_target_ids[pauli].add(detid)

            for pauli in ["Z", "X"]:
                new_targets_pauli = new_targets[pauli]
                if new_targets_pauli:
                    new_target_ids_pauli = frozenset(new_target_ids[pauli])
                    try:
                        pauli_decomposed_probs_dict[new_target_ids_pauli].append(i_inst)
                    except KeyError:
                        pauli_decomposed_probs_dict[new_target_ids_pauli] = [i_inst]
                        pauli_decomposed_targets_dict[new_target_ids_pauli] = (
                            new_targets_pauli
                        )

        # pauli_decomposed_targets = []
        # pauli_decomposed_probs = []
        # for key in pauli_decomposed_targets_dict:
        #     pauli_decomposed_targets.append(pauli_decomposed_targets_dict[
        #     key])
        #     pauli_decomposed_probs.append(pauli_decomposed_probs_dict[key])

        if benchmarking:
            print(time.time() - t0)
            t0 = time.time()

        # Obtain targets list for the two steps
        # dem1_targets_dict = {}
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

            if not decompose_non_edge_like_errors:
                if dem1_dets_sng:
                    if len(dem1_dets_sng) >= 3 or len(dem2_dets_sng) >= 2:
                        continue
                else:
                    if len(dem2_dets_sng) >= 3:
                        continue

            if dem1_det_ids:
                dem1_det_ids = frozenset(dem1_det_ids)
                try:
                    dem1_probs_dict[dem1_det_ids].extend(prob)
                    virtual_obs = dem1_virtual_obs_dict[dem1_det_ids]
                except KeyError:
                    virtual_obs = len(dem1_probs_dict)
                    dem1_obss_sng.append(stim.target_logical_observable_id(virtual_obs))
                    dem1_probs_dict[dem1_det_ids] = prob
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

        if benchmarking:
            print(time.time() - t0)
            t0 = time.time()

        # Convert dem1 information to lists
        dem1_probs = list(dem1_probs_dict.values())
        dem1_dets = [dem1_dets_dict[key] for key in dem1_probs_dict]
        dem1_obss = [dem1_obss_dict[key] for key in dem1_probs_dict]

        # Convert to DemTuple objects
        dem1_sym = _DemSymbolic(
            dem1_probs, dem1_dets, dem1_obss, org_dem_dets, org_dem.num_errors
        )
        dem2_sym = _DemSymbolic(
            dem2_probs, dem2_dets, dem2_obss, org_dem_dets, org_dem.num_errors
        )

        self.dems_sym_decomposed[color] = dem1_sym, dem2_sym

        if symbolic:
            return dem1_sym, dem2_sym

        else:
            dem1 = dem1_sym.to_dem(org_probs)
            dem2 = dem2_sym.to_dem(org_probs, sort=True)
            self.dems_decomposed[color] = dem1, dem2

            return dem1, dem2

    def decode_bp(
        self,
        detector_outcomes: np.ndarray,
        max_iter: int = 10,
        **kwargs,
    ):
        """
        Decode detector outcomes using belief propagation.

        This method uses the LDPC belief propagation decoder to decode the detector outcomes.
        It converts the detector error model to a parity check matrix and probability vector,
        then uses these to initialize a BpDecoder.

        Parameters
        ----------
        detector_outcomes : np.ndarray
            1D or 2D array of detector measurement outcomes to decode.
        max_iter : int
            Maximum number of belief propagation iterations to perform.
        **kwargs
            Additional keyword arguments to pass to the BpDecoder constructor.

        Returns
        -------
        pred : np.ndarray
            Predicted error pattern.
        llrs : np.ndarray
            Log probability ratios for each bit in the predicted error pattern.
        converge : bool
            Whether the belief propagation algorithm converged within max_iter iterations.
        """
        bp_inputs = self._bp_inputs
        if bp_inputs:
            H = bp_inputs["H"]
            p = bp_inputs["p"]
        else:
            if self.comparative_decoding:
                dem = remove_obs_from_dem(self.org_dem)
            else:
                dem = self.org_dem
            H, p = dem_to_parity_check(dem)
            bp_inputs["H"] = H
            bp_inputs["p"] = p

        if detector_outcomes.ndim == 1:
            bpd = BpDecoder(H, error_channel=p, max_iter=max_iter, **kwargs)
            pred = bpd.decode(detector_outcomes)
            llrs = bpd.log_prob_ratios
            converge = bpd.converge
        elif detector_outcomes.ndim == 2:
            pred = []
            llrs = []
            converge = []
            for det_sng in detector_outcomes:
                bpd = BpDecoder(H, error_channel=p, max_iter=max_iter, **kwargs)
                pred.append(bpd.decode(det_sng))
                llrs.append(bpd.log_prob_ratios)
                converge.append(bpd.converge)
            pred = np.stack(pred, axis=0)
            llrs = np.stack(llrs, axis=0)
            converge = np.stack(converge, axis=0)
        else:
            raise ValueError

        return pred, llrs, converge

    def decode(
        self,
        detector_outcomes: np.ndarray,
        dems: (
            Dict[
                Literal[
                    "r",
                    "g",
                    "b",
                ],
                Tuple[stim.DetectorErrorModel, stim.DetectorErrorModel],
            ]
            | None
        ) = None,
        colors: str | List[str] = "all",
        logical_value: int | Sequence[int] | None = None,
        bp_predecoding: bool = False,
        bp_prms: dict | None = None,
        erasure_matcher_predecoding: bool = False,
        partial_correction_by_predecoding: bool = False,
        full_output: bool = False,
        verbose: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, dict]:
        """
        Decode given detector outcomes using given decomposed DEMs.

        Parameters
        ----------
        detector_outcomes : 2D array-like of bool
            Array of input detector outcomes. Each row corresponds to a sample
            and each column corresponds to a detector. detector_outcomes[i, j]
            is True if and only if the detector with id j in the ith sample has
            the outcome −1.
        dems : dict with keys of {'r', 'g', 'b'} or None, default None
            Decomposed DEMs. dems['r'] = (dem1, dem2), which are the
            red-restricted and red-only DEMs, and similarly for the other
            colors.
        colors : str | List[str], default 'all'
            Colors to use for decoding. Can be 'all', one of {'r', 'g', 'b'},
            or a list containing any combination of {'r', 'g', 'b'}.
        logical_value : int | Sequence[int] | None, default None
            Logical value(s) to use for decoding. If None, all possible logical value combinations (i.e., logical classes) will be tried and the one with minimum weight will be selected.
        bp_predecoding : bool, default False
            Whether to use belief propagation as a pre-decoding step.
        bp_prms : dict, default None
            Parameters for the belief propagation decoder.
        erasure_matcher_predecoding : bool, default False
            Whether to use erasure matcher as a pre-decoding step.
        partial_correction_by_predecoding : bool, default False
            Whether to use the prediction from the erasure matcher predecoding as a partial correction for the second round of decoding, in the case that the predecoding fails to find a valid prediction.
        full_output : bool, default False
            Whether to return additional information about the decoding process.
        verbose : bool, default False
            Whether to print additional information during decoding.

        Returns
        -------
        obs_preds : 1D or 2D numpy array of bool
            Predicted observables. It is 1D if there is only one observable and
            2D if otherwise. obs_preds[i] or obs_preds[i,j] is True if and only
            if the j-th observable (j=0 when 1D) of the i-th sample is
            predicted to be -1.
        extra_outputs : dict, only when full_output is True
            Dictionary containing additional decoding results.
        """

        if erasure_matcher_predecoding:
            assert self.comparative_decoding

        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)

        if colors == "all":
            colors = ["r", "g", "b"]
        elif colors in ["r", "g", "b"]:
            colors = [colors]

        num_obs = self.num_obs

        all_logical_values = np.array(
            list(itertools.product([False, True], repeat=num_obs))
        )

        if bp_predecoding:
            if bp_prms is None:
                bp_prms = {}
            _, llrs, _ = self.decode_bp(detector_outcomes, **bp_prms)
            bp_probs = 1 / (1 + np.exp(llrs))
            eps = 1e-14
            bp_probs = bp_probs.clip(eps, 1 - eps)

            obs_preds = []
            extra_outputs = {}
            for det_outcomes_sng in detector_outcomes:
                dems = {}
                for c in colors:
                    dem1_sym, dem2_sym = self.decompose_detector_error_model(
                        c, symbolic=True
                    )
                    dem1 = dem1_sym.to_dem(self._bp_inputs["p"])
                    dem2 = dem2_sym.to_dem(self._bp_inputs["p"], sort=True)
                    dems[c] = (dem1, dem2)

                results = self.decode(
                    det_outcomes_sng.reshape(1, -1),
                    dems,
                    logical_value=logical_value,
                    full_output=full_output,
                )
                if full_output:
                    obs_preds_sng, extra_outputs_sng = results
                    for k, v in extra_outputs_sng.items():
                        try:
                            extra_outputs[k].append(v)
                        except KeyError:
                            extra_outputs[k] = [v]
                else:
                    obs_preds_sng = results
                obs_preds.append(obs_preds_sng)

            obs_preds = np.concatenate(obs_preds, axis=0)
            for k, v in extra_outputs.items():
                extra_outputs[k] = np.concatenate(v, axis=0)

            if full_output:
                return obs_preds, extra_outputs
            else:
                return obs_preds

        if dems is None:
            dems = {c: self.decompose_detector_error_model(c) for c in colors}

        # First round
        preds_dem1_all = []
        num_logical_classes = (
            len(all_logical_values)
            if self.comparative_decoding and logical_value is None
            else 1
        )
        if verbose:
            print("First-round decoding:")
        for i in range(num_logical_classes):
            preds_dem1_all.append({})
            for c, (dem1, _) in dems.items():
                if verbose:
                    print(f"    > logical class {i}, color {c}...")
                if self.comparative_decoding:
                    detector_outcomes_copy = detector_outcomes.copy()
                    if logical_value is not None:
                        detector_outcomes_copy[:, -num_obs:] = logical_value
                    else:
                        detector_outcomes_copy[:, -num_obs:] = all_logical_values[i]
                    preds_dem1_all[i][c] = self._decode_dem1(
                        dem1, detector_outcomes_copy, c
                    )
                else:
                    preds_dem1_all[i][c] = self._decode_dem1(dem1, detector_outcomes, c)

        # Erasure matcher predecoding
        if erasure_matcher_predecoding:
            assert len(preds_dem1_all) > 1

            if verbose:
                print("Erasure matcher predecoding:")
            (
                predecoding_obs_preds,
                predecoding_error_preds,
                predecoding_weights,
                predecoding_success,
            ) = self._erasure_matcher_predecoding(preds_dem1_all, detector_outcomes)

            predecoding_failure = ~predecoding_success
            detector_outcomes_left = detector_outcomes[predecoding_failure, :]
            preds_dem1_left = [
                {c: arr[predecoding_failure, :] for c, arr in preds_dem1_all[i].items()}
                for i in range(len(preds_dem1_all))
            ]

            if verbose:
                print(
                    "    > # of samples with successful predecoding:",
                    predecoding_success.sum(),
                )

            if partial_correction_by_predecoding:
                # When predecoding fails, use the predicted errors for partial correction
                predecoding_error_preds_failed = predecoding_error_preds[
                    predecoding_failure, :
                ]

                def get_partial_corr(matrix):
                    corr = (predecoding_error_preds_failed @ matrix.T) % 2
                    return corr.astype(bool)

                obs_partial_corr = get_partial_corr(self.obs_matrix)
                det_partial_corr = get_partial_corr(self.H)
                detector_outcomes_left ^= det_partial_corr
                for c in ["r", "g", "b"]:
                    error_mapping_arr = self.dems_sym_decomposed[c][0].error_mapping_arr
                    preds_dem1_corr = (
                        predecoding_error_preds_failed @ error_mapping_arr.T
                    ) % 2
                    preds_dem1_corr = preds_dem1_corr.astype(bool)
                    for preds_dem1_left_sng in preds_dem1_left:
                        preds_dem1_left_sng[c] ^= preds_dem1_corr

        else:
            detector_outcomes_left = detector_outcomes
            preds_dem1_left = preds_dem1_all

        # Second round
        if verbose:
            print("Second-round decoding:")
        if detector_outcomes_left.shape[0] > 0:
            obs_preds = []
            weights = []
            for i in range(len(preds_dem1_left)):
                obs_preds.append({})
                weights.append({})

                for c, (_, dem2) in dems.items():
                    if verbose:
                        print(f"    > logical class {i}, color {c}...")

                    if self.comparative_decoding:
                        detector_outcomes_copy = detector_outcomes_left.copy()
                        if logical_value is not None:
                            detector_outcomes_copy[:, -num_obs:] = logical_value
                        else:
                            detector_outcomes_copy[:, -num_obs:] = all_logical_values[i]
                        obs_preds_new, weights_new = self._decode_dem2(
                            dem2, detector_outcomes_copy, preds_dem1_left[i][c], c
                        )
                    else:
                        obs_preds_new, weights_new = self._decode_dem2(
                            dem2, detector_outcomes_left, preds_dem1_left[i][c], c
                        )

                    obs_preds[i][c] = obs_preds_new
                    weights[i][c] = weights_new

            # Obtain best prediction for each logical class
            obs_preds_final, best_colors, weights_final, logical_gaps = (
                self._get_final_predictions(obs_preds, weights)
            )

        # Merge predecoding & second-round decoding outcomes
        if erasure_matcher_predecoding and np.any(predecoding_success):
            if verbose:
                print("Merging predecoding & second-round decoding outcomes")
            # For samples with successful predecoding, use the predecoding results
            full_obs_preds_final = predecoding_obs_preds
            full_best_colors = np.full(detector_outcomes.shape[0], "P")
            full_weights_final = predecoding_weights
            full_logical_gaps = np.full(detector_outcomes.shape[0], -1)

            # For samples with failed predecoding, use the second-round decoding results
            if detector_outcomes_left.shape[0] > 0:
                if partial_correction_by_predecoding:
                    # Apply partial correction
                    obs_preds_final ^= obs_partial_corr

                full_obs_preds_final[predecoding_failure, :] = obs_preds_final
                full_best_colors[predecoding_failure] = best_colors
                full_weights_final[predecoding_failure] = weights_final
                full_logical_gaps[predecoding_failure] = logical_gaps

            obs_preds_final = full_obs_preds_final
            best_colors = full_best_colors
            weights_final = full_weights_final
            logical_gaps = full_logical_gaps

        if obs_preds_final.shape[1] == 1:
            obs_preds_final = obs_preds_final.ravel()

        if full_output:
            extra_outputs = {
                "best_colors": best_colors,
                "weights": weights_final,
            }
            if len(preds_dem1_all) > 1:
                extra_outputs["logical_gaps"] = logical_gaps
                extra_outputs["logical_values"] = all_logical_values
                if erasure_matcher_predecoding:
                    extra_outputs["erasure_matcher_success"] = predecoding_success
                    extra_outputs["predecoding_error_preds"] = predecoding_error_preds

        if full_output:
            return obs_preds_final, extra_outputs
        else:
            return obs_preds_final

    def _decode_dem1(
        self, dem1: stim.DetectorErrorModel, detector_outcomes: np.ndarray, color: str
    ) -> np.ndarray:
        det_outcomes_dem1 = detector_outcomes.copy()
        det_outcomes_dem1[:, self.detector_ids[color]] = False
        matching = pymatching.Matching.from_detector_error_model(dem1)
        preds_dem1 = matching.decode_batch(det_outcomes_dem1)
        del det_outcomes_dem1, matching

        return preds_dem1

    def _decode_dem2(
        self,
        dem2: stim.DetectorErrorModel,
        detector_outcomes: np.ndarray,
        preds_dem1: np.ndarray,
        color: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        det_outcome_dem2 = detector_outcomes.copy()
        mask = np.full_like(det_outcome_dem2, True)
        mask[:, self.detector_ids[color]] = False
        det_outcome_dem2[mask] = False
        del mask
        det_outcome_dem2 = np.concatenate([det_outcome_dem2, preds_dem1], axis=1)
        matching = pymatching.Matching.from_detector_error_model(dem2)
        preds, weights_new = matching.decode_batch(
            det_outcome_dem2, return_weights=True
        )
        del det_outcome_dem2, matching

        return preds, weights_new

    @staticmethod
    def _get_final_predictions(
        obs_preds: List[Dict[COLOR_LABEL, np.ndarray]],
        weights: List[Dict[COLOR_LABEL, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get final predictions and weights from multiple samples, logical classes, and colors.

        Parameters
        ----------
        obs_preds: list of dictionaries with keys ('r', 'g', 'b') and values (2D numpy arrays of int)
            Predictions of observables for each logical class, sample, and color.
            `obs_preds[i][c][k, l]` is the prediction for the l-th observable in the k-th sample for the i-th logical class and color `c`.
        weights : list of dictionaries with keys ('r', 'g', 'b') and values (1D numpy arrays of float)
            Weights of the predictions for each logical class, sample, and color.
            `weights[i][c][k]` is the weight of the prediction for the i-th logical class and color `c` in the k-th sample.

        Returns
        -------
        obs_preds_final : 2D numpy array of bool with shape (number of samples, number of observables)
            Final predictions of observables among different logical classes and colors that minimize the weight.
        best_colors : 1D numpy array of str with shape (number of samples,)
            Colors corresponding to the minimum weights.
        weights_final : 1D numpy array of float with shape (number of samples,)
            Weight of errors corresponding to the final prediction.
        logical_gap : None or 1D numpy array of float with shape (number of samples,)
            Logical gap value corresponding to the final prediction. Not calculated if `len(obs_preds) == 1`.
        """
        num_sets = len(obs_preds)
        if num_sets == 0 or len(weights) != num_sets:
            raise ValueError(
                "Input lists 'obs_preds' and 'weights' must be non-empty and have the same length."
            )

        # --- Data Preparation and Validation ---
        all_weights_flat = []
        all_preds_flat = []
        all_indices_flat = []  # Stores tuples of (set_index, color_str)
        num_k = -1  # Number of primary elements (N)
        num_l = -1  # Secondary dimension (M)

        for i, (preds_dict, weights_dict) in enumerate(zip(obs_preds, weights)):
            if not weights_dict:  # Check for empty dictionary in weights
                raise ValueError(f"Weight dictionary at index {i} is empty.")
            if not preds_dict:  # Check for empty dictionary in preds
                raise ValueError(f"Prediction dictionary at index {i} is empty.")
            if weights_dict.keys() != preds_dict.keys():
                raise ValueError(
                    f"Keys mismatch between weights and preds at index {i}."
                )

            for color, w_arr in weights_dict.items():
                if color not in preds_dict:
                    # This case is covered by the keys mismatch check above,
                    # but included for clarity if that check were removed.
                    raise ValueError(
                        f"Color '{color}' found in weights but not in preds at index {i}."
                    )

                p_arr = preds_dict[color]

                # Validate shapes and determine N, M on first valid entry
                if num_k == -1:
                    if w_arr.ndim != 1 or p_arr.ndim != 2:
                        raise ValueError(
                            f"Weight must be 1D and Preds must be 2D. Found shapes {w_arr.shape} and {p_arr.shape} for i={i}, c='{color}'."
                        )
                    num_k = w_arr.shape[0]
                    num_l = p_arr.shape[1]
                    if num_k == 0 or num_l == 0:
                        raise ValueError("Dimensions N and M must be greater than 0.")
                elif w_arr.shape != (num_k,) or p_arr.shape != (num_k, num_l):
                    raise ValueError(
                        f"Inconsistent shapes found for i={i}, c='{color}'. "
                        f"Expected weight shape ({num_k},) but got {w_arr.shape}. "
                        f"Expected prediction shape ({num_k}, {num_l}) but got {p_arr.shape}."
                    )

                all_weights_flat.append(w_arr)
                all_preds_flat.append(p_arr)
                all_indices_flat.append((i, color))  # Store set index and color

        if num_k == -1:
            # This happens if all dictionaries were empty, though caught earlier
            raise ValueError("No valid prediction/weight data found.")

        # --- Combine data for efficient processing ---
        # Stack weights: rows are combinations of (set, color), columns are k
        # Shape: (num_combinations, N)
        weights_stack = np.stack(all_weights_flat, axis=0)

        # Stack predictions: first dim is combinations, then N, then M
        # Shape: (num_combinations, N, M)
        preds_stack = np.stack(all_preds_flat, axis=0)

        # Convert indices to numpy array for potential advanced indexing (though used differently here)
        # Using object dtype because it contains strings (colors)
        indices_arr = np.array(
            all_indices_flat, dtype=object
        )  # Shape: (num_combinations, 2)

        # --- Find minimum weight and corresponding items for each k ---
        # Find the index of the minimum weight along the combinations dimension (axis=0) for each k
        # Shape: (N,)
        best_combo_indices = np.argmin(weights_stack, axis=0)

        # Get the minimum weights themselves (this is weights_final)
        # Shape: (N,)
        weights_final = np.min(weights_stack, axis=0)
        # Alternatively: weights_final = weights_stack[best_combo_indices, np.arange(num_k)]

        # Retrieve the corresponding (set_index, color) pairs for the minimums
        # Shape: (N, 2)
        winning_indices = indices_arr[best_combo_indices]

        # Extract the best colors
        # Shape: (N,)
        best_colors = winning_indices[:, 1].astype(str)  # Ensure correct dtype

        # Select the corresponding prediction rows using advanced indexing
        # We select from preds_stack using the best combination index for each k,
        # and we want all elements along the M dimension.
        # Shape: (N, M)
        obs_preds_final_int = preds_stack[best_combo_indices, np.arange(num_k), :]

        # Convert the final predictions to boolean as requested
        obs_preds_final = obs_preds_final_int.astype(bool)

        # --- Calculate Logical Gap (if applicable) ---
        logical_gap = None
        if num_sets > 1:
            min_weights_per_set = []
            # Group weights by original set index i
            weights_by_set = [[] for _ in range(num_sets)]
            for idx, w_arr in enumerate(all_weights_flat):
                set_idx, _ = all_indices_flat[idx]
                weights_by_set[set_idx].append(w_arr)

            # Calculate min weight within each set for each k
            for i in range(num_sets):
                if not weights_by_set[i]:
                    # Handle case where a set might have no valid entries (though prevented by earlier checks)
                    min_w_for_set_i = np.full(
                        (num_k,), np.inf
                    )  # Assign infinity if set was empty
                else:
                    # Stack weights for the current set: Shape (num_colors_in_set_i, N)
                    current_set_weights = np.stack(weights_by_set[i], axis=0)
                    # Find min across colors for each k: Shape (N,)
                    min_w_for_set_i = np.min(current_set_weights, axis=0)
                min_weights_per_set.append(min_w_for_set_i)

            # Stack the minimum weights across sets: Shape (num_sets, N)
            min_weights_per_set_stack = np.stack(min_weights_per_set, axis=0)

            # Partition to find the two smallest minimums for each k
            # We need the 0th (smallest) and 1st (second smallest) elements along axis 0 (sets)
            # np.partition is efficient for finding k-th smallest elements
            partitioned_weights = np.partition(min_weights_per_set_stack, kth=1, axis=0)

            # Logical gap is the difference between the second smallest and smallest
            # Shape: (N,)
            logical_gap = partitioned_weights[1, :] - partitioned_weights[0, :]

        return obs_preds_final, best_colors, weights_final, logical_gap

    def _find_error_set_intersection(
        self,
        preds_dem1: Dict[COLOR_LABEL, np.ndarray],
    ) -> np.ndarray:
        """
        Find the intersection of the error sets of the different colors.
        """
        possible_errors = []
        for c in ["r", "g", "b"]:
            preds_dem1_c = preds_dem1[c]
            error_mapping_arr = self.dems_sym_decomposed[c][0].error_mapping_arr
            possible_errors_c = (preds_dem1_c @ error_mapping_arr) > 0
            possible_errors.append(possible_errors_c)
        possible_errors = np.stack(possible_errors, axis=-1)

        # 2D array of bool with shape (num_samples, num_errors)
        error_set_intersection = np.all(possible_errors, axis=-1).astype(bool)

        return error_set_intersection

    def _erasure_matcher_predecoding(
        self,
        preds_dem1_all: List[Dict[COLOR_LABEL, np.ndarray]],
        detector_outcomes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predecoding using the erasure matcher method.

        Parameters
        ----------
        preds_dem1_all : List[Dict[COLOR_LABEL, numpy array of bool/int]]
            Predictions from the first round of decoding.
            `preds_dem1_all[i][c]`: 2D array with shape (number of samples, number of errors in DEM1 for color `c`) that contains the predictions for the i-th logical class and the color `c`.
        detector_outcomes : 2D numpy array of bool/int
            Detector outcomes with shape (number of samples, number of detectors)

        Returns
        -------
        obs_preds : 2D numpy array of bool
            Observable predictions with shape (number of samples, number of observables).
        error_preds : 2D numpy array of bool
            Error predictions with shape (number of samples, number of errors).
        weights : 1D numpy array of float
            Prediction weights with shape (number of samples,).
        validity : 1D numpy array of bool
            Validity of the predictions with shape (number of samples,).
            For samples with no valid predictions, `obs_preds`, `error_preds`,
            and `weights` correspond to the smallest-weight invalid predictions.
        """

        # All combinations of logical values (logical classes)
        all_logical_values = list(itertools.product([False, True], repeat=self.num_obs))
        all_logical_values = np.array(all_logical_values)

        ## Calculate the error set intersection and weights for each logical class
        error_preds_all = []
        weights_all = []
        for preds_dem1 in preds_dem1_all:
            error_preds = self._find_error_set_intersection(preds_dem1)
            llrs_all = np.log((1 - self.org_probs) / self.org_probs)
            llrs = np.zeros_like(error_preds, dtype=float)
            llrs[error_preds] = llrs_all[np.where(error_preds)[1]]
            weights = llrs.sum(axis=1)
            error_preds_all.append(error_preds)
            weights_all.append(weights)

        # (num_samples, num_logical_classes, num_errors), bool
        error_preds_all = np.stack(error_preds_all, axis=1)
        # (num_samples, num_logical_classes), float
        weights_all = np.stack(weights_all, axis=1)

        num_samples = error_preds_all.shape[0]

        ## Indices of logical classes sorted by prediction weight
        # (num_samples, num_logical_classes), int
        inds_logical_class_sorted = np.argsort(weights_all, axis=1)

        ## Error predictions for each sample, sorted by prediction weight
        # (num_samples, num_logical_classes, num_errors), bool
        error_preds_all_sorted = error_preds_all[
            np.arange(num_samples)[:, np.newaxis], inds_logical_class_sorted
        ]

        ## Weight for each sample, sorted by prediction weight
        # (num_samples, num_logical_classes), float
        weights_all_sorted = np.take_along_axis(
            weights_all, inds_logical_class_sorted, axis=1
        )

        ## Check if the predictions are valid (match with detectors & observables)
        # (num_samples, num_logical_classes), bool
        match_with_dets = np.all(
            (error_preds_all_sorted @ self.H.T.toarray()) % 2
            == detector_outcomes[:, np.newaxis, :],
            axis=-1,
        )
        # (num_samples, num_logical_classes, num_obs), bool
        logical_classes_sorted = all_logical_values[inds_logical_class_sorted]
        # (num_samples, num_logical_classes), bool
        match_with_obss = np.all(
            (error_preds_all_sorted @ self.obs_matrix.T.toarray()) % 2
            == logical_classes_sorted,
            axis=-1,
        )
        # (num_samples, num_logical_classes), bool
        validity_full = match_with_dets & match_with_obss

        ## Determine observable predictions among valid predictions for each sample
        # (num_samples,), int
        inds_first_valid_logical_classes = np.argmax(validity_full, axis=1)
        # (num_samples, num_obs), bool
        obs_preds = logical_classes_sorted[
            np.arange(num_samples), inds_first_valid_logical_classes, :
        ]
        # (num_samples,), bool
        validity = np.any(validity_full, axis=1)

        # Prediction weights
        # (num_samples,), float
        weights = weights_all_sorted[
            np.arange(num_samples), inds_first_valid_logical_classes
        ]
        weights[~validity] = np.inf

        # Prediction errors
        # (num_samples, num_errors), bool
        error_preds = error_preds_all_sorted[
            np.arange(num_samples), inds_first_valid_logical_classes, :
        ]

        return obs_preds, error_preds, weights, validity

    @timeit
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

    def simulate(
        self,
        shots: int,
        *,
        bp_predecoding: bool = False,
        bp_prms: dict | None = None,
        erasure_matcher_predecoding: bool = False,
        partial_correction_by_predecoding: bool = False,
        colors: Union[List[str], str] = "all",
        full_output: bool = False,
        alpha: float = 0.01,
        confint_method: str = "wilson",
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, dict]:
        """
        Monte-Carlo simulation of the concatenated MWPM decoder.

        Parameters
        ----------
        shots : int
            Number of shots to simulate.
        bp_predecoding : bool, default False
            If True, use belief propagation for predecoding.
        bp_prms : dict | None, default None
            Parameters for belief propagation predecoding.
        erasure_matcher_predecoding : bool, default False
            If True, use erasure matcher predecoding to identify errors common to all colors.
        partial_correction_by_predecoding : bool, default False
            If True, apply partial correction using predecoding results when erasure matcher predecoding fails.
        colors : Union[List[str], str], default 'all'
            Colors of the sub-decoding procedures to consider. Can be 'all', one of {'r', 'g', 'b'},
            or a list containing any combination of {'r', 'g', 'b'}.
        full_output : bool, default False
            If True, return additional statistics including estimated failure rate and confidence intervals.
            If False, return only the number of failures.
        alpha : float, default 0.01
            Significance level for the confidence interval calculation.
        confint_method : str, default 'wilson'
            Method to calculate the confidence interval.
            See statsmodels.stats.proportion.proportion_confint for available options.
        seed : Optional[int], default None
            Seed to initialize the random number generator.
        verbose : bool, default False
            If True, print progress information during simulation.

        Returns
        -------
        num_fails : numpy.ndarray
            Number of failures for each observable.
        extra_outputs : dict, optional
            Only returned when full_output is True. Dictionary containing additional information:
            - 'stats': Tuple of (pfail, delta_pfail) where pfail is the estimated failure rate
              and delta_pfail is the half-width of the confidence interval
            - 'fails': Boolean array indicating which samples failed
            - 'logical_gaps': Array of logical gaps (only when self.logical_gap is True)
        """
        if colors == "all":
            colors = ["r", "g", "b"]

        if verbose:
            print("Sampling...")
            time.sleep(1)

        shots = round(shots)
        det, obs = self.sample(shots, seed=seed)

        if verbose:
            print("Decomposing detector error model...")
            time.sleep(1)

        # dems = {}
        # for c in color:
        #     dems[c] = self.decompose_detector_error_model(c)

        if verbose:
            print("Decoding...")
            time.sleep(1)

        preds = self.decode(
            det,
            verbose=verbose,
            bp_predecoding=bp_predecoding,
            bp_prms=bp_prms,
            full_output=full_output,
        )
        if full_output:
            preds, extra_outputs = preds

        if verbose:
            print("Postprocessing...")
            time.sleep(1)

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

    def simulate_target_confint_gen(
        self,
        tol: Optional[float] = None,
        tol_zx: Optional[float] = None,
        rel_tol: Optional[float] = None,
        init_shots: int = 10_000,
        max_shots: Optional[int] = 160_000,
        max_time_per_round: Optional[int] = 600,
        max_time_total: Optional[int] = None,
        shots_mul_factor: int = 2,
        alpha: float = 0.01,
        confint_method: str = "wilson",
        color: Union[List[str], str] = "all",
        pregiven_shots: int = 0,
        pregiven_fails: int = 0,
        pfail_lower_bound: Optional[float] = None,
        verbose: bool = False,
    ):
        assert tol is not None or rel_tol is not None or tol_zx is not None

        shots_now = init_shots
        shots_total = pregiven_shots
        fails_total = pregiven_fails

        if shots_total > 0:
            pfail_low, pfail_high = proportion_confint(
                fails_total, shots_total, alpha=alpha, method=confint_method
            )
            pfail = (pfail_low + pfail_high) / 2
            delta_pfail = pfail_high - pfail

            if (
                (rel_tol is None or np.all(delta_pfail / pfail < rel_tol))
                and (tol is None or np.all(delta_pfail < tol))
                and (
                    tol_zx is None
                    or np.all(delta_pfail * np.sqrt(2 * (1 - pfail)) <= tol_zx)
                )
            ):
                res = pfail, delta_pfail, shots_total, fails_total
                yield res

        t0_total = time.time()
        trial = 0
        while True:
            if verbose:
                print(f"Sampling {shots_now} samples...", end="")
            t0 = time.time()
            fails_now = self.simulate(shots_now, colors=color)
            shots_total += shots_now
            fails_total += fails_now

            pfail, delta_pfail = get_pfail(
                shots_total, fails_total, alpha=alpha, confint_method=confint_method
            )

            res = pfail, delta_pfail, shots_total, fails_total
            yield res

            time_taken = time.time() - t0

            if verbose:
                if np.ndim(pfail) == 0:
                    print(
                        f" Result: {pfail:.2%} +- {delta_pfail:.2%} "
                        f"({time_taken}s taken)"
                    )
                else:
                    print(" Result: ", end="")
                    for pfail_indv, delta_pfail_indv in zip(pfail, delta_pfail):
                        print(f"{pfail_indv:.2%} +- {delta_pfail_indv:.2%}, ", end="")
                    print(f"({time_taken}s taken)")

            if (
                (rel_tol is None or np.all(delta_pfail / pfail < rel_tol))
                and (tol is None or np.all(delta_pfail < tol))
                and (
                    tol_zx is None
                    or np.all(delta_pfail * np.sqrt(2 * (1 - pfail)) <= tol_zx)
                )
            ):
                break

            if (
                trial > 0
                and max_time_total is not None
                and time.time() - t0_total >= max_time_total
            ):
                break

            if pfail_lower_bound is not None and np.all(pfail_high < pfail_lower_bound):
                break

            not_reach_max_shots = (
                max_shots is None or shots_now * shots_mul_factor <= max_shots
            )
            not_reach_max_time = (
                max_time_per_round is None
                or time_taken * shots_mul_factor <= max_time_per_round
            )

            if not_reach_max_time and not_reach_max_shots:
                shots_now *= shots_mul_factor

            trial += 1

    def simulate_target_confint(self, *args, **kwargs):
        for res in self.simulate_target_confint_gen(*args, **kwargs):
            pass
        return res
