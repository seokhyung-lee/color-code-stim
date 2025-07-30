import pickle
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import stim
from scipy.sparse import csc_matrix

from .circuit_builder import CircuitBuilder
from .config import CNOT_SCHEDULES, PAULI_LABEL, COLOR_LABEL, color_val_to_color
from .decoders import ConcatMatchingDecoder, BPDecoder, BeliefConcatMatchingDecoder
from .graph_builder import TannerGraphBuilder
from .cultivation import _load_cultivation_circuit
from .dem_utils.dem_decomp import DemDecomp
from .dem_utils.dem_manager import DemManager
from .noise_model import NoiseModel
from .simulation import Simulator
from .stim_utils import (
    dem_to_parity_check,
    remove_obs_from_dem,
)
from .visualization import draw_lattice, draw_tanner_graph


class ColorCode:
    """
    Class for constructing a color code circuit and simulating the concatenated MWPM decoder.

    Examples
    --------
    Triangular patch with uniform circuit-level noise of 1e-3:

    >>> from color_code_stim import ColorCode, NoiseModel
    >>> noise = NoiseModel.uniform_circuit_noise(1e-3)
    >>> colorcode = ColorCode(d=5, rounds=5, circuit_type="tri", noise_model=noise)
    >>> num_fails, info = colorcode.simulate(shots=10000, full_output=True)

    See `getting_started.ipynb` for more detailed usage.
    """

    tanner_graph: ig.Graph
    circuit: stim.Circuit
    d: int
    d2: Optional[int]
    rounds: int
    circuit_type: str
    temp_bdry_type: Literal["X", "Y", "Z", "r", "g", "b"]
    cnot_schedule: List[int]
    num_obs: int
    qubit_groups: Dict[str, ig.VertexSeq]
    obs_paulis: List[PAULI_LABEL]
    dem_xz: stim.DetectorErrorModel
    H: csc_matrix
    probs_xz: np.ndarray
    obs_matrix: csc_matrix
    detector_ids_by_color: Dict[COLOR_LABEL, List[int]]
    detectors_checks_map: List[Tuple[ig.Vertex, int]]
    cult_detector_ids: List[int]
    interface_detector_ids: List[int]
    dems_decomposed: Dict[COLOR_LABEL, DemDecomp]
    perfect_init_final: bool
    noise_model: Dict[
        Literal["bitflip", "reset", "meas", "cnot", "idle", "cult"], float
    ]
    comparative_decoding: bool
    exclude_non_essential_pauli_detectors: bool
    cultivation_circuit: Optional[stim.Circuit]
    remove_non_edge_like_errors: bool
    _benchmarking: bool
    _bp_inputs: Dict[str, Any]
    _dem_manager: Optional[DemManager]
    _concat_matching_decoder: Optional[ConcatMatchingDecoder]
    _bp_decoder: Optional[BPDecoder]
    _belief_concat_matching_decoder: Optional[BeliefConcatMatchingDecoder]
    _simulator: Optional[Simulator]

    def __init__(
        self,
        *,
        d: int,
        rounds: int,
        circuit_type: str = "tri",
        d2: int = None,
        cnot_schedule: Union[str, List[int]] = "tri_optimal",
        superdense_circuit: bool = False,
        temp_bdry_type: Optional[Literal["X", "Y", "Z", "x", "y", "z"]] = None,
        noise_model: Optional[NoiseModel] = None,
        perfect_logical_initialization: bool = False,
        perfect_logical_measurement: bool = False,
        perfect_init_final: bool = False,
        perfect_first_syndrome_extraction: bool = False,
        comparative_decoding: bool = False,
        exclude_non_essential_pauli_detectors: bool = False,
        cultivation_circuit: Optional[stim.Circuit] = None,
        remove_non_edge_like_errors: bool = True,
        shape: str = None,
        p_bitflip: float = 0.0,
        p_depol: float = 0.0,
        p_reset: float = 0.0,
        p_meas: float = 0.0,
        p_cnot: float = 0.0,
        p_idle: float = 0.0,
        p_circuit: Optional[float] = None,
        p_cult: Optional[float] = None,
        _generate_dem: bool = True,
        _decompose_dem: bool = True,
        _benchmarking: bool = False,
    ):
        """
        Class for constructing a color code circuit and simulating the
        concatenated MWPM decoder.

        Examples
        --------
        Triangular patch with uniform circuit-level noise of 1e-3:

        >>> from color_code_stim import ColorCode, NoiseModel
        >>> noise = NoiseModel.uniform_circuit_noise(1e-3)
        >>> colorcode = ColorCode(d=5, rounds=5, circuit_type="tri", noise_model=noise)
        >>> num_fails, info = colorcode.simulate(shots=10000, full_output=True)

        See `getting_started.ipynb` for more detailed usage.

        Parameters
        ----------
        d : int >= 3
            Code distance.

        rounds : int >= 1
            Number of syndrome extraction rounds.

        circuit_type : {'triangle', 'tri', 'rectangle', 'rec', 'rec_stability', 'growing', 'cult+growing'}, default 'tri'
            Circuit type.

            - 'triangle'/'tri': memory experiment of a triangular patch with distance
              `d`.

            - 'rectangle'/'rec': memory experiment of a rectangular patch with distance
              `d` and `d2`.

            - 'rec_stability': stability experiment of a rectangle-like patch with
              single-type boundaries. `d` and `d2` indicate the size of the patch,
              rather than code distances.

            - 'growing': growing operation from a triangular patch with distance `d` to
              a larger triangular patch with distance `d2`. Must be `d2 > d`.

            - 'cult+growing': cultivation on a triangular patch with distance `d`,
              followed by a growing operation to distance `d2`. Must be `d2 > d`.

        d2 : int >= 3, optional
            Second code distance required for circuit types 'rec'/'rectangle', 'growing',
            and 'cult+growing'. If not provided, `d2 = d` is used.

        cnot_schedule : {'tri_optimal', 'tri_optimal_reversed', 'superdense_default'} or list of 12 integers,
                        default 'tri_optimal'
            CNOT schedule.

            - List of 12 integers: (a, b, ... l) specifying the CNOT schedule.

            - 'tri_optimal': (2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2), which is the optimal
            schedule for the triangular color code.

            - 'tri_optimal_reversed': (3, 4, 7, 6, 5, 2, 2, 3, 6, 5, 4, 1),
            which has the X- and Z-part reversed from 'tri_optimal'.

            - 'superdense_default': (3, 1, 2, 3, 1, 2, 6, 4, 5, 6, 4, 5),
            which is used for superdense syndrome extraction circuits.

        superdense_circuit : bool, default False
            Whether to use superdense syndrome extraction circuit. When True, the syndrome
            extraction follows a 4-step pattern: (1) X-type anc → Z-type anc CNOTs,
            (2) data → anc CNOTs with spatial routing, (3) anc → data CNOTs, (4) repeat step 1.
            If True and cnot_schedule is 'tri_optimal', it automatically switches to
            'superdense_default'.

        temp_bdry_type : {'X', 'Y', 'Z', 'x', 'y', 'z'}, optional
            Type of the temporal boundaries, i.e., the reset/measurement basis of
            data qubits at the beginning and end of the circuit.
            Not supported for `rec_stability` and `cult+growing` circuits: the types of
            the temporal boundaries are fixed to red for `rec_stability` and `Y` for
            `cult+growing`. For the other circuit types, it is `Z` by default.

        noise_model : NoiseModel, optional
            Noise model specifying error rates for different operations. If provided,
            individual noise parameters (p_bitflip, p_depol, etc.) are ignored.
            If not provided, a NoiseModel is constructed from individual parameters.

        perfect_logical_initialization : bool, default False
            Whether logical initialization operations (data qubit reset) are noiseless
        perfect_logical_measurement : bool, default False
            Whether logical final measurement operations are noiseless
        perfect_init_final : bool, default False
            If True, sets both perfect_logical_initialization and perfect_logical_measurement
            to True.
        perfect_first_syndrome_extraction : bool, default False
            Whether the first syndrome extraction round is noiseless. Useful when
            starting from a perfect logical state (together with
            `perfect_logical_initialization=True`).
            *Note:* `rounds` still includes this perfect round, so set to `T + 1` where
            `T` is the number of actual faulty syndrome extraction rounds you want to consider.
        comparative_decoding : bool, default False
            Whether to use the comparative decoding technique. If True, observables are
            included as additional detectors and decoding can be done by running the
            decoder for each logical class and choosing the lowest-weight one. This also
            provides the logical gap information, which quantifies the reliability of
            decoding.
        exclude_non_essential_pauli_detectors : bool, default False
            If True and `temp_bdry_type` is not "Y", detectors with the Pauli type
            different from the temporal boundary type (e.g., X-type detectors for
            `temp_bdry_type="Z"`) are excluded from the circuit. This does not affect the
            decoding results since X and Z errors are independently decoded in our method
            and physical errors with the same pauli type as the temporal boundaries do
            not affect the logical values. If `temp_bdry_type="Y"` or
            `circuit_type="cult+growing"`, both types of detectors are required for decoding,
            so this option is ignored.
        cultivation_circuit: stim.Circuit, optional
            If given, it is used as the cultivation circuit for cultivation + growing
            circuit (`circuit_type == 'cult+growing'`). WARNING: Its validity is not
            checked internally.
        remove_non_edge_like_errors: bool, default True
            Whether to remove error mechanisms that are not edge-like when decomposing
            the detector error model.

        Parameters (legacy for backward compatibility)
        ----------
        shape: str, optional
            Same as `circuit_type`. If given, prioritized over `circuit_type`.

        p_bitflip : float, default 0
            Bit-flip noise on every data qubit at the start of each round.
            Ignored if noise_model is provided.

        p_depol : float, default 0
            Depolarizing noise on every data qubit at the start of each round.
            Ignored if noise_model is provided.

        p_reset : float, default 0
            Error rate for each reset (producing an orthogonal state).
            Ignored if noise_model is provided.

        p_meas : float, default 0
            Error rate for each measurement (flipped measurement outcome).
            Ignored if noise_model is provided.

        p_cnot : float, default 0
            Two-qubit depolarizing noise rate for each CNOT gate.
            Ignored if noise_model is provided.

        p_idle : float, default 0
            Single-qubit depolarizing noise rate for each idle gate.
            Ignored if noise_model is provided.

        p_circuit : float, optional
            If given, p_reset = p_meas = p_cnot = p_idle = p_circuit.
            Ignored if noise_model is provided.

        p_cult : float, optional
            Physical error rate during cultivation (only used for 'cult+growing'
            circuits). If not given, `p_cult = p_cnot` is used.
            Ignored if noise_model is provided.

        """
        # Automatic cnot_schedule selection for superdense circuits
        if superdense_circuit and cnot_schedule == "tri_optimal":
            cnot_schedule = "superdense_default"

        if isinstance(cnot_schedule, str):
            if cnot_schedule in CNOT_SCHEDULES:
                cnot_schedule = CNOT_SCHEDULES[cnot_schedule]
            else:
                raise ValueError(f"Invalid cnot schedule: {cnot_schedule}")
        else:
            cnot_schedule = list(cnot_schedule)
            assert len(cnot_schedule) == 12

        assert d > 1 and rounds >= 1

        # Handle noise model: use provided NoiseModel or create from individual parameters
        if noise_model is not None:
            # Use provided NoiseModel
            self.noise_model = noise_model
        else:
            # Create NoiseModel from individual parameters
            if p_circuit is not None:
                p_reset = p_meas = p_cnot = p_idle = p_circuit

            # For cult+growing, validate requirements
            if circuit_type in {"cultivation+growing", "cult+growing"}:
                if p_circuit is None:
                    raise ValueError(
                        "p_circuit must be provided for cult+growing circuit type"
                    )
                if p_bitflip > 0:
                    raise ValueError(
                        "p_bitflip must be 0 for cult+growing circuit type"
                    )

            # Create NoiseModel from individual parameters
            self.noise_model = NoiseModel(
                bitflip=p_bitflip,
                depol=p_depol,
                reset=p_reset,
                meas=p_meas,
                cnot=p_cnot,
                idle=p_idle,
                cult=p_cult,
            )

        self.d = d
        d2 = self.d2 = d if d2 is None else d2
        self.rounds = rounds

        if shape is not None:
            circuit_type = shape

        if circuit_type in {"triangle", "tri"}:
            assert d % 2 == 1
            self.circuit_type = "tri"
            self.num_obs = 1

        elif circuit_type in {"rectangle", "rec"}:
            assert d2 is not None
            assert d % 2 == 0 and d2 % 2 == 0
            self.circuit_type = "rec"
            self.num_obs = 2

        elif circuit_type == "rec_stability":
            assert d2 is not None
            assert d % 2 == 0 and d2 % 2 == 0
            self.circuit_type = "rec_stability"
            self.num_obs = 2

        elif circuit_type == "growing":
            assert d2 is not None
            assert d % 2 == 1 and d2 % 2 == 1 and d2 > d
            self.circuit_type = "growing"
            self.num_obs = 1

        elif circuit_type in {"cultivation+growing", "cult+growing"}:
            assert d2 is not None
            assert d % 2 == 1 and d2 % 2 == 1 and d2 > d
            self.circuit_type = "cult+growing"
            self.num_obs = 1

        else:
            raise ValueError(f"Invalid circuit type: {circuit_type}")

        if temp_bdry_type is None:
            if circuit_type == "rec_stability":
                temp_bdry_type = "r"
            elif circuit_type == "cult+growing":
                temp_bdry_type = "Y"
            else:
                temp_bdry_type = "Z"
        else:
            assert temp_bdry_type in {"X", "Y", "Z", "x", "y", "z"}
            assert circuit_type not in {"rec_stability", "cult+growing"}
            temp_bdry_type = temp_bdry_type.upper()

        self.temp_bdry_type = temp_bdry_type

        if circuit_type == "rec_stability":
            self.obs_paulis = ["Z", "X"]
        else:
            self.obs_paulis = [temp_bdry_type] * self.num_obs

        self.cnot_schedule = cnot_schedule
        self.superdense_circuit = superdense_circuit
        self.perfect_init_final = perfect_init_final

        # Handle backward compatibility: perfect_init_final sets both initialization and measurement
        if perfect_init_final:
            perfect_logical_initialization = True
            perfect_logical_measurement = True

        self.perfect_logical_initialization = perfect_logical_initialization
        self.perfect_logical_measurement = perfect_logical_measurement
        self.perfect_first_syndrome_extraction = perfect_first_syndrome_extraction

        self.comparative_decoding = comparative_decoding

        self.exclude_non_essential_pauli_detectors = (
            exclude_non_essential_pauli_detectors
        )

        self.remove_non_edge_like_errors = remove_non_edge_like_errors

        if self.comparative_decoding and self.circuit_type == "rec_stability":
            raise NotImplementedError

        if self.circuit_type == "cult+growing":
            if cultivation_circuit is None:
                cultivation_circuit = _load_cultivation_circuit(
                    d=d, p=self.noise_model["cult"]
                )

        else:
            cultivation_circuit = None
        self.cultivation_circuit = cultivation_circuit

        self._benchmarking = _benchmarking

        # Build Tanner graph using TannerGraphBuilder
        graph_builder = TannerGraphBuilder(
            circuit_type=self.circuit_type,
            d=self.d,
            d2=self.d2,
        )
        self.tanner_graph, self.qubit_groups = graph_builder.build()

        # Generate circuit using CircuitBuilder
        builder = CircuitBuilder(
            d=self.d,
            d2=self.d2,
            rounds=self.rounds,
            circuit_type=self.circuit_type,
            cnot_schedule=self.cnot_schedule,
            superdense_circuit=self.superdense_circuit,
            temp_bdry_type=self.temp_bdry_type,
            noise_model=self.noise_model,
            perfect_init_final=self.perfect_init_final,
            perfect_logical_initialization=self.perfect_logical_initialization,
            perfect_logical_measurement=self.perfect_logical_measurement,
            perfect_first_syndrome_extraction=self.perfect_first_syndrome_extraction,
            tanner_graph=self.tanner_graph,
            qubit_groups=self.qubit_groups,
            exclude_non_essential_pauli_detectors=self.exclude_non_essential_pauli_detectors,
            cultivation_circuit=self.cultivation_circuit,
            comparative_decoding=self.comparative_decoding,
        )
        self.circuit = builder.build()

        # Initialize DEM manager (lazy loading)
        self._dem_manager = None
        self._generate_dem = _generate_dem
        self._decompose_dem = _decompose_dem

        # Initialize decoders (lazy loading)
        self._concat_matching_decoder = None
        self._bp_decoder = None
        self._belief_concat_matching_decoder = None
        self._simulator = None

        self._bp_inputs = {}

    @property
    def dem_manager(self) -> DemManager:
        """Detector error model manager for circuit analysis and decomposition."""
        if self._dem_manager is None:
            if self._generate_dem:
                self._dem_manager = DemManager(
                    circuit=self.circuit,
                    tanner_graph=self.tanner_graph,
                    circuit_type=self.circuit_type,
                    comparative_decoding=self.comparative_decoding,
                    remove_non_edge_like_errors=self.remove_non_edge_like_errors,
                )
            else:
                # Create a minimal DEM manager for backward compatibility
                # when _generate_dem is False
                raise NotImplementedError("DEM generation is disabled")
        return self._dem_manager

    # Property delegation for backward compatibility
    @property
    def dem_xz(self) -> stim.DetectorErrorModel:
        """Detector error model with X and Z parts separated."""
        return self.dem_manager.dem_xz

    @property
    def H(self) -> csc_matrix:
        """Parity check matrix derived from the detector error model."""
        return self.dem_manager.H

    @property
    def obs_matrix(self) -> csc_matrix:
        """Logical observable matrix derived from the detector error model."""
        return self.dem_manager.obs_matrix

    @property
    def probs_xz(self) -> np.ndarray:
        """Probability values for each error mechanism in the detector error model."""
        return self.dem_manager.probs_xz

    @property
    def detector_ids_by_color(self) -> Dict[COLOR_LABEL, List[int]]:
        """Mapping of color labels ('r', 'g', 'b') to their corresponding detector IDs."""
        return self.dem_manager.detector_ids_by_color

    @property
    def cult_detector_ids(self) -> List[int]:
        """List of detector IDs associated with cultivation operations."""
        return self.dem_manager.cult_detector_ids

    @property
    def interface_detector_ids(self) -> List[int]:
        """List of detector IDs at the interface between cultivation and growing regions."""
        return self.dem_manager.interface_detector_ids

    @property
    def detectors_checks_map(self) -> List[Tuple[ig.Vertex, int]]:
        """Mapping from detector IDs to their corresponding ancilla qubits and rounds."""
        return self.dem_manager.detectors_checks_map

    @property
    def dems_decomposed(self) -> Dict[COLOR_LABEL, DemDecomp]:
        """Color-decomposed detector error models for individual MWPM decoding."""
        return self.dem_manager.dems_decomposed

    @property
    def concat_matching_decoder(self) -> ConcatMatchingDecoder:
        """Concatenated MWPM decoder."""
        if self._concat_matching_decoder is None:
            self._concat_matching_decoder = ConcatMatchingDecoder(
                dem_manager=self.dem_manager,
            )
        return self._concat_matching_decoder

    @property
    def bp_decoder(self) -> BPDecoder:
        """Belief propagation decoder."""
        if self._bp_decoder is None:
            self._bp_decoder = BPDecoder(
                dem_manager=self.dem_manager,
                comparative_decoding=self.comparative_decoding,
                cache_inputs=True,
            )
        return self._bp_decoder

    @property
    def belief_concat_matching_decoder(self) -> BeliefConcatMatchingDecoder:
        """BP + ConcatMWPM decoder."""
        if self._belief_concat_matching_decoder is None:
            self._belief_concat_matching_decoder = BeliefConcatMatchingDecoder(
                dem_manager=self.dem_manager,
                circuit_type=self.circuit_type,
                num_obs=self.num_obs,
                comparative_decoding=self.comparative_decoding,
                bp_cache_inputs=True,
            )
        return self._belief_concat_matching_decoder

    @property
    def simulator(self) -> Simulator:
        """Quantum circuit simulator for sampling detector and observable outcomes."""
        if self._simulator is None:
            self._simulator = Simulator(
                circuit=self.circuit,
                circuit_type=self.circuit_type,
            )
        return self._simulator

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

    def get_decomposed_dems(
        self, color: COLOR_LABEL
    ) -> Tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]:
        """
        Get decomposed detector error models for a specific color.

        Parameters
        ----------
        color : COLOR_LABEL
            Color ('r', 'g', or 'b') for which to get decomposed DEMs

        Returns
        -------
        tuple
            (dem1, dem2) - Stage 1 and stage 2 detector error models
        """
        return self.dem_manager.get_decomposed_dems(color)

    def draw_lattice(
        self,
        ax: Optional[plt.Axes] = None,
        show_axes: bool = False,
        highlight_qubits: Optional[
            List[int] | List[Tuple[float, float]] | List[str] | np.ndarray
        ] = None,
        highlight_qubits2: Optional[
            List[int] | List[Tuple[float, float]] | List[str] | np.ndarray
        ] = None,
        highlight_faces: Optional[
            List[int] | List[Tuple[float, float]] | List[str] | np.ndarray
        ] = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Draws the color code lattice.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis on which to draw the graph. If None, a new figure and
            axis will be created.
        show_axes : bool, default False
            Whether to show the x- and y-axis.
        highlight_qubits : list[int] | list[tuple] | list[str] | np.ndarray, optional
            Data qubits to highlight with orange triangles (by default).
            Can be a list of data qubit indices (ordered by code.vs.select(pauli=None)),
            a list of coordinate tuples [(x, y), ...], or a list of qubit names ['x-y', ...].
        highlight_qubits2 : list[int] | list[tuple] | list[str] | np.ndarray, optional
            Data qubits to highlight with purple rectangles (by default).
            Format is the same as highlight_qubits.
        highlight_faces : list[int] | list[tuple] | list[str] | np.ndarray, optional
            Z ancillary qubits whose corresponding faces should be highlighted.
            Can be a list of Z ancillary qubit indices (ordered by code.vs.select(pauli="Z")),
            a list of coordinate tuples [(x, y), ...], or a list of qubit names ['x-y', ...].
            Note that for names, the actual stored name includes a '-Z' suffix.
        edge_color : str, default 'black'
            Colors for edges.
        edge_linewidth : float, default 1.0
            Linewidth for edges.
        face_lightness : float, default 0.3
            Controls the lightness of face colors. Lower values make colors lighter.
        show_data_qubits : bool, default True
            Whether to draw circles representing data qubits.
        data_qubit_color : str, default 'black'
            Color for the data qubit circles (if shown).
        data_qubit_size : float, default 5.0
            Size for the data qubit circles (if shown).
        highlight_qubit_color : str, default 'orange'
            The color used to highlight qubits in `highlight_qubits`.
        highlight_qubit_color2 : str, default 'purple'
            The color used to highlight qubits in `highlight_qubits2`.
        highlight_qubit_marker : str, default '^' (triangle)
            The marker used to highlight qubits in `highlight_qubits`.
        highlight_qubit_marker2 : str, default 's' (square)
            The marker used to highlight qubits in `highlight_qubits2`.
        highlight_face_lightness : float, default 1.0
            Controls the lightness of the highlighted faces.

        Returns
        -------
        matplotlib.axes.Axes
            The axis containing the drawn lattice visualization.
        """
        return draw_lattice(
            self,
            ax=ax,
            show_axes=show_axes,
            highlight_qubits=highlight_qubits,
            highlight_qubits2=highlight_qubits2,
            highlight_faces=highlight_faces,
            **kwargs,
        )

    def draw_tanner_graph(
        self,
        ax: Optional[plt.Axes] = None,
        show_axes: bool = False,
        show_lattice: bool = False,
        pauli: Literal["Z", "X", "both"] = "X",
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
        pauli: Literal["Z", "X", "both"], default "X"
            Pauli type of ancillary qubits to include in the graph.
        **kwargs : dict
            Additional keyword arguments to pass to igraph.plot.

        Returns
        -------
        matplotlib.axes.Axes
            The axis containing the drawn graph.
        """
        return draw_tanner_graph(
            self,
            ax=ax,
            show_axes=show_axes,
            show_lattice=show_lattice,
            pauli=pauli,
            **kwargs,
        )

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

        if any(prob > 0 for key, prob in self.noise_model.items() if key != "bitflip"):
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

    def decode_bp(
        self,
        detector_outcomes: np.ndarray,
        max_iter: int = 10,
        **kwargs,
    ):
        """
        Decode detector outcomes using belief propagation.

        This method delegates to the BPDecoder while maintaining backward compatibility
        with the _bp_inputs caching mechanism for integration with pre-decoding.

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
        # Update _bp_inputs cache for compatibility with pre-decoding integration
        if not self._bp_inputs:
            if self.comparative_decoding:
                dem = remove_obs_from_dem(self.dem_xz)
            else:
                dem = self.dem_xz
            H, p = dem_to_parity_check(dem)
            self._bp_inputs["H"] = H
            self._bp_inputs["p"] = p

        # Delegate to BP decoder
        return self.bp_decoder.decode(detector_outcomes, max_iter=max_iter, **kwargs)

    def decode(
        self,
        detector_outcomes: np.ndarray,
        colors: str | List[str] = "all",
        logical_value: bool | Sequence[bool] | None = None,
        bp_predecoding: bool = False,
        bp_prms: dict | None = None,
        erasure_matcher_predecoding: bool = False,
        partial_correction_by_predecoding: bool = False,
        full_output: bool = False,
        check_validity: bool = False,
        verbose: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, dict]:
        """
        Decode detector outcomes using concatenated MWPM decoding.

        This method delegates to the ConcatMatchingDecoder while preserving backward
        compatibility and handling BP pre-decoding integration.

        Parameters
        ----------
        detector_outcomes : 1D or 2D array-like of bool
            Array of input detector outcomes for one or multiple samples.
            If 1D, it is interpreted as a single sample.
            If 2D, each row corresponds to a sample and each column corresponds to a
            detector. detector_outcomes[i, j] is True if and only if the detector with
            id j in the ith sample has the outcome −1.
        colors : str or list of str, default 'all'
            Colors to use for decoding. Can be 'all', one of {'r', 'g', 'b'},
            or a list containing any combination of {'r', 'g', 'b'}.
        logical_value : bool or 1D array-like of bool, optional
            Logical value(s) to use for decoding. If None, all possible logical value
            combinations (i.e., logical classes) will be tried and the one with minimum
            weight will be selected.
        bp_predecoding : bool, default False
            Whether to use belief propagation as a pre-decoding step.
        bp_prms : dict, default None
            Parameters for the belief propagation decoder.
        erasure_matcher_predecoding : bool, default False
            Whether to use erasure matcher as a pre-decoding step.
        partial_correction_by_predecoding : bool, default False
            Whether to use the prediction from the erasure matcher predecoding as a
            partial correction for the second round of decoding, in the case that the predecoding fails to find a valid prediction.
        full_output : bool, default False
            Whether to return extra information about the decoding process.
        check_validity : bool, default False
            Whether to check the validity of the predicted error patterns.
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
            Dictionary containing additional decoding outputs.
        """
        # Handle BP pre-decoding by delegating to BeliefConcatMatchingDecoder
        if bp_predecoding:
            return self.belief_concat_matching_decoder.decode(
                detector_outcomes=detector_outcomes,
                colors=colors,
                logical_value=logical_value,
                bp_prms=bp_prms,
                erasure_matcher_predecoding=erasure_matcher_predecoding,
                partial_correction_by_predecoding=partial_correction_by_predecoding,
                full_output=full_output,
                check_validity=check_validity,
                verbose=verbose,
            )

        # Delegate to ConcatMatchingDecoder for standard decoding
        return self.concat_matching_decoder.decode(
            detector_outcomes=detector_outcomes,
            colors=colors,
            logical_value=logical_value,
            erasure_matcher_predecoding=erasure_matcher_predecoding,
            partial_correction_by_predecoding=partial_correction_by_predecoding,
            full_output=full_output,
            check_validity=check_validity,
            verbose=verbose,
        )

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
        return self.simulator.sample(shots, seed=seed)

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
        return self.simulator.sample_with_errors(shots, seed=seed)

    def simulate(
        self,
        shots: int,
        *,
        bp_predecoding: bool = False,
        bp_prms: dict | None = None,
        erasure_matcher_predecoding: bool = False,
        partial_correction_by_predecoding: bool = False,
        colors: Union[List[str], str] = "all",
        alpha: float = 0.01,
        confint_method: str = "wilson",
        full_output: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        """
        Monte-Carlo simulation of the concatenated MWPM decoder. Delegated to simulator.

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
            Additional keyword arguments for the decoder (see `ColorCode.decode()`).

        Returns
        -------
        num_fails : numpy.ndarray
            Number of failures for each observable.
        extra_outputs : dict, optional
            Dictionary containing additional information:
            - 'stats': Tuple of (pfail, delta_pfail) where pfail is the estimated failure rate
              and delta_pfail is the half-width of the confidence interval
            - 'fails': Boolean array indicating which samples failed
            - 'logical_gaps': Array of logical gaps (only when self.logical_gap is True)
            - etc.
        """

        # Create decoder function for simulator
        def decoder_func(detector_outcomes, **decode_kwargs):
            return self.decode(
                detector_outcomes,
                bp_predecoding=bp_predecoding,
                bp_prms=bp_prms,
                erasure_matcher_predecoding=erasure_matcher_predecoding,
                partial_correction_by_predecoding=partial_correction_by_predecoding,
                **kwargs,  # Original kwargs from simulate
                **decode_kwargs,  # Additional kwargs from simulator
            )

        # Delegate to simulator
        result = self.simulator.simulate(
            shots=shots,
            decoder_func=decoder_func,
            colors=colors,
            alpha=alpha,
            confint_method=confint_method,
            full_output=full_output,
            seed=seed,
            verbose=verbose,
        )

        return result

    # ----- Save/Load Methods -----

    def _test_attribute_picklability(self) -> Tuple[List[str], List[str]]:
        """
        Test which attributes are picklable and which are not.

        Returns
        -------
        picklable : List[str]
            List of attribute names that can be pickled
        non_picklable : List[str]
            List of attribute names that cannot be pickled
        """
        import pickle
        import io

        picklable = []
        non_picklable = []

        for attr_name, attr_value in self.__dict__.items():
            try:
                # Try to pickle the attribute
                buffer = io.BytesIO()
                pickle.dump(attr_value, buffer)
                picklable.append(attr_name)
            except Exception:
                non_picklable.append(attr_name)

        return picklable, non_picklable

    def save(self, path: str):
        """
        Save the ColorCode object to a file using pickle.

        Automatically identifies and excludes non-picklable attributes such as igraph objects,
        complex decoder objects, and lazy-loaded managers. These will be reconstructed upon loading.

        Parameters
        ----------
        path : str
            The file path where the object should be saved.
        """
        data = self.__dict__.copy()

        # Known non-picklable attributes based on modular architecture
        known_non_picklable = [
            # igraph objects
            "tanner_graph",  # igraph.Graph
            "qubit_groups",  # Dict[str, ig.VertexSeq]
            # Complex manager and decoder objects (lazy-loaded)
            "_dem_manager",  # DemManager with igraph references
            "_concat_matching_decoder",  # ConcatMatchingDecoder
            "_bp_decoder",  # BPDecoder
            "_belief_concat_matching_decoder",  # BeliefConcatMatchingDecoder
            "_simulator",  # Simulator
            # Cache that should be reconstructed
            "_bp_inputs",  # Dictionary cache
        ]

        # Test remaining attributes for picklability and get dynamic exclusions
        temp_data = {k: v for k, v in data.items() if k not in known_non_picklable}
        _, additional_non_picklable = self._test_remaining_attributes(temp_data)

        # Combine known and discovered non-picklable attributes
        all_excluded = known_non_picklable + additional_non_picklable

        # Remove non-picklable attributes
        for key in all_excluded:
            if key in data:
                del data[key]

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _test_remaining_attributes(
        self, data_dict: dict
    ) -> Tuple[List[str], List[str]]:
        """
        Test remaining attributes for picklability after excluding known non-picklable ones.

        Parameters
        ----------
        data_dict : dict
            Dictionary of attributes to test

        Returns
        -------
        picklable : List[str]
            List of attribute names that can be pickled
        non_picklable : List[str]
            List of attribute names that cannot be pickled
        """
        import pickle
        import io

        picklable = []
        non_picklable = []

        for attr_name, attr_value in data_dict.items():
            try:
                # Try to pickle the attribute
                buffer = io.BytesIO()
                pickle.dump(attr_value, buffer)
                picklable.append(attr_name)
            except Exception:
                non_picklable.append(attr_name)

        return picklable, non_picklable

    @classmethod
    def load(cls, path: str) -> "ColorCode":
        """
        Load a ColorCode object from a file saved by the `save` method.

        Reconstructs non-picklable attributes excluded during saving using the
        modular architecture approach.

        Parameters
        ----------
        path : str
            The file path from which to load the object.

        Returns
        -------
        ColorCode
            The loaded ColorCode object.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Create a new instance without calling __init__
        instance = cls.__new__(cls)
        instance.__dict__.update(data)

        # Reconstruct non-picklable attributes in the correct order
        try:
            # Step 1: Reconstruct tanner_graph and qubit_groups
            instance._reconstruct_tanner_graph_and_qubit_groups()

            # Step 2: Clear lazy-loaded object caches (they will be recreated on demand)
            instance._dem_manager = None
            instance._concat_matching_decoder = None
            instance._bp_decoder = None
            instance._belief_concat_matching_decoder = None
            instance._simulator = None

            # Step 3: Initialize cache
            instance._bp_inputs = {}

        except Exception as e:
            print(f"Error during reconstruction: {e}")
            raise

        return instance

    def _reconstruct_tanner_graph_and_qubit_groups(self):
        """
        Reconstruct tanner_graph and qubit_groups using TannerGraphBuilder.

        This method recreates the igraph objects that cannot be pickled by using
        the same TannerGraphBuilder that was used during initialization.
        """
        # Reconstruct using TannerGraphBuilder with saved parameters
        graph_builder = TannerGraphBuilder(
            circuit_type=self.circuit_type,
            d=self.d,
            d2=self.d2,
        )
        self.tanner_graph, self.qubit_groups = graph_builder.build()
