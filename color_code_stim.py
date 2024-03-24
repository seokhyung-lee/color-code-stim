import math
import time
from functools import wraps
from typing import Union, Literal, List, Optional, Tuple, Dict, Iterable, \
    Callable

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim
from matplotlib.ticker import AutoLocator
from statsmodels.stats.proportion import proportion_confint


def timeit(func: Callable):
    @wraps(func)
    def wrap(*args, **kwargs):
        if args[0].benchmarking:
            start = time.time()
            res = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"Elapsed time for function '{func.__name__}': {elapsed:.2e} s")
        else:
            res = func(*args, **kwargs)
        return res

    return wrap


def get_pfail(shots, fails, alpha=0.01, confint_method='binom_test'):
    pfail_low, pfail_high = proportion_confint(fails,
                                               shots,
                                               alpha=alpha,
                                               method=confint_method)
    pfail = (pfail_low + pfail_high) / 2
    delta_pfail = pfail_high - pfail

    return pfail, delta_pfail


class ColorCode:
    tanner_graph: ig.Graph
    circuit: stim.Circuit
    d: int
    rounds: int
    qubit_groups: dict

    def __init__(self,
                 *,
                 d: int,
                 rounds: int,
                 shape: str = 'triangle',
                 cnot_schedule: Union[str, Iterable[int]] = 'LLB',
                 p_bitflip: float = 0.,
                 p_reset: float = 0.,
                 p_meas: float = 0.,
                 p_cnot: float = 0.,
                 p_idle: float = 0.,
                 p_circuit: Optional[float] = None,
                 # custom_noise_channel: Optional[Tuple[str, object]] = None,
                 # dem_decomposed: Optional[Dict[str, Tuple[
                 #     stim.DetectorErrorModel, stim.DetectorErrorModel]]] = None,
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
        shape : {'triangle'}, default 'triangle'
            Shape of the color code patch. Currently support only triangular patch.
        cnot_schedule : {12-tuple of integers, 'LLB', 'BKS'}, default 'LLB'
            CNOT schedule.
            If this is a 12-tuple of integers, it indicate (a, b, ... l)
            specifying the CNOT schedule.
            If this is 'LLB', it is (2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2),
            which is the optimal schedule from our paper.
            If this is 'LLB_reversed', it is (3, 4, 7, 6, 5, 2, 2, 3, 6, 5, 4, 1),
            which has the X- and Z-part reversed from 'LLB'.
            If this is 'BKS', it is (4, 1, 2, 3, 6, 5, 3, 2, 5, 6, 7, 4),
            which is the optimal schedule from [Beverland et al.,
            PRXQuantum.2.020341].
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
        benchmarking : bool, default False
            Whether to measure execution time of each step.
        """
        if isinstance(cnot_schedule, str):
            if cnot_schedule == 'BKS':
                cnot_schedule = (4, 1, 2, 3, 6, 5, 3, 2, 5, 6, 7, 4)
            elif cnot_schedule == 'LLB':
                cnot_schedule = (2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2)
            elif cnot_schedule == 'LLB_reversed':
                cnot_schedule = (3, 4, 7, 6, 5, 2, 2, 3, 6, 5, 4, 1)
            else:
                raise ValueError
        else:
            assert len(cnot_schedule) == 12

        assert d >= 3 and rounds >= 1
        assert shape in ['triangle']

        if p_circuit is not None:
            p_reset = p_meas = p_cnot = p_idle = p_circuit

        self.d = d
        self.rounds = rounds
        self.shape = shape
        self.cnot_schedule = cnot_schedule
        self.probs = {
            'bitflip': p_bitflip,
            'reset': p_reset,
            'meas': p_meas,
            'cnot': p_cnot,
            'idle': p_idle
        }
        # self.custom_noise_channel = custom_noise_channel

        self.tanner_graph = ig.Graph()
        self.circuit = stim.Circuit()

        self.benchmarking = benchmarking

        # Mapping between detector ids and ancillary qubits
        # self.detectors[detector_id] = (anc_qubit, time_coord)
        self.detectors = []

        # Detector ids grouped by colors
        self.detector_ids = {'r': [], 'g': [], 'b': []}

        # Various qubit groups
        self.qubit_groups = {}

        # Decomposed detector error models
        # It is generated when required.
        # if dem_decomposed is None:
        #     dem_decomposed = {}
        self.dems_decomposed = {}

        tanner_graph = self.tanner_graph

        self._add_tanner_graph_vertices()

        data_qubits = tanner_graph.vs.select(pauli=None)
        anc_qubits = tanner_graph.vs.select(pauli_ne=None)
        anc_Z_qubits = anc_qubits.select(pauli='Z')
        anc_X_qubits = anc_qubits.select(pauli='X')
        anc_red_qubits = anc_qubits.select(color='r')
        anc_green_qubits = anc_qubits.select(color='g')
        anc_blue_qubits = anc_qubits.select(color='b')

        self.qubit_groups.update({
            'data': data_qubits,
            'anc': anc_qubits,
            'anc_Z': anc_Z_qubits,
            'anc_X': anc_X_qubits,
            'anc_red': anc_red_qubits,
            'anc_green': anc_green_qubits,
            'anc_blue': anc_blue_qubits
        })

        self._generate_circuit()

        # Get detector list
        detector_coords_dict = self.circuit.get_detector_coordinates()
        for detector_id in range(self.circuit.num_detectors):
            coords = detector_coords_dict[detector_id]
            x = math.floor(coords[0])
            y = round(coords[1])
            t = round(coords[2])
            try:
                name = f'{x}-{y}-X'
                qubit = tanner_graph.vs.find(name=name)
            except ValueError:
                name = f'{x + 1}-{y}-Z'
                qubit = tanner_graph.vs.find(name=name)
            self.detectors.append((qubit, t))
            self.detector_ids[qubit['color']].append(detector_id)

    @timeit
    def _add_tanner_graph_vertices(self):
        shape = self.shape
        d = self.d
        tanner_graph = self.tanner_graph

        if shape in {'triangle'}:
            assert d % 2 == 1

            detid = 0
            L = round(3 * (d - 1) / 2)
            for y in range(L + 1):
                if y % 3 == 0:
                    anc_qubit_color = 'g'
                    anc_qubit_pos = 2
                elif y % 3 == 1:
                    anc_qubit_color = 'b'
                    anc_qubit_pos = 0
                else:
                    anc_qubit_color = 'r'
                    anc_qubit_pos = 1

                for x in range(y, 2 * L - y + 1, 2):
                    boundary = []
                    if y == 0:
                        boundary.append('r')
                    if x == y:
                        boundary.append('g')
                    if x == 2 * L - y:
                        boundary.append('b')
                    boundary = ''.join(boundary)
                    if not boundary:
                        boundary = None

                    if round((x - y) / 2) % 3 != anc_qubit_pos:
                        tanner_graph.add_vertex(name=f"{x}-{y}",
                                                x=x,
                                                y=y,
                                                qid=tanner_graph.vcount(),
                                                pauli=None,
                                                color=None,
                                                boundary=boundary)
                    else:
                        for pauli in ['Z', 'X']:
                            tanner_graph.add_vertex(name=f"{x}-{y}-{pauli}",
                                                    x=x,
                                                    y=y,
                                                    qid=tanner_graph.vcount(),
                                                    pauli=pauli,
                                                    color=anc_qubit_color,
                                                    boundary=boundary)
                            detid += 1

        else:
            raise ValueError

    @timeit
    def _generate_circuit(self):
        qubit_groups = self.qubit_groups
        cnot_schedule = self.cnot_schedule
        tanner_graph = self.tanner_graph
        circuit = self.circuit
        rounds = self.rounds

        probs = self.probs
        p_bitflip = probs['bitflip']
        p_reset = probs['reset']
        p_meas = probs['meas']
        p_cnot = probs['cnot']
        p_idle = probs['idle']

        # custom_noise_channel = self.custom_noise_channel

        data_qubits = qubit_groups['data']
        # anc_qubits = qubit_groups['anc']
        anc_Z_qubits = qubit_groups['anc_Z']
        anc_X_qubits = qubit_groups['anc_X']

        data_qids = data_qubits['qid']
        # anc_qids = anc_qubits['qid']
        anc_Z_qids = anc_Z_qubits['qid']
        anc_X_qids = anc_X_qubits['qid']

        num_data_qubits = len(data_qids)
        num_anc_Z_qubits = len(anc_Z_qubits)
        num_anc_X_qubits = len(anc_X_qubits)
        num_anc_qubits = num_anc_X_qubits + num_anc_Z_qubits

        num_qubits = tanner_graph.vcount()
        all_qids = list(range(num_qubits))
        all_qids_set = set(all_qids)

        # Syndrome extraction circuit without SPAM
        synd_extr_circuit_without_spam = stim.Circuit()
        for timeslice in range(1, max(cnot_schedule) + 1):
            targets = [i for i, val in enumerate(cnot_schedule)
                       if val == timeslice]
            operated_qids = set()
            for target in targets:
                if target in {0, 6}:
                    offset = (-1, 1)
                elif target in {1, 7}:
                    offset = (1, 1)
                elif target in {2, 8}:
                    offset = (2, 0)
                elif target in {3, 9}:
                    offset = (1, -1)
                elif target in {4, 10}:
                    offset = (-1, -1)
                else:
                    offset = (-2, 0)

                target_anc_qubits \
                    = anc_Z_qubits if target < 6 else anc_X_qubits
                for anc_qubit in target_anc_qubits:
                    data_qubit_x = anc_qubit['x'] + offset[0]
                    data_qubit_y = anc_qubit['y'] + offset[1]
                    data_qubit_name = f"{data_qubit_x}-{data_qubit_y}"
                    try:
                        data_qubit = tanner_graph.vs.find(name=data_qubit_name)
                    except ValueError:
                        continue
                    anc_qid = anc_qubit.index
                    data_qid = data_qubit.index
                    operated_qids.update({anc_qid, data_qid})

                    tanner_graph.add_edge(anc_qid, data_qid)
                    CX_target = [data_qid, anc_qid] if target < 6 \
                        else [anc_qid, data_qid]
                    synd_extr_circuit_without_spam.append('CX', CX_target)
                    if p_cnot > 0:
                        synd_extr_circuit_without_spam.append('DEPOLARIZE2',
                                                              CX_target,
                                                              p_cnot)

            if p_idle > 0:
                idling_qids = list(all_qids_set - operated_qids)
                synd_extr_circuit_without_spam.append("DEPOLARIZE1",
                                                      idling_qids,
                                                      p_idle)

            synd_extr_circuit_without_spam.append("TICK")

        def get_qubit_coords(qubit: ig.Vertex):
            coords = [qubit['x'], qubit['y']]
            if qubit['pauli'] == 'Z':
                coords[0] -= 0.5
            elif qubit['pauli'] == 'X':
                coords[0] += 0.5

            return tuple(coords)

        # Syndrome extraction circuit with measurement & detector
        def get_synd_extr_circuit(first=False):
            synd_extr_circuit = synd_extr_circuit_without_spam.copy()

            synd_extr_circuit.append('MRZ', anc_Z_qids, p_meas)
            for j, anc_qubit in enumerate(anc_Z_qubits):
                lookback = -num_anc_Z_qubits + j
                coords = get_qubit_coords(anc_qubit)
                coords += (0,)
                if first:
                    target = stim.target_rec(lookback)
                else:
                    target = [stim.target_rec(lookback),
                              stim.target_rec(lookback - num_anc_qubits)]
                synd_extr_circuit.append('DETECTOR', target, coords)

            synd_extr_circuit.append('MRX', anc_X_qids, p_meas)
            if not first:
                for j, anc_qubit in enumerate(anc_X_qubits):
                    lookback = -num_anc_X_qubits + j
                    coords = get_qubit_coords(anc_qubit)
                    coords += (0,)
                    target = [stim.target_rec(lookback),
                              stim.target_rec(lookback - num_anc_qubits)]
                    synd_extr_circuit.append('DETECTOR', target, coords)

            if p_reset > 0:
                synd_extr_circuit.append("X_ERROR", anc_Z_qids, p_reset)
                synd_extr_circuit.append("Z_ERROR", anc_X_qids, p_reset)
            if p_idle > 0:
                synd_extr_circuit.append("DEPOLARIZE1",
                                         data_qids,
                                         p_idle)
            if p_bitflip > 0:
                synd_extr_circuit.append("X_ERROR", data_qids, p_bitflip)

            # if custom_noise_channel is not None:
            #     synd_extr_circuit.append(custom_noise_channel[0],
            #                              data_qids,
            #                              custom_noise_channel[1])

            synd_extr_circuit.append("TICK")
            synd_extr_circuit.append("SHIFT_COORDS", (), (0, 0, 1))

            return synd_extr_circuit

        # Main circuit
        for qubit in tanner_graph.vs:
            coords = get_qubit_coords(qubit)
            circuit.append("QUBIT_COORDS", qubit.index, coords)

        # Initialize qubits
        qids_Z_reset = data_qids + anc_Z_qids
        circuit.append("RZ", qids_Z_reset)
        circuit.append("RX", anc_X_qids)

        if p_reset > 0:
            circuit.append("X_ERROR", qids_Z_reset, p_reset)
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

        # Final data qubit measurements
        circuit.append("MZ", data_qids, p_meas)
        for j_anc, anc_qubit in enumerate(anc_Z_qubits):
            anc_qubit: ig.Vertex
            ngh_data_qubits = anc_qubit.neighbors()
            lookback_inds \
                = [-num_data_qubits + data_qids.index(q.index) for q in
                   ngh_data_qubits]
            lookback_inds.append(-num_data_qubits - num_anc_qubits + j_anc)
            target = [stim.target_rec(ind) for ind in lookback_inds]
            circuit.append("DETECTOR",
                           target,
                           get_qubit_coords(anc_qubit) + (0,))
        qubits_logical_Z \
            = tanner_graph.vs.select(boundary_in=['r', 'rg', 'rb'], pauli=None)
        lookback_inds = [-num_data_qubits + data_qids.index(q.index)
                         for q in qubits_logical_Z]
        target = [stim.target_rec(ind) for ind in lookback_inds]
        circuit.append("OBSERVABLE_INCLUDE", target, 0)

    def draw_tanner_graph(self, ax=None, show_axes=False, **kwargs):
        """
        Draw the tanner graph of the code.

        Parameters
        ----------
        ax : axis object of matplotlib, optional
        show_axes : bool, default False
            Whether to show the x- and y-axis.
        """
        if ax is None:
            _, ax = plt.subplots()
        g = self.tanner_graph.copy()
        color_dict = {
            'r': 'red',
            'g': 'green',
            'b': 'blue'
        }
        g.vs['color'] = ['black' if c is None else color_dict[c]
                         for c in g.vs['color']]
        ig.plot(g, target=ax, **kwargs)
        if show_axes:
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.xaxis.set_major_locator(AutoLocator())  # solution
            ax.yaxis.set_major_locator(AutoLocator())

    def get_detector(self, detector_id):
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

    @timeit
    def decompose_detector_error_model(
            self,
            color: Literal['r', 'g', 'b'],
            remove_non_edge_like_errors=True,
            benchmarking=False
    ) -> Tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]:
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
            dem1, dem2 = self.dems_decomposed[color]
            return dem1, dem2
        except KeyError:
            pass

        # Set of detector ids to be reduced
        det_ids_to_reduce = set(self.detector_ids[color])

        t0 = time.time()

        # Original DEM
        org_dem = self.circuit.detector_error_model(flatten_loops=True)
        num_org_error_sources = org_dem.num_errors
        num_detectors = org_dem.num_detectors
        org_dem_dets = org_dem[num_org_error_sources:]
        org_dem_errors = org_dem[:num_org_error_sources]

        if benchmarking:
            print(time.time() - t0)
            t0 = time.time()

        # Decompose into X and Z errors
        # (i.e., ignore correlations between X and Z errors)
        pauli_decomposed_targets_dict = {}
        pauli_decomposed_probs_dict = {}

        for inst in org_dem_errors:
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()
            new_targets = {'Z': [], 'X': []}
            new_target_ids = {'Z': set(), 'X': set()}

            for target in targets:
                if target.is_logical_observable_id():
                    new_targets['Z'].append(target)
                    new_target_ids['Z'].add('L')
                else:
                    detid = int(str(target)[1:])
                    pauli = self.detectors[detid][0]['pauli']
                    new_targets[pauli].append(target)
                    new_target_ids[pauli].add(detid)

            for pauli in ['Z', 'X']:
                new_targets_pauli = new_targets[pauli]
                if new_targets_pauli:
                    new_target_ids_pauli = frozenset(new_target_ids[pauli])
                    try:
                        current_prob \
                            = pauli_decomposed_probs_dict[new_target_ids_pauli]
                        updated_prob \
                            = current_prob + prob - 2 * current_prob * prob
                        pauli_decomposed_probs_dict[new_target_ids_pauli] \
                            = updated_prob
                    except KeyError:
                        pauli_decomposed_probs_dict[
                            new_target_ids_pauli] = prob
                        pauli_decomposed_targets_dict[new_target_ids_pauli] \
                            = new_targets_pauli

        # pauli_decomposed_targets = []
        # pauli_decomposed_probs = []
        # for key in pauli_decomposed_targets_dict:
        #     pauli_decomposed_targets.append(pauli_decomposed_targets_dict[key])
        #     pauli_decomposed_probs.append(pauli_decomposed_probs_dict[key])

        if benchmarking:
            print(time.time() - t0)
            t0 = time.time()

        # Obtain targets list for the two steps
        dem1_targets_dict = {}
        dem1_probs_dict = {}
        dem1_virtual_obs_dict = {}

        dem2_targets_list = []
        dem2_probs = []

        for target_ids in pauli_decomposed_targets_dict:

            targets = pauli_decomposed_targets_dict[target_ids]
            prob = pauli_decomposed_probs_dict[target_ids]

            dem1_dets = []
            dem1_obss = []
            dem2_dets = []
            dem2_obss = []
            dem1_det_ids = set()

            for target in targets:
                if target.is_logical_observable_id():
                    dem2_obss.append(target)
                else:
                    det_id = int(str(target)[1:])
                    if det_id in det_ids_to_reduce:
                        dem2_dets.append(target)
                    else:
                        dem1_dets.append(target)
                        dem1_det_ids.add(det_id)

            if remove_non_edge_like_errors:
                if dem1_dets:
                    if len(dem1_dets) >= 3 or len(dem2_dets) >= 2:
                        continue
                else:
                    if len(dem2_dets) >= 3:
                        continue

            if dem1_det_ids:
                dem1_det_ids = frozenset(dem1_det_ids)
                try:
                    dem1_curr_prob = dem1_probs_dict[dem1_det_ids]
                    dem1_updated_prob \
                        = dem1_curr_prob + prob - 2 * dem1_curr_prob * prob
                    dem1_probs_dict[dem1_det_ids] = dem1_updated_prob
                    virtual_obs = dem1_virtual_obs_dict[dem1_det_ids]
                except KeyError:
                    virtual_obs = len(dem1_probs_dict)
                    dem1_obss.append(
                        stim.target_logical_observable_id(virtual_obs)
                    )
                    dem1_probs_dict[dem1_det_ids] = prob
                    dem1_targets_dict[dem1_det_ids] = dem1_dets + dem1_obss
                    dem1_virtual_obs_dict[dem1_det_ids] = virtual_obs

                virtual_det_id = num_detectors + virtual_obs
                dem2_dets.append(
                    stim.target_relative_detector_id(virtual_det_id)
                )

            # Add a virtual observable to dem2 for distinguishing error sources
            # L0: real observable. L1, L2, ...: virtual observables.
            # dem2_obss.append(
            #     stim.target_logical_observable_id(len(dem2_targets_list) + 1)
            # )
            dem2_targets_list.append(dem2_dets + dem2_obss)
            dem2_probs.append(prob)

        if benchmarking:
            print(time.time() - t0)
            t0 = time.time()

        # Create first-round DEM
        dem1 = stim.DetectorErrorModel()
        for key, prob in dem1_probs_dict.items():
            targets = dem1_targets_dict[key]
            dem1.append('error', prob, targets)

        dem1 += org_dem_dets

        # Create second-round DEM
        dem2 = stim.DetectorErrorModel()

        for prob, targets in zip(dem2_probs, dem2_targets_list):
            dem2.append('error', prob, targets)
        dem2 += org_dem_dets

        self.dems_decomposed[color] = dem1, dem2

        if benchmarking:
            print(time.time() - t0)
            t0 = time.time()

        return dem1, dem2

    @timeit
    def decode(self,
               detector_outcomes: np.ndarray,
               dems: Dict[Literal['r', 'g', 'b',],
                          Tuple[stim.DetectorErrorModel,
                                stim.DetectorErrorModel]],
               get_color: bool = False,
               verbose: bool = False):
        """
        Decode given detector outcomes using given decomposed DEMs.

        Parameters
        ----------
        detector_outcomes : 2D array-like of bool
            Array of input detector outcomes. Each row corresponds to a sample
            and each column corresponds to a detector. detector_outcomes[i, j]
            is True if and only if the detector with id j in the ith sample has
            the outcome −1.
        dems : dict with keys of {'r', 'g', 'b'}
            Decomposed DEMs. dems['r'] = (dem1, dem2), which are the
            red-restricted and red-only DEMs, and similarly for the other
            colors.
        get_color : bool, default False
            Whether to get the best color
        verbose : bool, default False

        Returns
        -------
        preds_obs : 2D numpy array of bool
            Predicted observables. preds_obs[i] is True if and only if the
            observable of the i-th sample is predicted to be -1.
        """
        preds_obs = None
        weights = None
        best_colors = None
        for c, dems_color in dems.items():
            dem1, dem2 = dems_color
            if verbose:
                print(f"color {c}, step-1 decoding..")
            preds_dem1 = self._decode_dem1(dem1, detector_outcomes, c)
            if verbose:
                print(f"color {c}, step-2 decoding..")
            preds, weights_new = self._decode_dem2(dem2,
                                                   detector_outcomes,
                                                   preds_dem1,
                                                   c)
            del preds_dem1

            if verbose:
                print(f"color {c}, postprocessing..")

            preds_obs_new = preds[:, 0].flatten()
            del preds

            if preds_obs is None:
                preds_obs = preds_obs_new
                weights = weights_new
                if get_color:
                    best_colors = np.chararray(preds_obs.size)
                    best_colors[:] = c

            else:
                cond = weights_new < weights
                preds_obs = np.where(cond, preds_obs_new, preds_obs)
                del preds_obs_new
                weights = np.where(cond, weights_new, weights)
                del weights_new
                if get_color:
                    best_colors = np.where(cond, c, best_colors)
                del cond

        preds_obs = preds_obs.astype('bool')

        if get_color:
            return preds_obs, best_colors
        else:
            return preds_obs

    def _decode_dem1(self, dem1, detector_outcomes, color):
        det_outcomes_dem1 = detector_outcomes.copy()
        det_outcomes_dem1[:, self.detector_ids[color]] = False
        matching = pymatching.Matching.from_detector_error_model(dem1)
        preds_dem1 = matching.decode_batch(det_outcomes_dem1)
        del det_outcomes_dem1, matching

        return preds_dem1

    def _decode_dem2(self, dem2, detector_outcomes, preds_dem1, color):
        det_outcome_dem2 = detector_outcomes.copy()
        mask = np.full_like(det_outcome_dem2, True)
        mask[:, self.detector_ids[color]] = False
        det_outcome_dem2[mask] = False
        del mask
        det_outcome_dem2 = np.concatenate([det_outcome_dem2, preds_dem1],
                                          axis=1)
        matching = pymatching.Matching.from_detector_error_model(dem2)
        preds, weights_new = matching.decode_batch(det_outcome_dem2,
                                                   return_weights=True)
        del det_outcome_dem2, matching
        return preds, weights_new

    @timeit
    def sample(self, shots, seed=None):
        """
        Sample detector outcomes and observables.

        Parameters
        ----------
        shots : int
            Number of samples
        seed : int, optional
            Seed to initialize the RNG

        Returns
        -------
        det : 2D numpy array of bool
            Detector outcomes. det[i,j] is True if and only if the detector
            with id j in the i-th sample has an outcome of −1.
        obs : 1D numpy array of bool
            Observable outcomes. obs[i] is True if and only if the observable
            of the i-th sample has an outcome of -1.
        """
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        det, obs = sampler.sample(shots, separate_observables=True)
        obs = obs.flatten()
        return det, obs

    def simulate(self,
                 shots: int,
                 *,
                 color: Union[List[str], str] = 'all',
                 get_stats: bool = False,
                 alpha: float = 0.01,
                 confint_method: str = 'binom_test',
                 seed: Optional[int] = None,
                 verbose: bool = False):
        """
        Monte-Carlo simulation of the concatenated MWPM decoder.

        Parameters
        ----------
        shots : int
            Number of shots
        color : 'all' or one of {'r', 'g', 'b'} or a list of {'r', 'g', 'b'}, default 'all'
            Colors of the sub-decoding procedures to consider.
        get_stats : bool, default False
            If True, return the number of failures, estimated failure rate,
            and half the width of its confidence interval.
            If False, return only the number of failures.
        alpha : float, default 0.01
            Significance level of the confidence interval of pfail.
        confint_method : str, default 'binom_test'
            Method to calculate the confidence interval.
            See statsmodels.stats.proportion.proportion_confint for possible
            options.
        seed : int, optional
            Seed to initialize the RNG.
        verbose : bool, default False

        Returns
        -------
        num_fails : int
            Number of failed samples
        pfail : float, only when get_stats is True
            Estimation of the failure rate
        delta_pfail : float, only when get_stats is True
            Half the width of the confidence interval of pfail
        """
        if color == 'all':
            color = ['r', 'g', 'b']

        if verbose:
            print("Sampling...")
            time.sleep(1)

        det, obs = self.sample(shots, seed=seed)

        if verbose:
            print("Decomposing detector error model...")
            time.sleep(1)

        dems = {}
        for c in color:
            dems[c] = self.decompose_detector_error_model(c)

        if verbose:
            print("Decoding...")
            time.sleep(1)

        preds = self.decode(det, dems, verbose=verbose)

        if verbose:
            print("Postprocessing...")
            time.sleep(1)

        fails = np.logical_xor(obs, preds)
        num_fails = np.sum(fails)

        if get_stats:
            pfail, delta_pfail = get_pfail(shots,
                                           num_fails,
                                           alpha=alpha,
                                           confint_method=confint_method)
            return num_fails, pfail, delta_pfail

        else:
            return num_fails

    def simulate_target_confint_gen(self,
                                    tol: Optional[float] = None,
                                    rel_tol: Optional[float] = None,
                                    init_shots: int = 10_000,
                                    max_shots: Optional[int] = 160_000,
                                    max_time_per_round: Optional[int] = 600,
                                    max_time_total: Optional[int] = None,
                                    shots_mul_factor: int = 2,
                                    alpha: float = 0.01,
                                    confint_method: str = 'binom_test',
                                    color: Union[List[str], str] = 'all',
                                    pregiven_shots: int = 0,
                                    pregiven_fails: int = 0,
                                    pfail_lower_bound: Optional[float] = None,
                                    verbose: bool = False):
        assert tol is not None or rel_tol is not None

        shots_now = init_shots
        shots_total = pregiven_shots
        fails_total = pregiven_fails

        if shots_total > 0:
            pfail_low, pfail_high = proportion_confint(fails_total,
                                                       shots_total,
                                                       alpha=alpha,
                                                       method=confint_method)
            pfail = (pfail_low + pfail_high) / 2
            delta_pfail = pfail_high - pfail

            if (rel_tol is None or delta_pfail / pfail < rel_tol) \
                    and (tol is None or delta_pfail < tol):
                res = pfail, delta_pfail, shots_total, fails_total
                yield res

        t0_total = time.time()
        trial = 0
        while True:
            if verbose:
                print(f"Sampling {shots_now} samples...", end='')
            t0 = time.time()
            fails_now = self.simulate(shots_now,
                                      color=color)
            shots_total += shots_now
            fails_total += fails_now

            pfail, delta_pfail = get_pfail(shots_total,
                                           fails_total,
                                           alpha=alpha,
                                           confint_method=confint_method)

            res = pfail, delta_pfail, shots_total, fails_total
            yield res

            time_taken = time.time() - t0

            if verbose:
                print(f" Result: {pfail:.2%} +- {delta_pfail:.2%} "
                      f"({time_taken}s taken)")

            if (rel_tol is None or delta_pfail / pfail < rel_tol) \
                    and (tol is None or delta_pfail < tol):
                break

            if trial > 0 and max_time_total is not None \
                    and time.time() - t0_total >= max_time_total:
                break

            if pfail_lower_bound is not None \
                    and pfail_high < pfail_lower_bound:
                break

            not_reach_max_shots \
                = max_shots is None \
                  or shots_now * shots_mul_factor <= max_shots
            not_reach_max_time \
                = max_time_per_round is None \
                  or time_taken * shots_mul_factor <= max_time_per_round

            if not_reach_max_time and not_reach_max_shots:
                shots_now *= shots_mul_factor

            trial += 1

    def simulate_target_confint(self, *args, **kwargs):
        for res in self.simulate_target_confint_gen(*args, **kwargs):
            pass
        return res
