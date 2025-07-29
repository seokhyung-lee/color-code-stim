"""
Tanner graph construction for color code quantum error correction.

This module handles the construction of Tanner graphs for different color code
patch types, separating graph construction logic from circuit generation.
"""

from typing import Any, Dict, Optional, Tuple

import igraph as ig

from .config import PATCH_TYPE, CIRCUIT_TYPE


class TannerGraphBuilder:
    """
    Builder class for constructing Tanner graphs for color code patches.

    The graph structure depends on the patch type:
    - 'tri': Triangular patches (used for tri, growing, cult+growing circuits)
    - 'rec': Rectangular patches
    - 'rec_stability': Stability experiment patches
    """

    def __init__(self, circuit_type: CIRCUIT_TYPE, d: int, d2: Optional[int] = None):
        """
        Initialize the Tanner graph builder.

        Parameters
        ----------
        circuit_type : CIRCUIT_TYPE
            Type of circuit that determines graph structure requirements.
        d : int
            Code distance for the initial patch.
        d2 : int, optional
            Code distance for the target patch (used in growing/cult+growing).
        """
        self.circuit_type = circuit_type
        self.d = d
        self.d2 = d2 or d

        # Map circuit types to patch types
        if circuit_type in {"tri", "growing", "cult+growing"}:
            self.patch_type: PATCH_TYPE = "tri"
        elif circuit_type == "rec":
            self.patch_type = "rec"
        elif circuit_type == "rec_stability":
            self.patch_type = "rec_stability"
        else:
            raise ValueError(f"Invalid circuit type: {circuit_type}")

        self.tanner_graph = ig.Graph()
        self.qubit_groups: Dict[str, Any] = {}

    def build(self) -> Tuple[ig.Graph, Dict[str, Any]]:
        """
        Build the complete Tanner graph and qubit groups.

        Returns
        -------
        Tuple[ig.Graph, Dict[str, Any]]
            The constructed Tanner graph and qubit group mappings.
        """
        # Build vertices based on patch type
        if self.patch_type == "tri":
            self._build_triangular_graph()
        elif self.patch_type == "rec":
            self._build_rectangular_graph()
        elif self.patch_type == "rec_stability":
            self._build_stability_graph()
        else:
            raise ValueError(f"Invalid patch type: {self.patch_type}")

        # Update qubit groups
        self._update_qubit_groups()

        # Add edges
        self._add_tanner_edges()

        # Assign colors to lattice edges
        self._assign_link_colors()

        return self.tanner_graph, self.qubit_groups

    def _build_triangular_graph(self) -> None:
        """Build vertices for triangular patch geometry."""
        if self.circuit_type == "tri":
            d = self.d
        else:  # growing, cult+growing
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

                if self.circuit_type in {"tri"}:
                    obs = boundary in ["r", "rg", "rb"]
                elif self.circuit_type in {"growing", "cult+growing"}:
                    obs = boundary in ["g", "gb", "rg"]
                else:
                    obs = False

                if round((x / 2 - y) / 2) % 3 != anc_qubit_pos:
                    self.tanner_graph.add_vertex(
                        name=f"{x}-{y}",
                        x=x,
                        y=y,
                        qid=self.tanner_graph.vcount(),
                        pauli=None,
                        color=None,
                        obs=obs,
                        boundary=boundary,
                    )
                else:
                    for pauli in ["Z", "X"]:
                        # Calculate ancilla qubit coordinates
                        anc_x = x - 1 if pauli == "Z" else x + 1
                        self.tanner_graph.add_vertex(
                            name=f"{anc_x}-{y}-{pauli}",
                            x=anc_x,
                            y=y,
                            face_x=x,
                            face_y=y,
                            qid=self.tanner_graph.vcount(),
                            pauli=pauli,
                            color=anc_qubit_color,
                            obs=False,
                            boundary=boundary,
                        )
                        detid += 1

    def _build_rectangular_graph(self) -> None:
        """Build vertices for rectangular patch geometry."""
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

                    self.tanner_graph.add_vertex(
                        name=f"{x}-{y}",
                        x=x,
                        y=y,
                        qid=self.tanner_graph.vcount(),
                        pauli=None,
                        color=None,
                        obs_r=obs_r,
                        obs_g=obs_g,
                        boundary=boundary,
                    )
                else:
                    for pauli in ["Z", "X"]:
                        # Calculate ancilla qubit coordinates
                        anc_x = x - 1 if pauli == "Z" else x + 1
                        self.tanner_graph.add_vertex(
                            name=f"{anc_x}-{y}-{pauli}",
                            x=anc_x,
                            y=y,
                            face_x=x,
                            face_y=y,
                            qid=self.tanner_graph.vcount(),
                            pauli=pauli,
                            color=anc_qubit_color,
                            obs_r=False,
                            obs_g=False,
                            boundary=boundary,
                        )
                        detid += 1

        # Additional corner vertex for rectangular patches
        x = 2 * L2 + 2
        y = L2 + 1
        self.tanner_graph.add_vertex(
            name=f"{x}-{y}",
            x=x,
            y=y,
            qid=self.tanner_graph.vcount(),
            pauli=None,
            color=None,
            obs_r=False,
            obs_g=False,
            boundary="rg",
        )

    def _build_stability_graph(self) -> None:
        """Build vertices for stability experiment patch geometry."""
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
                    self.tanner_graph.add_vertex(
                        name=f"{x}-{y}",
                        x=x,
                        y=y,
                        qid=self.tanner_graph.vcount(),
                        pauli=None,
                        color=None,
                        boundary=boundary,
                    )
                else:
                    for pauli in ["Z", "X"]:
                        # Calculate ancilla qubit coordinates
                        anc_x = x - 1 if pauli == "Z" else x + 1
                        self.tanner_graph.add_vertex(
                            name=f"{anc_x}-{y}-{pauli}",
                            x=anc_x,
                            y=y,
                            face_x=x,
                            face_y=y,
                            qid=self.tanner_graph.vcount(),
                            pauli=pauli,
                            color=anc_qubit_color,
                            boundary=boundary,
                        )
                        detid += 1

    def _update_qubit_groups(self) -> None:
        """Update qubit group mappings after vertex creation."""
        data_qubits = self.tanner_graph.vs.select(pauli=None)
        anc_qubits = self.tanner_graph.vs.select(pauli_ne=None)
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

    def _add_tanner_edges(self) -> None:
        """Add Tanner graph edges between ancilla and data qubits."""
        links = []
        offsets = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)]

        for anc_qubit in self.qubit_groups["anc"]:
            data_qubits = []
            for offset in offsets:
                data_qubit_x = anc_qubit["face_x"] + offset[0]
                data_qubit_y = anc_qubit["face_y"] + offset[1]
                data_qubit_name = f"{data_qubit_x}-{data_qubit_y}"
                try:
                    data_qubit = self.tanner_graph.vs.find(name=data_qubit_name)
                except ValueError:
                    continue
                data_qubits.append(data_qubit)
                self.tanner_graph.add_edge(
                    anc_qubit, data_qubit, kind="tanner", color=None
                )

            if anc_qubit["pauli"] == "Z":
                weight = len(data_qubits)
                for i in range(weight):
                    qubit = data_qubits[i]
                    next_qubit = data_qubits[(i + 1) % weight]
                    if not self.tanner_graph.are_adjacent(qubit, next_qubit):
                        link = self.tanner_graph.add_edge(
                            qubit, next_qubit, kind="lattice", color=None
                        )
                        links.append(link)

        self._links = links  # Store for color assignment

    def _assign_link_colors(self) -> None:
        """Assign colors to lattice edges based on neighboring ancilla qubits."""
        for link in self._links:
            v1, v2 = link.target_vertex, link.source_vertex
            ngh_ancs_1 = {anc.index for anc in v1.neighbors() if anc["pauli"] == "Z"}
            ngh_ancs_2 = {anc.index for anc in v2.neighbors() if anc["pauli"] == "Z"}
            symmetric_diff = ngh_ancs_1 ^ ngh_ancs_2
            if symmetric_diff:
                color = self.tanner_graph.vs[symmetric_diff.pop()]["color"]
                link["color"] = color
            else:
                # Handle edge case where no unique ancilla is found
                link["color"] = None
