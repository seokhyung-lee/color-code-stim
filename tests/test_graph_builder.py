"""
Tests for TannerGraphBuilder class.

This module contains unit tests for the TannerGraphBuilder class to verify
correct graph construction for different circuit types and patch configurations.
"""

import pytest
import igraph as ig

from src.color_code_stim.graph_builder import TannerGraphBuilder


class TestTannerGraphBuilder:
    """Test cases for TannerGraphBuilder class."""

    def test_triangular_patch_construction(self):
        """Test triangular patch graph construction for tri circuit type."""
        builder = TannerGraphBuilder("tri", d=3)
        graph, qubit_groups = builder.build()
        
        # Basic structure checks
        assert isinstance(graph, ig.Graph)
        assert graph.vcount() > 0
        assert graph.ecount() > 0
        
        # Check qubit groups
        assert "data" in qubit_groups
        assert "anc" in qubit_groups
        assert "anc_Z" in qubit_groups
        assert "anc_X" in qubit_groups
        
        # Verify data qubits exist
        data_qubits = qubit_groups["data"]
        assert len(data_qubits) > 0
        for qubit in data_qubits:
            assert qubit["pauli"] is None
            assert qubit["color"] is None
        
        # Verify ancilla qubits exist  
        anc_qubits = qubit_groups["anc"]
        assert len(anc_qubits) > 0
        for qubit in anc_qubits:
            assert qubit["pauli"] in ["X", "Z"]
            assert qubit["color"] in ["r", "g", "b"]
    
    def test_rectangular_patch_construction(self):
        """Test rectangular patch graph construction for rec circuit type."""
        builder = TannerGraphBuilder("rec", d=4, d2=6)
        graph, qubit_groups = builder.build()
        
        # Basic structure checks
        assert isinstance(graph, ig.Graph)
        assert graph.vcount() > 0
        assert graph.ecount() > 0
        
        # Check qubit groups
        data_qubits = qubit_groups["data"]
        anc_qubits = qubit_groups["anc"]
        
        # Verify rectangular patch has obs_r and obs_g attributes
        for qubit in data_qubits:
            assert "obs_r" in qubit.attributes()
            assert "obs_g" in qubit.attributes()
    
    def test_stability_patch_construction(self):
        """Test stability patch graph construction for rec_stability circuit type."""
        builder = TannerGraphBuilder("rec_stability", d=4, d2=6)
        graph, qubit_groups = builder.build()
        
        # Basic structure checks
        assert isinstance(graph, ig.Graph)
        assert graph.vcount() > 0
        assert graph.ecount() > 0
        
        # Check qubit groups
        data_qubits = qubit_groups["data"]
        anc_qubits = qubit_groups["anc"]
        
        # Verify structure exists
        assert len(data_qubits) > 0
        assert len(anc_qubits) > 0

    def test_growing_circuit_uses_triangular_patch(self):
        """Test that growing circuit type uses triangular patch structure."""
        builder = TannerGraphBuilder("growing", d=3, d2=5)
        graph, qubit_groups = builder.build()
        
        # Should use triangular patch logic with d2
        assert isinstance(graph, ig.Graph)
        assert graph.vcount() > 0
        
        # Verify triangular-specific observable attributes
        data_qubits = qubit_groups["data"]
        for qubit in data_qubits:
            assert "obs" in qubit.attributes()  # triangular patches use 'obs'
            assert "obs_r" not in qubit.attributes()  # not rectangular attributes

    def test_cult_growing_circuit_uses_triangular_patch(self):
        """Test that cult+growing circuit type uses triangular patch structure."""
        builder = TannerGraphBuilder("cult+growing", d=3, d2=5)
        graph, qubit_groups = builder.build()
        
        # Should use triangular patch logic with d2
        assert isinstance(graph, ig.Graph)
        assert graph.vcount() > 0
        
        # Verify triangular-specific observable attributes
        data_qubits = qubit_groups["data"]
        for qubit in data_qubits:
            assert "obs" in qubit.attributes()  # triangular patches use 'obs'

    def test_patch_type_mapping(self):
        """Test correct patch type mapping from circuit types."""
        # Triangular patch types
        for circuit_type in ["tri", "growing", "cult+growing"]:
            builder = TannerGraphBuilder(circuit_type, d=3, d2=5)
            assert builder.patch_type == "tri"
        
        # Rectangular patch type
        builder = TannerGraphBuilder("rec", d=4, d2=6)
        assert builder.patch_type == "rec"
        
        # Stability patch type
        builder = TannerGraphBuilder("rec_stability", d=4, d2=6)
        assert builder.patch_type == "rec_stability"

    def test_invalid_circuit_type(self):
        """Test that invalid circuit types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid circuit type"):
            TannerGraphBuilder("invalid_type", d=3)

    def test_edge_structure(self):
        """Test that edges are correctly added to the graph."""
        builder = TannerGraphBuilder("tri", d=3)
        graph, qubit_groups = builder.build()
        
        # Check that edges exist
        assert graph.ecount() > 0
        
        # Check for different edge types
        edge_kinds = set()
        for edge in graph.es:
            edge_kinds.add(edge["kind"])
        
        assert "tanner" in edge_kinds  # Tanner graph edges
        # Lattice edges may or may not exist depending on configuration

    def test_qubit_coordinates(self):
        """Test that qubits have proper coordinate attributes."""
        builder = TannerGraphBuilder("tri", d=3)
        graph, qubit_groups = builder.build()
        
        for qubit in graph.vs:
            assert "x" in qubit.attributes()
            assert "y" in qubit.attributes()
            assert isinstance(qubit["x"], int)
            assert isinstance(qubit["y"], int)

    def test_qubit_groups_completeness(self):
        """Test that qubit groups contain all qubits."""
        builder = TannerGraphBuilder("tri", d=3)
        graph, qubit_groups = builder.build()
        
        total_qubits = graph.vcount()
        data_count = len(qubit_groups["data"])
        anc_count = len(qubit_groups["anc"])
        
        # All qubits should be either data or ancilla
        assert data_count + anc_count == total_qubits
        
        # Ancilla qubits should be split into X and Z
        anc_x_count = len(qubit_groups["anc_X"])
        anc_z_count = len(qubit_groups["anc_Z"])
        assert anc_x_count + anc_z_count == anc_count

    def test_color_groups_completeness(self):
        """Test that color groups contain all ancilla qubits."""
        builder = TannerGraphBuilder("tri", d=3)
        graph, qubit_groups = builder.build()
        
        anc_count = len(qubit_groups["anc"])
        red_count = len(qubit_groups["anc_red"])
        green_count = len(qubit_groups["anc_green"])
        blue_count = len(qubit_groups["anc_blue"])
        
        # All ancilla qubits should have a color
        assert red_count + green_count + blue_count == anc_count

    def test_distance_parameters(self):
        """Test different distance parameters."""
        # Test odd distance for triangular
        builder = TannerGraphBuilder("tri", d=5)
        graph1, _ = builder.build()
        
        builder = TannerGraphBuilder("tri", d=3)
        graph2, _ = builder.build()
        
        # Larger distance should result in more qubits
        assert graph1.vcount() > graph2.vcount()
        
        # Test even distance for rectangular
        builder = TannerGraphBuilder("rec", d=4, d2=6)
        graph3, _ = builder.build()
        
        builder = TannerGraphBuilder("rec", d=4, d2=4)
        graph4, _ = builder.build()
        
        # Larger distance should result in more qubits
        assert graph3.vcount() > graph4.vcount()

    def test_d2_parameter_usage(self):
        """Test that d2 parameter is used correctly."""
        # For tri circuit, should use d
        builder1 = TannerGraphBuilder("tri", d=3, d2=5)
        graph1, _ = builder1.build()
        
        builder2 = TannerGraphBuilder("tri", d=3)
        graph2, _ = builder2.build()
        
        # Should be identical (d2 ignored for tri)
        assert graph1.vcount() == graph2.vcount()
        
        # For growing circuit, should use d2
        builder3 = TannerGraphBuilder("growing", d=3, d2=5)
        graph3, _ = builder3.build()
        
        builder4 = TannerGraphBuilder("growing", d=3, d2=3)
        graph4, _ = builder4.build()
        
        # Different d2 should give different sizes
        assert graph3.vcount() != graph4.vcount()


@pytest.mark.parametrize("circuit_type,d,d2", [
    ("tri", 3, None),
    ("rec", 4, 6),
    ("rec_stability", 4, 6), 
    ("growing", 3, 5),
    ("cult+growing", 3, 5),
])
def test_all_circuit_types_build_successfully(circuit_type, d, d2):
    """Parametrized test to ensure all circuit types build without errors."""
    builder = TannerGraphBuilder(circuit_type, d=d, d2=d2)
    graph, qubit_groups = builder.build()
    
    # Basic validation
    assert isinstance(graph, ig.Graph)
    assert graph.vcount() > 0
    assert isinstance(qubit_groups, dict)
    assert "data" in qubit_groups
    assert "anc" in qubit_groups


if __name__ == "__main__":
    pytest.main([__file__])