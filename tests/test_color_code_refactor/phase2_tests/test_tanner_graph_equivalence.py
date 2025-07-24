"""
Phase 2: Tanner Graph Construction Equivalence Tests

This module tests the equivalence of Tanner graph construction between the legacy
ColorCode implementation and the refactored TannerGraphBuilder module.

Focus Areas:
- Graph structure and topology
- Vertex and edge attributes
- Qubit group organization
- Patch type mappings
- Coordinate systems and geometry
"""

import pytest
import sys
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(utils_path))

test_data_path = Path(__file__).parent.parent / "test_data"
sys.path.insert(0, str(test_data_path))

from comparison_utils import (
    create_test_instances,
    compare_tanner_graphs,
    compare_qubit_groups,
    compare_full_instances,
    assert_equivalence,
    print_comparison_report
)
from comprehensive_test_cases import (
    get_test_cases_by_category,
    get_extended_test_cases,
    get_quick_test_suite,
    get_stress_test_cases,
    get_edge_case_test_cases,
    get_test_case_name
)


class TestPhase2TannerGraphEquivalence:
    """Test suite for Phase 2: Tanner graph construction equivalence testing."""
    
    @pytest.mark.parametrize("test_params", get_quick_test_suite())
    def test_quick_graph_equivalence(self, test_params):
        """Quick Tanner graph equivalence test for all circuit types."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        graph_result = compare_tanner_graphs(legacy.tanner_graph, refactored.tanner_graph)
        
        if not graph_result.passed:
            print_comparison_report([graph_result], f"Quick Graph Test: {test_name}")
        
        assert_equivalence([graph_result], test_name)
    
    @pytest.mark.parametrize("test_params", get_test_cases_by_category()["triangular"])
    def test_triangular_graph_construction(self, test_params):
        """Test triangular Tanner graph construction equivalence."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to graph-specific results for Phase 2
        graph_results = [r for r in results if r.component in ["Tanner Graph Structure", "Qubit Groups"]]
        
        if any(not r.passed for r in graph_results):
            print_comparison_report(graph_results, f"Triangular Graph: {test_name}")
        
        assert_equivalence(graph_results, test_name)
    
    @pytest.mark.parametrize("test_params", get_test_cases_by_category()["rectangular"])
    def test_rectangular_graph_construction(self, test_params):
        """Test rectangular Tanner graph construction equivalence."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to graph-specific results for Phase 2
        graph_results = [r for r in results if r.component in ["Tanner Graph Structure", "Qubit Groups"]]
        
        if any(not r.passed for r in graph_results):
            print_comparison_report(graph_results, f"Rectangular Graph: {test_name}")
        
        assert_equivalence(graph_results, test_name)
    
    @pytest.mark.parametrize("test_params", get_test_cases_by_category()["stability"])
    def test_stability_graph_construction(self, test_params):
        """Test stability Tanner graph construction equivalence."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to graph-specific results for Phase 2
        graph_results = [r for r in results if r.component in ["Tanner Graph Structure", "Qubit Groups"]]
        
        if any(not r.passed for r in graph_results):
            print_comparison_report(graph_results, f"Stability Graph: {test_name}")
        
        assert_equivalence(graph_results, test_name)
    
    @pytest.mark.parametrize("test_params", get_test_cases_by_category()["growing"])
    def test_growing_graph_construction(self, test_params):
        """Test growing Tanner graph construction equivalence."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to graph-specific results for Phase 2
        graph_results = [r for r in results if r.component in ["Tanner Graph Structure", "Qubit Groups"]]
        
        if any(not r.passed for r in graph_results):
            print_comparison_report(graph_results, f"Growing Graph: {test_name}")
        
        assert_equivalence(graph_results, test_name)
    
    @pytest.mark.parametrize("test_params", get_test_cases_by_category()["cultivation"])
    def test_cultivation_graph_construction(self, test_params):
        """Test cultivation Tanner graph construction equivalence."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to graph-specific results for Phase 2
        graph_results = [r for r in results if r.component in ["Tanner Graph Structure", "Qubit Groups"]]
        
        if any(not r.passed for r in graph_results):
            print_comparison_report(graph_results, f"Cultivation Graph: {test_name}")
        
        assert_equivalence(graph_results, test_name)


class TestPhase2GraphStructureDetails:
    """Detailed tests for specific graph structure components."""
    
    def test_vertex_attributes_completeness(self):
        """Test that all vertex attributes are preserved."""
        test_cases = [
            {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 4, "d2": 6, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
            {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001},
        ]
        
        for test_params in test_cases:
            test_name = get_test_case_name(test_params)
            legacy, refactored = create_test_instances(test_params)
            
            # Check vertex attributes in detail
            legacy_graph = legacy.tanner_graph
            refactored_graph = refactored.tanner_graph
            
            assert legacy_graph.vcount() == refactored_graph.vcount(), f"{test_name}: Vertex count mismatch"
            
            for i in range(legacy_graph.vcount()):
                legacy_vertex = legacy_graph.vs[i]
                refactored_vertex = refactored_graph.vs[i]
                
                # Check all expected attributes exist
                expected_attrs = {"name", "x", "y", "qid", "pauli", "color"}
                
                legacy_attrs = set(legacy_vertex.attributes())
                refactored_attrs = set(refactored_vertex.attributes())
                
                assert expected_attrs.issubset(legacy_attrs), f"{test_name}: Missing expected attributes in legacy vertex {i}"
                assert expected_attrs.issubset(refactored_attrs), f"{test_name}: Missing expected attributes in refactored vertex {i}"
                assert legacy_attrs == refactored_attrs, f"{test_name}: Attribute sets differ for vertex {i}"
    
    def test_edge_structure_consistency(self):
        """Test that edge structure and attributes are consistent."""
        test_params = {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001}
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        
        legacy_graph = legacy.tanner_graph
        refactored_graph = refactored.tanner_graph
        
        assert legacy_graph.ecount() == refactored_graph.ecount(), f"{test_name}: Edge count mismatch"
        
        # Check edge types
        legacy_edge_kinds = set()
        refactored_edge_kinds = set()
        
        for edge in legacy_graph.es:
            if "kind" in edge.attributes():
                legacy_edge_kinds.add(edge["kind"])
                
        for edge in refactored_graph.es:
            if "kind" in edge.attributes():
                refactored_edge_kinds.add(edge["kind"])
        
        assert legacy_edge_kinds == refactored_edge_kinds, f"{test_name}: Edge kinds differ"
        assert "tanner" in legacy_edge_kinds, f"{test_name}: Missing tanner edges"
    
    def test_qubit_groups_organization(self):
        """Test that qubit groups are organized identically."""
        test_cases = [
            {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
            {"d": 4, "d2": 4, "rounds": 4, "circuit_type": "rec_stability", "p_circuit": 0.001},
        ]
        
        for test_params in test_cases:
            test_name = get_test_case_name(test_params)
            legacy, refactored = create_test_instances(test_params)
            
            groups_result = compare_qubit_groups(legacy.qubit_groups, refactored.qubit_groups)
            
            if not groups_result.passed:
                print_comparison_report([groups_result], f"Qubit Groups: {test_name}")
            
            assert_equivalence([groups_result], f"Qubit groups for {test_name}")


class TestPhase2PatchTypeMapping:
    """Test patch type mapping functionality from circuit types."""
    
    def test_patch_type_inference(self):
        """Test that patch types are correctly inferred from circuit types."""
        # Import TannerGraphBuilder directly to test patch_type mapping
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
        from color_code_stim.graph_builder import TannerGraphBuilder
        
        # Test triangular patch mapping
        triangular_types = ["tri", "growing", "cult+growing"]
        for circuit_type in triangular_types:
            builder = TannerGraphBuilder(circuit_type, d=3, d2=5)
            assert builder.patch_type == "tri", f"{circuit_type} should map to triangular patch"
        
        # Test rectangular patch mapping
        builder = TannerGraphBuilder("rec", d=4, d2=6)
        assert builder.patch_type == "rec", "rec should map to rectangular patch"
        
        # Test stability patch mapping
        builder = TannerGraphBuilder("rec_stability", d=4, d2=6)
        assert builder.patch_type == "rec_stability", "rec_stability should map to stability patch"
    
    def test_distance_parameter_usage(self):
        """Test that distance parameters d and d2 are used correctly."""
        # Test triangular circuits use d for structure
        tri_params = {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001}
        tri_legacy, tri_refactored = create_test_instances(tri_params)
        
        # Test growing circuits use d2 for structure
        growing_params = {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001}
        growing_legacy, growing_refactored = create_test_instances(growing_params)
        
        # Growing graph should be larger than tri graph due to d2 > d
        assert growing_legacy.tanner_graph.vcount() > tri_legacy.tanner_graph.vcount()
        assert growing_refactored.tanner_graph.vcount() > tri_refactored.tanner_graph.vcount()
        
        # But equivalence should still hold within each type
        tri_result = compare_tanner_graphs(tri_legacy.tanner_graph, tri_refactored.tanner_graph)
        growing_result = compare_tanner_graphs(growing_legacy.tanner_graph, growing_refactored.tanner_graph)
        
        assert_equivalence([tri_result], "Triangular d parameter")
        assert_equivalence([growing_result], "Growing d2 parameter")


class TestPhase2GraphGeometry:
    """Test geometric properties and coordinate systems."""
    
    def test_coordinate_consistency(self):
        """Test that qubit coordinates are consistent and valid."""
        test_params = {"d": 5, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001}
        
        legacy, refactored = create_test_instances(test_params)
        
        # Check coordinate ranges and consistency
        for graph, name in [(legacy.tanner_graph, "legacy"), (refactored.tanner_graph, "refactored")]:
            x_coords = [v["x"] for v in graph.vs]
            y_coords = [v["y"] for v in graph.vs]
            
            assert all(isinstance(x, int) for x in x_coords), f"{name}: All x coordinates should be integers"
            assert all(isinstance(y, int) for y in y_coords), f"{name}: All y coordinates should be integers"
            assert min(x_coords) >= 0, f"{name}: x coordinates should be non-negative"
            assert min(y_coords) >= 0, f"{name}: y coordinates should be non-negative"
        
        # Compare coordinate consistency between implementations
        graph_result = compare_tanner_graphs(legacy.tanner_graph, refactored.tanner_graph)
        assert_equivalence([graph_result], "Coordinate consistency")
    
    def test_boundary_attribute_consistency(self):
        """Test that boundary attributes are consistent across patch types."""
        test_cases = [
            {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 4, "d2": 6, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
            {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001},
        ]
        
        for test_params in test_cases:
            test_name = get_test_case_name(test_params)
            legacy, refactored = create_test_instances(test_params)
            
            # Check that boundary attributes exist and are consistent
            for graph, name in [(legacy.tanner_graph, "legacy"), (refactored.tanner_graph, "refactored")]:
                boundary_qubits = [v for v in graph.vs if v.get("boundary") is not None]
                
                if test_params["circuit_type"] in ["tri", "growing", "cult+growing"]:
                    # Triangular patches should have boundary qubits
                    assert len(boundary_qubits) > 0, f"{name} {test_name}: Should have boundary qubits"
                
                # All boundary values should be valid color combinations
                for qubit in boundary_qubits:
                    boundary = qubit["boundary"]
                    assert isinstance(boundary, str), f"{name} {test_name}: Boundary should be string"
                    assert all(c in "rgb" for c in boundary), f"{name} {test_name}: Invalid boundary color {boundary}"
            
            # Compare equivalence
            graph_result = compare_tanner_graphs(legacy.tanner_graph, refactored.tanner_graph)
            assert_equivalence([graph_result], f"Boundary consistency for {test_name}")


class TestPhase2StressAndEdgeCases:
    """Stress tests and edge cases for Tanner graph construction."""
    
    @pytest.mark.parametrize("test_params", get_stress_test_cases())
    def test_graph_stress_cases(self, test_params):
        """Test Tanner graph construction with large/complex parameters."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        graph_result = compare_tanner_graphs(legacy.tanner_graph, refactored.tanner_graph)
        groups_result = compare_qubit_groups(legacy.qubit_groups, refactored.qubit_groups)
        
        if not graph_result.passed or not groups_result.passed:
            print_comparison_report([graph_result, groups_result], f"Graph Stress Test: {test_name}")
        
        assert_equivalence([graph_result, groups_result], test_name)
    
    @pytest.mark.parametrize("test_params", get_edge_case_test_cases())
    def test_graph_edge_cases(self, test_params):
        """Test Tanner graph construction edge cases and boundary conditions."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        graph_result = compare_tanner_graphs(legacy.tanner_graph, refactored.tanner_graph)
        groups_result = compare_qubit_groups(legacy.qubit_groups, refactored.qubit_groups)
        
        if not graph_result.passed or not groups_result.passed:
            print_comparison_report([graph_result, groups_result], f"Graph Edge Case: {test_name}")
        
        assert_equivalence([graph_result, groups_result], test_name)


if __name__ == "__main__":
    # Run quick test for manual verification
    print("Running Phase 2 Tanner Graph Equivalence Quick Test...")
    
    quick_cases = get_quick_test_suite()[:3]  # Just first 3 for manual testing
    
    for test_params in quick_cases:
        test_name = get_test_case_name(test_params)
        print(f"\nTesting: {test_name}")
        
        try:
            legacy, refactored = create_test_instances(test_params)
            graph_result = compare_tanner_graphs(legacy.tanner_graph, refactored.tanner_graph)
            groups_result = compare_qubit_groups(legacy.qubit_groups, refactored.qubit_groups)
            
            if graph_result.passed and groups_result.passed:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                if not graph_result.passed:
                    print("Graph Result:", graph_result.get_report())
                if not groups_result.passed:
                    print("Groups Result:", groups_result.get_report())
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\nManual test completed.")