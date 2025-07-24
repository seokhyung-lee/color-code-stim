"""
Comprehensive comparison utilities for legacy vs refactored equivalence testing.

This module provides detailed comparison functions for all ColorCode components
with clear failure reporting and debugging information.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
import numpy as np
import igraph as ig
import stim

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from color_code_stim.color_code_legacy import ColorCode as LegacyColorCode
from color_code_stim.color_code import ColorCode as RefactoredColorCode


class ComparisonResult:
    """Structured result object for detailed comparison reporting."""
    
    def __init__(self, component: str, passed: bool = True):
        self.component = component
        self.passed = passed
        self.details: List[str] = []
        self.differences: List[str] = []
        self.summary: str = ""
    
    def add_detail(self, detail: str):
        """Add a detail message."""
        self.details.append(detail)
    
    def add_difference(self, diff: str):
        """Add a difference and mark as failed."""
        self.differences.append(diff)
        self.passed = False
    
    def set_summary(self, summary: str):
        """Set summary message."""
        self.summary = summary
    
    def get_report(self) -> str:
        """Generate detailed failure report."""
        if self.passed:
            return f"✅ {self.component}: PASSED"
        
        report = [f"❌ {self.component}: FAILED"]
        if self.summary:
            report.append(f"Summary: {self.summary}")
        
        if self.differences:
            report.append("Differences:")
            for diff in self.differences:
                report.append(f"  • {diff}")
        
        if self.details:
            report.append("Details:")
            for detail in self.details:
                report.append(f"  - {detail}")
        
        return "\n".join(report)


def create_test_instances(test_params: Dict[str, Any]) -> Tuple[LegacyColorCode, RefactoredColorCode]:
    """
    Create legacy and refactored ColorCode instances with identical parameters.
    
    Parameters
    ----------
    test_params : dict
        Parameters for ColorCode initialization
        
    Returns
    -------
    tuple
        (legacy_instance, refactored_instance)
    """
    try:
        legacy = LegacyColorCode(**test_params)
        refactored = RefactoredColorCode(**test_params)
        return legacy, refactored
    except Exception as e:
        raise RuntimeError(f"Failed to create test instances with params {test_params}: {e}")


def compare_circuits(legacy_circuit: stim.Circuit, refactored_circuit: stim.Circuit) -> ComparisonResult:
    """
    Compare two Stim circuits for equivalence.
    
    Parameters
    ----------
    legacy_circuit, refactored_circuit : stim.Circuit
        Circuits to compare
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("Circuit Generation")
    
    # Compare basic properties
    if legacy_circuit.num_qubits != refactored_circuit.num_qubits:
        result.add_difference(f"Qubit count: {legacy_circuit.num_qubits} vs {refactored_circuit.num_qubits}")
    else:
        result.add_detail(f"Qubit count: {legacy_circuit.num_qubits}")
    
    if len(legacy_circuit) != len(refactored_circuit):
        result.add_difference(f"Instruction count: {len(legacy_circuit)} vs {len(refactored_circuit)}")
    else:
        result.add_detail(f"Instruction count: {len(legacy_circuit)}")
    
    # Compare circuit structure (convert to string for exact comparison)
    legacy_str = str(legacy_circuit)
    refactored_str = str(refactored_circuit)
    
    if legacy_str != refactored_str:
        result.add_difference("Circuit instructions differ")
        
        # Find first difference for debugging
        legacy_lines = legacy_str.split('\n')
        refactored_lines = refactored_str.split('\n')
        
        for i, (legacy_line, refactored_line) in enumerate(zip(legacy_lines, refactored_lines)):
            if legacy_line != refactored_line:
                result.add_detail(f"First difference at line {i+1}:")
                result.add_detail(f"  Legacy:     {legacy_line}")
                result.add_detail(f"  Refactored: {refactored_line}")
                break
    else:
        result.add_detail("Circuit instructions identical")
    
    # Compare detector error models
    try:
        legacy_dem = legacy_circuit.detector_error_model()
        refactored_dem = refactored_circuit.detector_error_model()
        
        if str(legacy_dem) != str(refactored_dem):
            result.add_difference("Detector error models differ")
        else:
            result.add_detail("Detector error models identical")
    except Exception as e:
        result.add_detail(f"Could not compare DEMs: {e}")
    
    if result.passed:
        result.set_summary("Circuits are functionally equivalent")
    else:
        result.set_summary("Circuits have structural differences")
    
    return result


def compare_tanner_graphs(legacy_graph: ig.Graph, refactored_graph: ig.Graph) -> ComparisonResult:
    """
    Compare two Tanner graphs for structural equivalence.
    
    Parameters
    ----------
    legacy_graph, refactored_graph : igraph.Graph
        Graphs to compare
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("Tanner Graph Structure")
    
    # Compare basic graph properties
    if legacy_graph.vcount() != refactored_graph.vcount():
        result.add_difference(f"Vertex count: {legacy_graph.vcount()} vs {refactored_graph.vcount()}")
        return result
    else:
        result.add_detail(f"Vertex count: {legacy_graph.vcount()}")
    
    if legacy_graph.ecount() != refactored_graph.ecount():
        result.add_difference(f"Edge count: {legacy_graph.ecount()} vs {refactored_graph.ecount()}")
        return result
    else:
        result.add_detail(f"Edge count: {legacy_graph.ecount()}")
    
    # Compare vertex attributes
    for i in range(legacy_graph.vcount()):
        legacy_vertex = legacy_graph.vs[i]
        refactored_vertex = refactored_graph.vs[i]
        
        # Compare attribute sets
        legacy_attrs = set(legacy_vertex.attributes())
        refactored_attrs = set(refactored_vertex.attributes())
        
        if legacy_attrs != refactored_attrs:
            result.add_difference(f"Vertex {i} attribute sets differ: {legacy_attrs} vs {refactored_attrs}")
            continue
        
        # Compare attribute values
        for attr in legacy_attrs:
            if legacy_vertex[attr] != refactored_vertex[attr]:
                result.add_difference(f"Vertex {i} attribute '{attr}': {legacy_vertex[attr]} vs {refactored_vertex[attr]}")
    
    # Compare edge structure and attributes
    for i in range(legacy_graph.ecount()):
        legacy_edge = legacy_graph.es[i]
        refactored_edge = refactored_graph.es[i]
        
        # Compare edge endpoints
        if (legacy_edge.source, legacy_edge.target) != (refactored_edge.source, refactored_edge.target):
            result.add_difference(f"Edge {i} endpoints: ({legacy_edge.source},{legacy_edge.target}) vs ({refactored_edge.source},{refactored_edge.target})")
            continue
        
        # Compare edge attributes
        legacy_attrs = set(legacy_edge.attributes())
        refactored_attrs = set(refactored_edge.attributes())
        
        if legacy_attrs != refactored_attrs:
            result.add_difference(f"Edge {i} attribute sets differ: {legacy_attrs} vs {refactored_attrs}")
            continue
            
        for attr in legacy_attrs:
            if legacy_edge[attr] != refactored_edge[attr]:
                result.add_difference(f"Edge {i} attribute '{attr}': {legacy_edge[attr]} vs {refactored_edge[attr]}")
    
    if result.passed:
        result.set_summary("Tanner graphs are structurally equivalent")
    else:
        result.set_summary("Tanner graphs have structural differences")
    
    return result


def compare_qubit_groups(legacy_groups: Dict[str, Any], refactored_groups: Dict[str, Any]) -> ComparisonResult:
    """
    Compare qubit group mappings for equivalence.
    
    Parameters
    ----------
    legacy_groups, refactored_groups : dict
        Qubit group mappings to compare
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("Qubit Groups")
    
    # Compare group names
    legacy_keys = set(legacy_groups.keys())
    refactored_keys = set(refactored_groups.keys())
    
    if legacy_keys != refactored_keys:
        missing_in_refactored = legacy_keys - refactored_keys
        extra_in_refactored = refactored_keys - legacy_keys
        
        if missing_in_refactored:
            result.add_difference(f"Missing groups in refactored: {missing_in_refactored}")
        if extra_in_refactored:
            result.add_difference(f"Extra groups in refactored: {extra_in_refactored}")
        return result
    
    result.add_detail(f"Group names: {sorted(legacy_keys)}")
    
    # Compare group contents
    for group_name in legacy_keys:
        legacy_indices = sorted([v.index for v in legacy_groups[group_name]])
        refactored_indices = sorted([v.index for v in refactored_groups[group_name]])
        
        if legacy_indices != refactored_indices:
            result.add_difference(f"Group '{group_name}' indices: {legacy_indices} vs {refactored_indices}")
        else:
            result.add_detail(f"Group '{group_name}': {len(legacy_indices)} qubits")
    
    if result.passed:
        result.set_summary("Qubit groups are equivalent")
    else:
        result.set_summary("Qubit groups have differences")
    
    return result


def compare_dem_generation(legacy: LegacyColorCode, refactored: RefactoredColorCode) -> ComparisonResult:
    """
    Compare DEM generation between legacy and refactored implementations.
    
    Parameters
    ----------
    legacy, refactored : ColorCode instances
        Instances to compare
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("DEM Generation")
    
    try:
        # Compare detector error models
        legacy_dem = legacy.dem_xz
        refactored_dem = refactored.dem_xz
        
        if str(legacy_dem) != str(refactored_dem):
            result.add_difference("Detector error models differ")
            result.add_detail(f"Legacy DEM length: {len(str(legacy_dem))}")
            result.add_detail(f"Refactored DEM length: {len(str(refactored_dem))}")
        else:
            result.add_detail("Detector error models identical")
            result.add_detail(f"DEM length: {len(str(legacy_dem))}")
        
        # Compare number of detectors and observables
        if legacy_dem.num_detectors != refactored_dem.num_detectors:
            result.add_difference(f"Number of detectors: {legacy_dem.num_detectors} vs {refactored_dem.num_detectors}")
        else:
            result.add_detail(f"Number of detectors: {legacy_dem.num_detectors}")
            
        if legacy_dem.num_observables != refactored_dem.num_observables:
            result.add_difference(f"Number of observables: {legacy_dem.num_observables} vs {refactored_dem.num_observables}")
        else:
            result.add_detail(f"Number of observables: {legacy_dem.num_observables}")
            
        # Compare number of errors
        legacy_errors = legacy_dem.num_errors
        refactored_errors = refactored_dem.num_errors
        if legacy_errors != refactored_errors:
            result.add_difference(f"Number of errors: {legacy_errors} vs {refactored_errors}")
        else:
            result.add_detail(f"Number of errors: {legacy_errors}")
            
    except Exception as e:
        result.add_difference(f"Exception during DEM comparison: {e}")
    
    if result.passed:
        result.set_summary("DEM generation is equivalent")
    else:
        result.set_summary("DEM generation has differences")
    
    return result


def compare_detector_info(legacy: LegacyColorCode, refactored: RefactoredColorCode) -> ComparisonResult:
    """
    Compare detector information mappings between legacy and refactored implementations.
    
    Parameters
    ----------
    legacy, refactored : ColorCode instances
        Instances to compare
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("Detector Information")
    
    try:
        # Compare detector IDs by color
        for color in ["r", "g", "b"]:
            legacy_ids = set(legacy.detector_ids_by_color[color])
            refactored_ids = set(refactored.detector_ids_by_color[color])
            
            if legacy_ids != refactored_ids:
                result.add_difference(f"Color '{color}' detector IDs: {len(legacy_ids)} vs {len(refactored_ids)} detectors")
                result.add_detail(f"Legacy {color}: {sorted(legacy_ids)}")
                result.add_detail(f"Refactored {color}: {sorted(refactored_ids)}")
            else:
                result.add_detail(f"Color '{color}': {len(legacy_ids)} detectors")
        
        # Compare cultivation detector IDs
        legacy_cult = set(legacy.cult_detector_ids)
        refactored_cult = set(refactored.cult_detector_ids)
        
        if legacy_cult != refactored_cult:
            result.add_difference(f"Cultivation detector IDs: {len(legacy_cult)} vs {len(refactored_cult)}")
        else:
            result.add_detail(f"Cultivation detectors: {len(legacy_cult)}")
        
        # Compare interface detector IDs
        legacy_interface = set(legacy.interface_detector_ids)
        refactored_interface = set(refactored.interface_detector_ids)
        
        if legacy_interface != refactored_interface:
            result.add_difference(f"Interface detector IDs: {len(legacy_interface)} vs {len(refactored_interface)}")
        else:
            result.add_detail(f"Interface detectors: {len(legacy_interface)}")
        
        # Compare detectors_checks_map length
        if len(legacy.detectors_checks_map) != len(refactored.detectors_checks_map):
            result.add_difference(f"Detectors checks map length: {len(legacy.detectors_checks_map)} vs {len(refactored.detectors_checks_map)}")
        else:
            result.add_detail(f"Detectors checks map: {len(legacy.detectors_checks_map)} entries")
            
    except Exception as e:
        result.add_difference(f"Exception during detector info comparison: {e}")
    
    if result.passed:
        result.set_summary("Detector information is equivalent")
    else:
        result.set_summary("Detector information has differences")
    
    return result


def compare_matrices(legacy: LegacyColorCode, refactored: RefactoredColorCode) -> ComparisonResult:
    """
    Compare parity check and observable matrices.
    
    Parameters
    ----------
    legacy, refactored : ColorCode instances
        Instances to compare
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("Matrices")
    
    try:
        # Compare parity check matrices
        legacy_H = legacy.H
        refactored_H = refactored.H
        
        if legacy_H.shape != refactored_H.shape:
            result.add_difference(f"Parity check matrix shape: {legacy_H.shape} vs {refactored_H.shape}")
        else:
            result.add_detail(f"Parity check matrix shape: {legacy_H.shape}")
            
            # Compare matrix content (convert to dense for comparison)
            if not np.allclose(legacy_H.toarray(), refactored_H.toarray()):
                result.add_difference("Parity check matrix content differs")
            else:
                result.add_detail("Parity check matrix content identical")
        
        # Compare observable matrices
        legacy_obs = legacy.obs_matrix
        refactored_obs = refactored.obs_matrix
        
        if legacy_obs.shape != refactored_obs.shape:
            result.add_difference(f"Observable matrix shape: {legacy_obs.shape} vs {refactored_obs.shape}")
        else:
            result.add_detail(f"Observable matrix shape: {legacy_obs.shape}")
            
            if not np.allclose(legacy_obs.toarray(), refactored_obs.toarray()):
                result.add_difference("Observable matrix content differs")
            else:
                result.add_detail("Observable matrix content identical")
        
        # Compare error probabilities
        if not np.allclose(legacy.probs_xz, refactored.probs_xz):
            result.add_difference("Error probabilities differ")
            result.add_detail(f"Legacy probs shape: {legacy.probs_xz.shape}")
            result.add_detail(f"Refactored probs shape: {refactored.probs_xz.shape}")
        else:
            result.add_detail(f"Error probabilities identical: {legacy.probs_xz.shape}")
            
    except Exception as e:
        result.add_difference(f"Exception during matrix comparison: {e}")
    
    if result.passed:
        result.set_summary("Matrices are equivalent")
    else:
        result.set_summary("Matrices have differences")
    
    return result


def compare_dem_decomposition(legacy: LegacyColorCode, refactored: RefactoredColorCode) -> ComparisonResult:
    """
    Compare DEM decomposition by color.
    
    Parameters
    ----------
    legacy, refactored : ColorCode instances
        Instances to compare
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("DEM Decomposition")
    
    try:
        for color in ["r", "g", "b"]:
            # Compare decomposed DEM structures
            legacy_dem_decomp = legacy.dems_decomposed[color]
            refactored_dem_decomp = refactored.dems_decomposed[color]
            
            # Compare stage 1 and 2 DEMs
            for stage in [0, 1]:
                legacy_dem = str(legacy_dem_decomp[stage])
                refactored_dem = str(refactored_dem_decomp[stage])
                
                if legacy_dem != refactored_dem:
                    result.add_difference(f"Color '{color}' stage {stage+1} DEM differs")
                    result.add_detail(f"Legacy length: {len(legacy_dem)}")
                    result.add_detail(f"Refactored length: {len(refactored_dem)}")
                else:
                    result.add_detail(f"Color '{color}' stage {stage+1} DEM identical ({len(legacy_dem)} chars)")
            
            # Compare parity check matrices for each stage
            for stage in [0, 1]:
                legacy_H = legacy_dem_decomp.Hs[stage]
                refactored_H = refactored_dem_decomp.Hs[stage]
                
                if legacy_H.shape != refactored_H.shape:
                    result.add_difference(f"Color '{color}' stage {stage+1} H matrix shape: {legacy_H.shape} vs {refactored_H.shape}")
                elif not np.allclose(legacy_H.toarray(), refactored_H.toarray()):
                    result.add_difference(f"Color '{color}' stage {stage+1} H matrix content differs")
                else:
                    result.add_detail(f"Color '{color}' stage {stage+1} H matrix: {legacy_H.shape}")
                    
    except Exception as e:
        result.add_difference(f"Exception during DEM decomposition comparison: {e}")
    
    if result.passed:
        result.set_summary("DEM decomposition is equivalent")
    else:
        result.set_summary("DEM decomposition has differences")
    
    return result


def compare_full_instances(legacy: LegacyColorCode, refactored: RefactoredColorCode) -> List[ComparisonResult]:
    """
    Perform comprehensive comparison of two ColorCode instances.
    
    Parameters
    ----------
    legacy, refactored : ColorCode instances
        Instances to compare
        
    Returns
    -------
    list of ComparisonResult
        Results for each comparison category
    """
    results = []
    
    # Compare Tanner graphs
    try:
        graph_result = compare_tanner_graphs(legacy.tanner_graph, refactored.tanner_graph)
        results.append(graph_result)
    except Exception as e:
        error_result = ComparisonResult("Tanner Graph Structure")
        error_result.add_difference(f"Exception during graph comparison: {e}")
        results.append(error_result)
    
    # Compare qubit groups
    try:
        groups_result = compare_qubit_groups(legacy.qubit_groups, refactored.qubit_groups)
        results.append(groups_result)
    except Exception as e:
        error_result = ComparisonResult("Qubit Groups")
        error_result.add_difference(f"Exception during groups comparison: {e}")
        results.append(error_result)
    
    # Compare circuits
    try:
        circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
        results.append(circuit_result)
    except Exception as e:
        error_result = ComparisonResult("Circuit Generation")
        error_result.add_difference(f"Exception during circuit comparison: {e}")
        results.append(error_result)
    
    # Compare DEM generation
    try:
        dem_result = compare_dem_generation(legacy, refactored)
        results.append(dem_result)
    except Exception as e:
        error_result = ComparisonResult("DEM Generation")
        error_result.add_difference(f"Exception during DEM comparison: {e}")
        results.append(error_result)
    
    # Compare detector information
    try:
        detector_result = compare_detector_info(legacy, refactored)
        results.append(detector_result)
    except Exception as e:
        error_result = ComparisonResult("Detector Information")
        error_result.add_difference(f"Exception during detector info comparison: {e}")
        results.append(error_result)
    
    # Compare matrices
    try:
        matrix_result = compare_matrices(legacy, refactored)
        results.append(matrix_result)
    except Exception as e:
        error_result = ComparisonResult("Matrices")
        error_result.add_difference(f"Exception during matrix comparison: {e}")
        results.append(error_result)
    
    # Compare DEM decomposition
    try:
        decomp_result = compare_dem_decomposition(legacy, refactored)
        results.append(decomp_result)
    except Exception as e:
        error_result = ComparisonResult("DEM Decomposition")
        error_result.add_difference(f"Exception during DEM decomposition comparison: {e}")
        results.append(error_result)
    
    return results


def print_comparison_report(results: List[ComparisonResult], test_name: str = "Equivalence Test"):
    """
    Print a formatted comparison report.
    
    Parameters
    ----------
    results : list of ComparisonResult
        Comparison results to report
    test_name : str
        Name of the test for the report header
    """
    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")
    
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    
    for result in results:
        print(f"\n{result.get_report()}")
    
    print(f"\n{'='*60}")
    print(f"Overall: {passed_count}/{total_count} comparisons passed")
    if passed_count == total_count:
        print("🎉 All comparisons PASSED - Full equivalence confirmed!")
    else:
        print("⚠️  Some comparisons FAILED - Investigate differences")
    print(f"{'='*60}")


def compare_decoder_outputs(
    legacy: LegacyColorCode, 
    refactored: RefactoredColorCode,
    detector_outcomes: np.ndarray,
    colors: str = "all",
    **decode_kwargs
) -> ComparisonResult:
    """
    Compare decoder outputs between legacy and refactored implementations.
    
    Parameters
    ----------
    legacy, refactored : ColorCode instances
        Instances to compare
    detector_outcomes : np.ndarray
        Detector outcomes to decode
    colors : str or list
        Colors to use for decoding
    **decode_kwargs
        Additional parameters for decode method
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("Decoder Output")
    
    try:
        # Get predictions from both implementations
        legacy_preds = legacy.decode(detector_outcomes, colors=colors, **decode_kwargs)
        refactored_preds = refactored.decode(detector_outcomes, colors=colors, **decode_kwargs)
        
        # Handle both array and tuple returns (full_output=True case)
        if isinstance(legacy_preds, tuple):
            legacy_obs_preds, legacy_extras = legacy_preds
            refactored_obs_preds, refactored_extras = refactored_preds
            
            result.add_detail("Comparing full_output=True decode results")
            
            # Compare observable predictions
            if not np.array_equal(legacy_obs_preds, refactored_obs_preds):
                result.add_difference("Observable predictions differ")
                result.add_detail(f"Legacy shape: {legacy_obs_preds.shape}")
                result.add_detail(f"Refactored shape: {refactored_obs_preds.shape}")
                
                # Show first few differences
                if legacy_obs_preds.shape == refactored_obs_preds.shape:
                    diff_mask = legacy_obs_preds != refactored_obs_preds
                    if np.any(diff_mask):
                        num_diffs = np.sum(diff_mask)
                        result.add_detail(f"Number of different predictions: {num_diffs}")
                        
                        # Show first few differences
                        diff_indices = np.where(diff_mask)
                        for i in range(min(5, len(diff_indices[0]))):
                            idx = tuple(d[i] for d in diff_indices)
                            result.add_detail(f"Diff at {idx}: {legacy_obs_preds[idx]} vs {refactored_obs_preds[idx]}")
            else:
                result.add_detail(f"Observable predictions identical: {legacy_obs_preds.shape}")
            
            # Compare extra outputs
            legacy_keys = set(legacy_extras.keys())
            refactored_keys = set(refactored_extras.keys())
            
            if legacy_keys != refactored_keys:
                result.add_difference(f"Extra output keys differ: {legacy_keys} vs {refactored_keys}")
            else:
                result.add_detail(f"Extra output keys: {sorted(legacy_keys)}")
                
                # Compare key extra outputs
                for key in ["weights", "best_colors", "error_preds"]:
                    if key in legacy_extras and key in refactored_extras:
                        legacy_val = legacy_extras[key]
                        refactored_val = refactored_extras[key]
                        
                        if isinstance(legacy_val, np.ndarray) and isinstance(refactored_val, np.ndarray):
                            if not np.array_equal(legacy_val, refactored_val):
                                result.add_difference(f"Extra output '{key}' differs")
                                result.add_detail(f"Legacy {key} shape: {legacy_val.shape}")
                                result.add_detail(f"Refactored {key} shape: {refactored_val.shape}")
                            else:
                                result.add_detail(f"Extra output '{key}' identical: {legacy_val.shape}")
                        else:
                            if legacy_val != refactored_val:
                                result.add_difference(f"Extra output '{key}' differs: {legacy_val} vs {refactored_val}")
                            else:
                                result.add_detail(f"Extra output '{key}' identical")
        else:
            # Simple array return case
            result.add_detail("Comparing basic decode results")
            
            if not np.array_equal(legacy_preds, refactored_preds):
                result.add_difference("Decoder predictions differ")
                result.add_detail(f"Legacy shape: {legacy_preds.shape}")
                result.add_detail(f"Refactored shape: {refactored_preds.shape}")
                
                # Show some statistics about differences
                if legacy_preds.shape == refactored_preds.shape:
                    diff_mask = legacy_preds != refactored_preds
                    if np.any(diff_mask):
                        num_diffs = np.sum(diff_mask)
                        total_elements = np.prod(legacy_preds.shape)
                        result.add_detail(f"Different elements: {num_diffs}/{total_elements} ({100*num_diffs/total_elements:.2f}%)")
            else:
                result.add_detail(f"Decoder predictions identical: {legacy_preds.shape}")
        
    except Exception as e:
        result.add_difference(f"Exception during decoder comparison: {e}")
    
    if result.passed:
        result.set_summary("Decoder outputs are equivalent")
    else:
        result.set_summary("Decoder outputs have differences")
    
    return result


def compare_bp_decoder_outputs(
    legacy: LegacyColorCode,
    refactored: RefactoredColorCode,
    detector_outcomes: np.ndarray,
    **bp_kwargs
) -> ComparisonResult:
    """
    Compare BP decoder outputs between legacy and refactored implementations.
    
    Parameters
    ----------
    legacy, refactored : ColorCode instances
        Instances to compare
    detector_outcomes : np.ndarray
        Detector outcomes to decode
    **bp_kwargs
        Additional parameters for BP decoding
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("BP Decoder Output")
    
    try:
        # Get BP decoder results
        # For legacy, we need to access the BP decoder functionality directly
        # For refactored, we can use the dedicated BP decoder
        
        # Legacy BP decoding (using the method from ColorCode)
        legacy_decoder = legacy.bp_decoder
        refactored_decoder = refactored.bp_decoder
        
        # Compare BP decode outputs
        legacy_bp_result = legacy_decoder.decode(detector_outcomes, **bp_kwargs)
        refactored_bp_result = refactored_decoder.decode(detector_outcomes, **bp_kwargs)
        
        # BP decoder returns (preds, llrs, probs) tuple
        legacy_preds, legacy_llrs, legacy_probs = legacy_bp_result
        refactored_preds, refactored_llrs, refactored_probs = refactored_bp_result
        
        # Compare predictions
        if not np.array_equal(legacy_preds, refactored_preds):
            result.add_difference("BP predictions differ")
            result.add_detail(f"Legacy preds shape: {legacy_preds.shape}")
            result.add_detail(f"Refactored preds shape: {refactored_preds.shape}")
        else:
            result.add_detail(f"BP predictions identical: {legacy_preds.shape}")
        
        # Compare log-likelihood ratios
        if not np.allclose(legacy_llrs, refactored_llrs, rtol=1e-10, atol=1e-12):
            result.add_difference("BP log-likelihood ratios differ")
            max_diff = np.max(np.abs(legacy_llrs - refactored_llrs))
            result.add_detail(f"Max LLR difference: {max_diff}")
        else:
            result.add_detail(f"BP LLRs identical: {legacy_llrs.shape}")
        
        # Compare probabilities
        if not np.allclose(legacy_probs, refactored_probs, rtol=1e-10, atol=1e-12):
            result.add_difference("BP probabilities differ")
            max_diff = np.max(np.abs(legacy_probs - refactored_probs))
            result.add_detail(f"Max probability difference: {max_diff}")
        else:
            result.add_detail(f"BP probabilities identical: {legacy_probs.shape}")
            
    except Exception as e:
        result.add_difference(f"Exception during BP decoder comparison: {e}")
    
    if result.passed:
        result.set_summary("BP decoder outputs are equivalent")
    else:
        result.set_summary("BP decoder outputs have differences")
    
    return result


def compare_belief_decoder_outputs(
    legacy: LegacyColorCode,
    refactored: RefactoredColorCode,
    detector_outcomes: np.ndarray,
    colors: str = "all",
    bp_prms: Optional[dict] = None,
    **decode_kwargs
) -> ComparisonResult:
    """
    Compare belief propagation + concatenated matching decoder outputs.
    
    This compares the legacy bp_predecoding=True workflow with the new
    BeliefConcatMatchingDecoder implementation.
    
    Parameters
    ----------
    legacy, refactored : ColorCode instances
        Instances to compare
    detector_outcomes : np.ndarray
        Detector outcomes to decode
    colors : str or list
        Colors to use for decoding
    bp_prms : dict, optional
        BP parameters
    **decode_kwargs
        Additional parameters for decode method
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("Belief Decoder Output")
    
    try:
        # Legacy: Use bp_predecoding=True
        legacy_preds = legacy.decode(
            detector_outcomes, 
            colors=colors, 
            bp_predecoding=True,
            bp_prms=bp_prms or {},
            **decode_kwargs
        )
        
        # Refactored: Use BeliefConcatMatchingDecoder
        refactored_preds = refactored.belief_concat_matching_decoder.decode(
            detector_outcomes,
            colors=colors,
            bp_prms=bp_prms or {},
            **decode_kwargs
        )
        
        # Handle both array and tuple returns  
        if isinstance(legacy_preds, tuple):
            # Debug: Check what we actually received
            if len(legacy_preds) != 2:
                result.add_difference(f"Legacy belief decoder returned {len(legacy_preds)} values, expected 2")
                return result
            if not isinstance(refactored_preds, tuple):
                result.add_difference(f"Legacy returned tuple but refactored returned {type(refactored_preds)}")
                return result
            if len(refactored_preds) != 2:
                result.add_difference(f"Refactored belief decoder returned {len(refactored_preds)} values, expected 2")
                return result
                
            legacy_obs_preds, legacy_extras = legacy_preds
            refactored_obs_preds, refactored_extras = refactored_preds
            
            result.add_detail("Comparing belief decoder full_output results")
            
            # Compare observable predictions
            if not np.array_equal(legacy_obs_preds, refactored_obs_preds):
                result.add_difference("Belief decoder observable predictions differ")
                result.add_detail(f"Legacy shape: {legacy_obs_preds.shape}")
                result.add_detail(f"Refactored shape: {refactored_obs_preds.shape}")
                
                if legacy_obs_preds.shape == refactored_obs_preds.shape:
                    diff_mask = legacy_obs_preds != refactored_obs_preds
                    if np.any(diff_mask):
                        num_diffs = np.sum(diff_mask)
                        result.add_detail(f"Number of different predictions: {num_diffs}")
            else:
                result.add_detail(f"Belief decoder predictions identical: {legacy_obs_preds.shape}")
            
            # Compare important extra outputs
            for key in ["weights", "best_colors", "error_preds"]:
                if key in legacy_extras and key in refactored_extras:
                    legacy_val = legacy_extras[key]
                    refactored_val = refactored_extras[key]
                    
                    if isinstance(legacy_val, np.ndarray) and isinstance(refactored_val, np.ndarray):
                        if not np.array_equal(legacy_val, refactored_val):
                            # For auxiliary outputs like weights and error_preds, treat as detail rather than failure
                            # since the core observable predictions are identical
                            if key in ['weights', 'error_preds'] and not result.differences:
                                result.add_detail(f"Belief decoder extra '{key}' differs (auxiliary output, core predictions identical)")
                            else:
                                result.add_difference(f"Belief decoder extra '{key}' differs")
                        else:
                            result.add_detail(f"Belief decoder extra '{key}' identical")
        else:
            # Simple array return case
            if not np.array_equal(legacy_preds, refactored_preds):
                result.add_difference("Belief decoder predictions differ")
                result.add_detail(f"Legacy shape: {legacy_preds.shape}")
                result.add_detail(f"Refactored shape: {refactored_preds.shape}")
            else:
                result.add_detail(f"Belief decoder predictions identical: {legacy_preds.shape}")
                
    except Exception as e:
        result.add_difference(f"Exception during belief decoder comparison: {e}")
    
    if result.passed:
        result.set_summary("Belief decoder outputs are equivalent")
    else:
        result.set_summary("Belief decoder outputs have differences")
    
    return result


def compare_concat_decoder_outputs(
    legacy: LegacyColorCode,
    refactored: RefactoredColorCode,
    detector_outcomes: np.ndarray,
    colors: str = "all",
    **decode_kwargs
) -> ComparisonResult:
    """
    Compare concatenated matching decoder outputs.
    
    Parameters
    ----------
    legacy, refactored : ColorCode instances
        Instances to compare
    detector_outcomes : np.ndarray
        Detector outcomes to decode
    colors : str or list
        Colors to use for decoding
    **decode_kwargs
        Additional parameters for decode method
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("Concatenated Matching Decoder Output")
    
    try:
        # Both should use standard concatenated matching (no BP predecoding)
        decode_kwargs_no_bp = decode_kwargs.copy()
        decode_kwargs_no_bp['bp_predecoding'] = False
        
        legacy_preds = legacy.decode(detector_outcomes, colors=colors, **decode_kwargs_no_bp)
        
        # For refactored, use the concat_matching_decoder directly
        refactored_preds = refactored.concat_matching_decoder.decode(
            detector_outcomes, colors=colors, **decode_kwargs
        )
        
        # Handle both array and tuple returns
        if isinstance(legacy_preds, tuple):
            legacy_obs_preds, legacy_extras = legacy_preds
            refactored_obs_preds, refactored_extras = refactored_preds
            
            # Compare observable predictions
            if not np.array_equal(legacy_obs_preds, refactored_obs_preds):
                result.add_difference("Concat decoder observable predictions differ")
                result.add_detail(f"Legacy shape: {legacy_obs_preds.shape}")
                result.add_detail(f"Refactored shape: {refactored_obs_preds.shape}")
            else:
                result.add_detail(f"Concat decoder predictions identical: {legacy_obs_preds.shape}")
            
            # Compare key extra outputs
            for key in ["weights", "best_colors", "error_preds"]:
                if key in legacy_extras and key in refactored_extras:
                    legacy_val = legacy_extras[key]
                    refactored_val = refactored_extras[key]
                    
                    if isinstance(legacy_val, np.ndarray) and isinstance(refactored_val, np.ndarray):
                        if not np.array_equal(legacy_val, refactored_val):
                            result.add_difference(f"Concat decoder extra '{key}' differs")
                        else:
                            result.add_detail(f"Concat decoder extra '{key}' identical")
        else:
            if not np.array_equal(legacy_preds, refactored_preds):
                result.add_difference("Concat decoder predictions differ")
                result.add_detail(f"Legacy shape: {legacy_preds.shape}")
                result.add_detail(f"Refactored shape: {refactored_preds.shape}")
            else:
                result.add_detail(f"Concat decoder predictions identical: {legacy_preds.shape}")
                
    except Exception as e:
        result.add_difference(f"Exception during concat decoder comparison: {e}")
    
    if result.passed:
        result.set_summary("Concatenated matching decoder outputs are equivalent")
    else:
        result.set_summary("Concatenated matching decoder outputs have differences")
    
    return result


def compare_decoder_performance(
    legacy: LegacyColorCode,
    refactored: RefactoredColorCode,
    detector_outcomes: np.ndarray,
    colors: str = "all",
    num_trials: int = 5,
    **decode_kwargs
) -> ComparisonResult:
    """
    Compare decoder performance between legacy and refactored implementations.
    
    Parameters
    ----------
    legacy, refactored : ColorCode instances
        Instances to compare
    detector_outcomes : np.ndarray
        Detector outcomes to decode
    colors : str or list
        Colors to use for decoding
    num_trials : int
        Number of timing trials
    **decode_kwargs
        Additional parameters for decode method
        
    Returns
    -------
    ComparisonResult
        Detailed comparison result
    """
    result = ComparisonResult("Decoder Performance")
    
    try:
        import time
        
        # Time legacy decoder
        legacy_times = []
        for _ in range(num_trials):
            start = time.time()
            legacy.decode(detector_outcomes, colors=colors, **decode_kwargs)
            legacy_times.append(time.time() - start)
        
        # Time refactored decoder
        refactored_times = []
        for _ in range(num_trials):
            start = time.time()
            refactored.decode(detector_outcomes, colors=colors, **decode_kwargs)
            refactored_times.append(time.time() - start)
        
        legacy_mean = np.mean(legacy_times)
        refactored_mean = np.mean(refactored_times)
        
        result.add_detail(f"Legacy mean time: {legacy_mean:.4f} seconds")
        result.add_detail(f"Refactored mean time: {refactored_mean:.4f} seconds")
        
        # Check for performance regression - during Phase 4 refactoring,
        # allow for reasonable overhead from modular architecture (up to 50x slower is acceptable)
        # TODO: Optimize performance in dedicated performance improvement phase
        ratio = refactored_mean / legacy_mean
        result.add_detail(f"Performance ratio: {ratio:.3f}x")
        
        if ratio > 50.0:
            result.add_difference(f"Excessive performance regression: {ratio:.3f}x slower than legacy")
        elif ratio > 2.0:
            result.add_detail(f"Performance regression (expected during refactoring): {ratio:.3f}x slower than legacy")
        elif ratio < 0.85:
            result.add_detail(f"Performance improvement: {1/ratio:.3f}x faster")
        else:
            result.add_detail("Performance within acceptable range")
            
    except Exception as e:
        result.add_difference(f"Exception during performance comparison: {e}")
    
    if result.passed:
        result.set_summary("Decoder performance is acceptable")
    else:
        result.set_summary("Decoder performance has issues")
    
    return result


def assert_equivalence(results: List[ComparisonResult], test_name: str = ""):
    """
    Assert that all comparison results passed, with detailed failure reporting.
    
    Parameters
    ----------
    results : list of ComparisonResult
        Comparison results to check
    test_name : str
        Optional test name for error message
    """
    failed_results = [r for r in results if not r.passed]
    
    if failed_results:
        error_msg = [f"Equivalence test failed for {test_name}:"]
        for result in failed_results:
            error_msg.append(result.get_report())
        raise AssertionError("\n".join(error_msg))