#!/usr/bin/env python3
"""
Compare circuit generation results between main and dev branches.

This script analyzes the test results from both branches and generates
a comprehensive equivalence verification report.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


class EquivalenceAnalyzer:
    """Analyzer for comparing circuit generation results between branches."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.main_results = self._load_results("main")
        self.dev_results = self._load_results("dev")
        
    def _load_results(self, branch: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load test results for a specific branch."""
        results_file = self.results_dir / f"results_{branch}.json"
        if not results_file.exists():
            print(f"Warning: No results file found for {branch} branch at {results_file}")
            return {}
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def analyze_equivalence(self) -> Dict[str, Any]:
        """Perform comprehensive equivalence analysis."""
        analysis = {
            "summary": {},
            "circuit_types": {},
            "individual_tests": {},
            "performance": {},
            "errors": {"main": {}, "dev": {}}
        }
        
        # Overall summary
        analysis["summary"] = self._analyze_summary()
        
        # Circuit type breakdown
        for circuit_type in ["triangular_circuits", "rectangular_circuits", "stability_circuits", "growing_circuits", "edge_cases"]:
            if circuit_type in self.main_results or circuit_type in self.dev_results:
                analysis["circuit_types"][circuit_type] = self._analyze_circuit_type(circuit_type)
        
        # Individual test comparisons
        analysis["individual_tests"] = self._analyze_individual_tests()
        
        # Performance analysis
        analysis["performance"] = self._analyze_performance()
        
        # Error analysis
        analysis["errors"] = self._analyze_errors()
        
        return analysis
    
    def _analyze_summary(self) -> Dict[str, Any]:
        """Generate overall summary statistics."""
        main_total = sum(len(tests) for tests in self.main_results.values())
        dev_total = sum(len(tests) for tests in self.dev_results.values())
        
        main_successful = sum(
            len([t for t in tests if "error" not in t])
            for tests in self.main_results.values()
        )
        dev_successful = sum(
            len([t for t in tests if "error" not in t])
            for tests in self.dev_results.values()
        )
        
        return {
            "main_total_tests": main_total,
            "dev_total_tests": dev_total,
            "main_successful": main_successful,
            "dev_successful": dev_successful,
            "main_success_rate": (main_successful / main_total * 100) if main_total > 0 else 0,
            "dev_success_rate": (dev_successful / dev_total * 100) if dev_total > 0 else 0,
            "tests_match": main_total == dev_total,
            "success_rates_match": main_successful == dev_successful
        }
    
    def _analyze_circuit_type(self, circuit_type: str) -> Dict[str, Any]:
        """Analyze a specific circuit type."""
        main_tests = self.main_results.get(circuit_type, [])
        dev_tests = self.dev_results.get(circuit_type, [])
        
        # Create lookup by test name
        main_by_name = {t.get("metadata", {}).get("test_name", f"test_{i}"): t for i, t in enumerate(main_tests)}
        dev_by_name = {t.get("metadata", {}).get("test_name", f"test_{i}"): t for i, t in enumerate(dev_tests)}
        
        all_test_names = set(main_by_name.keys()) | set(dev_by_name.keys())
        
        matches = 0
        differences = []
        missing_in_main = []
        missing_in_dev = []
        
        for test_name in all_test_names:
            main_test = main_by_name.get(test_name)
            dev_test = dev_by_name.get(test_name)
            
            if main_test is None:
                missing_in_main.append(test_name)
            elif dev_test is None:
                missing_in_dev.append(test_name)
            else:
                # Compare the tests
                if self._tests_match(main_test, dev_test):
                    matches += 1
                else:
                    differences.append({
                        "test_name": test_name,
                        "difference": self._get_test_difference(main_test, dev_test)
                    })
        
        return {
            "total_tests": len(all_test_names),
            "matches": matches,
            "differences": len(differences),
            "missing_in_main": missing_in_main,
            "missing_in_dev": missing_in_dev,
            "match_rate": (matches / len(all_test_names) * 100) if all_test_names else 100,
            "difference_details": differences[:10],  # Limit to first 10 for readability
        }
    
    def _tests_match(self, main_test: Dict[str, Any], dev_test: Dict[str, Any]) -> bool:
        """Check if two test results match."""
        # Both have errors
        if "error" in main_test and "error" in dev_test:
            return True  # Both failed, consider as match
        
        # One has error, other doesn't
        if "error" in main_test or "error" in dev_test:
            return False
        
        # Compare circuit strings (most comprehensive check)
        main_circuit = main_test.get("circuit_string", "")
        dev_circuit = dev_test.get("circuit_string", "")
        
        return main_circuit == dev_circuit
    
    def _get_test_difference(self, main_test: Dict[str, Any], dev_test: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed difference between two tests."""
        difference = {}
        
        # Error status
        main_has_error = "error" in main_test
        dev_has_error = "error" in dev_test
        
        if main_has_error != dev_has_error:
            difference["error_mismatch"] = {
                "main_has_error": main_has_error,
                "dev_has_error": dev_has_error,
                "main_error": main_test.get("error") if main_has_error else None,
                "dev_error": dev_test.get("error") if dev_has_error else None
            }
        
        # Circuit properties (if both successful)
        if not main_has_error and not dev_has_error:
            properties = ["num_qubits", "num_instructions", "num_detectors", "num_observables"]
            for prop in properties:
                main_val = main_test.get(prop)
                dev_val = dev_test.get(prop)
                if main_val != dev_val:
                    difference[f"{prop}_mismatch"] = {"main": main_val, "dev": dev_val}
            
            # Circuit string comparison
            main_circuit = main_test.get("circuit_string", "")
            dev_circuit = dev_test.get("circuit_string", "")
            if main_circuit != dev_circuit:
                difference["circuit_string_differs"] = True
                # Find first differing line
                main_lines = main_circuit.split('\n')
                dev_lines = dev_circuit.split('\n')
                for i, (main_line, dev_line) in enumerate(zip(main_lines, dev_lines)):
                    if main_line != dev_line:
                        difference["first_diff_line"] = {
                            "line_number": i,
                            "main": main_line,
                            "dev": dev_line
                        }
                        break
        
        return difference
    
    def _analyze_individual_tests(self) -> Dict[str, Any]:
        """Analyze individual test comparisons across all circuit types."""
        all_comparisons = []
        
        for circuit_type in set(self.main_results.keys()) | set(self.dev_results.keys()):
            main_tests = self.main_results.get(circuit_type, [])
            dev_tests = self.dev_results.get(circuit_type, [])
            
            # Create lookup by test name
            main_by_name = {t.get("metadata", {}).get("test_name", f"test_{i}"): t for i, t in enumerate(main_tests)}
            dev_by_name = {t.get("metadata", {}).get("test_name", f"test_{i}"): t for i, t in enumerate(dev_tests)}
            
            all_test_names = set(main_by_name.keys()) | set(dev_by_name.keys())
            
            for test_name in all_test_names:
                main_test = main_by_name.get(test_name)
                dev_test = dev_by_name.get(test_name)
                
                comparison = {
                    "test_name": test_name,
                    "circuit_type": circuit_type,
                    "match": self._tests_match(main_test, dev_test) if main_test and dev_test else False,
                    "main_exists": main_test is not None,
                    "dev_exists": dev_test is not None
                }
                
                if main_test and dev_test:
                    comparison["difference"] = self._get_test_difference(main_test, dev_test)
                
                all_comparisons.append(comparison)
        
        total_comparisons = len(all_comparisons)
        matches = len([c for c in all_comparisons if c["match"]])
        
        return {
            "total_comparisons": total_comparisons,
            "matches": matches,
            "match_rate": (matches / total_comparisons * 100) if total_comparisons > 0 else 100,
            "mismatches": [c for c in all_comparisons if not c["match"]][:20]  # First 20 mismatches
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance differences between branches."""
        main_times = []
        dev_times = []
        
        for circuit_type in self.main_results:
            if circuit_type in self.dev_results:
                main_tests = self.main_results[circuit_type]
                dev_tests = self.dev_results[circuit_type]
                
                # Collect generation times for successful tests
                for main_test in main_tests:
                    if "error" not in main_test:
                        time = main_test.get("metadata", {}).get("generation_time")
                        if time is not None:
                            main_times.append(time)
                
                for dev_test in dev_tests:
                    if "error" not in dev_test:
                        time = dev_test.get("metadata", {}).get("generation_time")
                        if time is not None:
                            dev_times.append(time)
        
        if main_times and dev_times:
            main_avg = sum(main_times) / len(main_times)
            dev_avg = sum(dev_times) / len(dev_times)
            performance_ratio = (dev_avg / main_avg) * 100 if main_avg > 0 else 100
            
            return {
                "main_avg_time": main_avg,
                "dev_avg_time": dev_avg,
                "main_total_time": sum(main_times),
                "dev_total_time": sum(dev_times),
                "performance_ratio_percent": performance_ratio,
                "performance_acceptable": performance_ratio <= 105  # Within 5% regression
            }
        else:
            return {"insufficient_data": True}
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns in both branches."""
        main_errors = defaultdict(list)
        dev_errors = defaultdict(list)
        
        for circuit_type, tests in self.main_results.items():
            for test in tests:
                if "error" in test:
                    error_msg = test["error"]
                    main_errors[error_msg].append({
                        "circuit_type": circuit_type,
                        "test_name": test.get("metadata", {}).get("test_name", "unknown")
                    })
        
        for circuit_type, tests in self.dev_results.items():
            for test in tests:
                if "error" in test:
                    error_msg = test["error"]
                    dev_errors[error_msg].append({
                        "circuit_type": circuit_type,
                        "test_name": test.get("metadata", {}).get("test_name", "unknown")
                    })
        
        return {
            "main": dict(main_errors),
            "dev": dict(dev_errors),
            "main_error_count": sum(len(tests) for tests in main_errors.values()),
            "dev_error_count": sum(len(tests) for tests in dev_errors.values()),
            "common_errors": list(set(main_errors.keys()) & set(dev_errors.keys()))
        }
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable equivalence report."""
        report = []
        report.append("=" * 80)
        report.append("CIRCUIT EQUIVALENCE VERIFICATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        summary = analysis["summary"]
        report.append("📊 SUMMARY")
        report.append("-" * 40)
        report.append(f"Main branch: {summary['main_successful']}/{summary['main_total_tests']} tests passed ({summary['main_success_rate']:.1f}%)")
        report.append(f"Dev branch:  {summary['dev_successful']}/{summary['dev_total_tests']} tests passed ({summary['dev_success_rate']:.1f}%)")
        report.append(f"Test counts match: {summary['tests_match']}")
        report.append(f"Success rates match: {summary['success_rates_match']}")
        report.append("")
        
        # Individual test analysis
        individual = analysis["individual_tests"]
        if "total_comparisons" in individual:
            report.append("🔍 CIRCUIT EQUIVALENCE")
            report.append("-" * 40)
            report.append(f"Total comparisons: {individual['total_comparisons']}")
            report.append(f"Exact matches: {individual['matches']}")
            report.append(f"Equivalence rate: {individual['match_rate']:.1f}%")
            
            if individual['match_rate'] == 100:
                report.append("✅ PERFECT EQUIVALENCE ACHIEVED!")
            else:
                report.append(f"⚠️  Found {len(individual.get('mismatches', []))} mismatches")
            report.append("")
        
        # Circuit type breakdown
        report.append("📈 CIRCUIT TYPE ANALYSIS")
        report.append("-" * 40)
        for circuit_type, analysis_data in analysis["circuit_types"].items():
            if analysis_data["total_tests"] > 0:
                status = "✅" if analysis_data["match_rate"] == 100 else "⚠️"
                report.append(f"{status} {circuit_type}: {analysis_data['matches']}/{analysis_data['total_tests']} ({analysis_data['match_rate']:.1f}%)")
        report.append("")
        
        # Performance analysis
        perf = analysis["performance"]
        if "performance_ratio_percent" in perf:
            report.append("⏱️  PERFORMANCE ANALYSIS")
            report.append("-" * 40)
            report.append(f"Main avg time: {perf['main_avg_time']:.4f}s")
            report.append(f"Dev avg time:  {perf['dev_avg_time']:.4f}s")
            report.append(f"Performance ratio: {perf['performance_ratio_percent']:.1f}%")
            
            if perf["performance_acceptable"]:
                report.append("✅ Performance regression within acceptable limits (<5%)")
            else:
                report.append("⚠️  Performance regression exceeds 5%")
            report.append("")
        
        # Error analysis
        errors = analysis["errors"]
        if errors["main_error_count"] > 0 or errors["dev_error_count"] > 0:
            report.append("❌ ERROR ANALYSIS")
            report.append("-" * 40)
            report.append(f"Main branch errors: {errors['main_error_count']}")
            report.append(f"Dev branch errors:  {errors['dev_error_count']}")
            
            if errors["common_errors"]:
                report.append(f"Common errors: {len(errors['common_errors'])}")
                for error in errors["common_errors"][:3]:  # Show first 3
                    report.append(f"  • {error[:100]}...")
            report.append("")
        
        # Final verdict
        report.append("🎯 FINAL VERDICT")
        report.append("-" * 40)
        
        if "individual_tests" in analysis and analysis["individual_tests"].get("match_rate", 0) == 100:
            report.append("🎉 VERIFICATION COMPLETE: 100% CIRCUIT EQUIVALENCE ACHIEVED!")
            report.append("✅ The refactored CircuitBuilder generates circuits identical to the original implementation.")
        else:
            match_rate = analysis.get("individual_tests", {}).get("match_rate", 0)
            report.append(f"⚠️  VERIFICATION INCOMPLETE: {match_rate:.1f}% equivalence achieved.")
            report.append("❌ Some circuits differ between original and refactored implementations.")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main execution function."""
    # Find results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / "fixtures"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        sys.exit(1)
    
    # Check for results files
    main_file = results_dir / "results_main.json"
    dev_file = results_dir / "results_dev.json"
    
    if not main_file.exists():
        print(f"Error: Main branch results not found at {main_file}")
        print("Run 'tox -e main' first to generate main branch results.")
        sys.exit(1)
    
    if not dev_file.exists():
        print(f"Error: Dev branch results not found at {dev_file}")
        print("Run 'tox -e dev' first to generate dev branch results.")
        sys.exit(1)
    
    print("🔍 Analyzing circuit equivalence between main and dev branches...")
    
    # Perform analysis
    analyzer = EquivalenceAnalyzer(results_dir)
    analysis = analyzer.analyze_equivalence()
    
    # Generate and display report
    report = analyzer.generate_report(analysis)
    print(report)
    
    # Save detailed analysis to file
    analysis_file = results_dir / "equivalence_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n📄 Detailed analysis saved to: {analysis_file}")
    
    # Exit with appropriate code
    match_rate = analysis.get("individual_tests", {}).get("match_rate", 0)
    sys.exit(0 if match_rate == 100 else 1)


if __name__ == "__main__":
    main()