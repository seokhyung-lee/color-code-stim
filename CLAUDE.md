# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for simulating and decoding 2D color code circuits, implementing the Concatenated Minimum-Weight Perfect Matching (MWPM) Decoder. The package is used for quantum error correction research, particularly for the paper "Color code decoder with improved scaling for correcting circuit-level noise".

## Development Commands

### Installation
```bash
# Install in development mode
pip install -e .

# Install from git (for users)
pip install git+https://github.com/seokhyung-lee/color-code-stim.git
```

### Code Formatting and Linting
```bash
# Format code with black (line length 88)
black src/

# Sort imports with isort
isort src/
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_specific.py::test_function_name

# Run equivalence tests between legacy and refactored implementations
pytest tests/test_color_code_refactor/phase1_tests/ -v    # Circuit generation equivalence
pytest tests/test_color_code_refactor/phase2_tests/ -v    # Tanner graph equivalence
pytest tests/test_color_code_refactor/integration_tests/ -v # Integration and performance tests

# Run quick validation tests
pytest tests/test_color_code_refactor/phase1_tests/test_circuit_equivalence.py::TestPhase1CircuitEquivalence::test_quick_circuit_equivalence -v
pytest tests/test_color_code_refactor/phase2_tests/test_tanner_graph_equivalence.py::TestPhase2TannerGraphEquivalence::test_quick_graph_equivalence -v
pytest tests/test_color_code_refactor/phase3_tests/test_dem_equivalence.py::TestPhase3DEMEquivalence::test_quick_dem_equivalence -v
```

## Code Architecture

### Core Components

1. **ColorCode Class** (`src/color_code_stim/color_code.py`): Main interface for creating color code circuits and running simulations. Handles:
   - High-level circuit configuration and parameter management
   - Detector error model (DEM) generation and decomposition
   - Integration with the concatenated MWPM decoder
   - Monte Carlo simulation framework

2. **CircuitBuilder Class** (`src/color_code_stim/circuit_builder.py`): Modular circuit generation engine that handles:
   - Circuit generation for different topologies (triangular, rectangular, stability experiments, growing operations)
   - CNOT scheduling and syndrome extraction circuits
   - Detector and observable placement
   - Integration with cultivation circuits

3. **TannerGraphBuilder Class** (`src/color_code_stim/graph_builder.py`): Modular Tanner graph construction that handles:
   - Graph structure creation for different patch types (triangular, rectangular)
   - Vertex and edge attribute management (coordinates, colors, types)
   - Qubit group organization and coordinate mapping
   - Lattice and Tanner edge construction

4. **Configuration Module** (`src/color_code_stim/config.py`): Centralized configuration and constants:
   - CNOT schedules for different circuit topologies
   - Type definitions and color/pauli mappings
   - Helper functions for coordinate and color conversions

5. **Circuit Types**: The package supports multiple circuit configurations:
   - `"tri"`: Memory experiments on triangular patches
   - `"rec"`: Memory experiments on rectangular patches  
   - `"rec_stability"`: Stability experiments on rectangle-like patches
   - `"growing"`: Growing operations from one distance to another
   - `"cult+growing"`: Cultivation followed by growing operations

6. **DEM Decomposition** (`src/color_code_stim/dem_decomp.py`): Handles decomposition of detector error models by color for the concatenated decoder.

7. **Cultivation Support** (`src/color_code_stim/cultivation.py`): Implements cultivation circuits based on Gidney, Shutty, and Jones' work, with pre-computed circuits stored in `src/color_code_stim/assets/cultivation_circuits/`.

8. **Utilities**: 
   - `stim_utils.py`: Helper functions for working with Stim circuits and error models
   - `visualization.py`: Plotting functions for lattices and Tanner graphs
   - `utils.py`: General utility functions for file I/O and performance calculations

### Key Dependencies

- **stim**: Circuit simulation and error model generation
- **pymatching**: MWPM decoder implementation
- **python-igraph**: Graph operations for Tanner graphs
- **numpy/matplotlib**: Numerical operations and visualization
- **statsmodels**: Statistical analysis for confidence intervals

### Important Implementation Details

1. **Concatenated MWPM Decoder**: The decoder runs 6 MWPM decoders (2 per color) and combines results. When `comparative_decoding=True`, it evaluates all logical classes to find the minimum-weight correction.

2. **Error Models**: Supports various noise parameters:
   - `p_bitflip`: Bit-flip noise rate
   - `p_reset`: Reset operation error rate
   - `p_meas`: Measurement error rate
   - `p_cnot`: CNOT gate error rate
   - `p_idle`: Idle qubit error rate
   - `p_circuit`: Uniform circuit-level noise (overrides individual rates)

3. **Logical Gap**: When using comparative decoding, the logical gap quantifies correction reliability for post-selection in magic state distillation.

### Entry Points

The main entry point is the `ColorCode` class constructor in `color_code.py`. Example usage can be found in `getting_started.ipynb`.

### Testing Structure

The project uses a comprehensive equivalence testing approach to validate the ongoing modular refactoring:

9. **Legacy Reference** (`src/color_code_stim/color_code_legacy.py`): Copy of the original monolithic implementation used as ground truth for equivalence testing.

10. **Modular Test Suite** (`tests/test_color_code_refactor/`): Phase-based equivalence testing structure:
    - **Phase 1 Tests**: Circuit generation equivalence (CircuitBuilder vs legacy)
    - **Phase 2 Tests**: Tanner graph construction equivalence (TannerGraphBuilder vs legacy)
    - **Integration Tests**: End-to-end equivalence and performance regression testing
    - **Test Utilities**: Comprehensive comparison functions and test parameter generation
    - **Future Phases**: Ready scaffolding for DEM, decoder, and simulation testing

The test suite ensures 100% functional equivalence between the refactored modular implementation and the original monolithic codebase while enabling systematic validation of each migration phase.