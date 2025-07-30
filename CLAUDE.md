# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for simulating and decoding 2D color code circuits, developed for quantum error correction research. The package implements the Concatenated Minimum-Weight Perfect Matching (MWPM) decoder for color codes and supports various circuit types for quantum memory and growing operations.

## Development Commands

### Installation and Setup
```bash
# Install from source (development mode)
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Alternative: Install from GitHub
pip install git+https://github.com/seokhyung-lee/color-code-stim.git
```

## Architecture Overview

### Core Components

**ColorCode Class** (`src/color_code_stim/color_code.py`)
- Main interface for creating and working with color code circuits
- Handles circuit generation, simulation, and decoding
- Supports multiple circuit types: "tri", "rec", "rec_stability", "growing", "cult+growing", "sdqc_memory"
- Manages tanner graphs, detector error models, and quantum circuits

**Modular Architecture Components:**
- **CircuitBuilder** (`circuit_builder.py`): Generates quantum circuits for different color code configurations
- **NoiseModel** (`noise_model.py`): Encapsulates noise parameters with dictionary-like access and validation
- **DemManager** (`dem_utils/dem_manager.py`): Manages detector error models with caching and decomposition
- **Simulator** (`simulation/simulator.py`): Handles Monte Carlo simulations and sampling
- **Decoders** (`decoders/`): Modular decoder implementations with base class architecture

### Decoder Architecture

**Base Decoder System** (`src/color_code_stim/decoders/`)
- `BaseDecoder`: Abstract base class defining decoder interface
- `ConcatMatchingDecoder`: Main concatenated MWPM decoder implementation
- `BPDecoder`: Belief propagation decoder
- `BeliefConcatMatchingDecoder`: Belief-based concatenated decoder

### Graph and Circuit Generation

**TannerGraphBuilder** (`graph_builder.py`)
- Constructs tanner graphs for color codes using igraph
- Handles vertex and edge creation for different lattice geometries

**Circuit Types:**
- `tri`: Triangular patch memory experiment (distance d, odd)
- `rec`: Rectangular patch memory experiment (distances d and d2, both even)  
- `rec_stability`: Rectangle-like stability experiment
- `growing`: Growing operation from distance d to d2 (both odd)
- `cult+growing`: Cultivation followed by growing (uses circuits from arXiv:2409.17595)
- `sdqc_memory`: SDQC triangular patch memory with superdense syndrome extraction and shuttling operations

### Key Data Structures

**Tanner Graph** (igraph.Graph)
- Vertices represent qubits and stabilizer checks
- Attributes: 
  - Ancilla positions: (x,y) coordinates for Z-type at (face_x-1, face_y), X-type at (face_x+1, face_y)
  - Face centers: (face_x, face_y) coordinates for stabilizer face centers and connectivity
  - Additional: qubit ID, Pauli type, color, boundary info
- Used for visualizing lattice structure and decoder operations

**Detector Error Models** (stim.DetectorErrorModel)
- Managed by DemManager with decomposition for concatenated decoding
- Supports caching and equivalence testing for performance
- Decomposed into stage-1 (restricted) and stage-2 (monochromatic) DEMs

## Key Features and Usage Patterns

### Basic Usage Pattern
```python
from color_code_stim import ColorCode, NoiseModel

# Create color code instance with NoiseModel (recommended)
noise = NoiseModel.uniform_circuit_noise(1e-3)  # Circuit-level noise
colorcode = ColorCode(
    d=5,                    # Code distance
    rounds=5,              # Syndrome extraction rounds
    circuit_type="tri",    # Circuit type
    noise_model=noise,     # Unified noise configuration
    cnot_schedule="tri_optimal"
)

# Alternative: individual noise parameters (backward compatibility)
colorcode = ColorCode(
    d=5,                    # Code distance
    rounds=5,              # Syndrome extraction rounds
    circuit_type="tri",    # Circuit type
    p_circuit=1e-3,        # Circuit-level noise
    cnot_schedule="tri_optimal"
)

# Run simulation
num_fails, info = colorcode.simulate(shots=10000, full_output=True)
```

### Comparative Decoding
- Set `comparative_decoding=True` to run decoder multiple times across logical classes
- Returns logical gap values for post-selection
- Useful for magic state distillation applications

### Error Analysis and Visualization
- Supports error visualization on lattice (bit-flip noise only)
- Methods: `draw_tanner_graph()`, `draw_lattice()`, `sample_with_errors()`
- Highlight error qubits, corrections, and violated stabilizers

## Important Dependencies

- **stim**: Quantum circuit simulation and error model generation
- **pymatching**: Minimum-weight perfect matching for decoding
- **igraph**: Graph data structures for tanner graphs
- **numpy/scipy**: Numerical computations and sparse matrices
- **matplotlib**: Visualization and plotting

## Testing

### Comprehensive Circuit Generation and Simulation Tests
The repository includes comprehensive test coverage in `tests/test_circuit_generation_simulation.py`:

- **All Circuit Types**: Tests all supported circuit types ("tri", "rec", "rec_stability", "growing", "cult+growing", "sdqc_memory")
- **Parameter Combinations**: Parametrized testing across all major configuration options.
- **Validation Focus**: Ensures no exceptions are raised during circuit generation and simulation
- **Implementation Status**: Properly handles unimplemented features (e.g., comparative_decoding for rec_stability)

These tests are critical for maintaining stability across the diverse parameter space of color code configurations.

## Repository Guidelines

- Keep `CLAUDE.md` sufficiently concise, containing only high-level overview. No need to contain too much details.