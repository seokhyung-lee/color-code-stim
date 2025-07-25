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
- Supports multiple circuit types: "tri", "rec", "rec_stability", "growing", "cult+growing"
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

### Granular Noise Control
The NoiseModel class supports fine-grained control over reset and measurement operations for different qubit types:

```python
# Granular reset/measurement noise control
noise = NoiseModel(
    reset=0.001,       # Base reset rate for all qubits
    meas=0.002,        # Base measurement rate for all qubits
    reset_data=0.005,  # Override reset rate for data qubits
    reset_anc_X=None,  # Use base reset rate for X-type ancilla qubits
    reset_anc_Z=None,  # Use base reset rate for Z-type ancilla qubits
    meas_data=None,    # Use base measurement rate for data qubits
    meas_anc_X=0.003,  # Override measurement rate for X-type ancilla qubits
    meas_anc_Z=None,   # Use base measurement rate for Z-type ancilla qubits
)
colorcode = ColorCode(d=5, rounds=5, circuit_type="tri", noise_model=noise)
```

**Granular Parameters:**
- `reset_data`, `reset_anc_X`, `reset_anc_Z`: Override reset rates for specific qubit types
- `meas_data`, `meas_anc_X`, `meas_anc_Z`: Override measurement rates for specific qubit types
- If None (default), parameters fall back to base `reset` or `meas` rates
- Enables precise control over noise characteristics for different circuit elements

### Superdense Syndrome Extraction
The package supports superdense syndrome extraction circuits for enhanced quantum error correction:

```python
# Enable superdense syndrome extraction
colorcode = ColorCode(
    d=5,
    rounds=5,
    circuit_type="tri",
    noise_model=noise,
    superdense_circuit=True,           # Enable superdense mode
    cnot_schedule="superdense_default" # Use superdense-optimized schedule
)
```

**Key Features:**
- **Classical Controlled Gates**: Implements measurement-based feedback using `CX` gates with `stim.target_rec()` targets
- **4-Step Superdense Pattern**: X→Z anc CNOTs, data→anc CNOTs, anc→data CNOTs, repeat X→Z CNOTs
- **Spatial Routing**: Data qubits connect to Z-type (x < face_x) or X-type (x > face_x) ancillae based on position
- **Connection Tracking**: Automatically tracks Z-ancilla to data qubit connections for feedback control
- **Simultaneous Extraction**: Enables simultaneous X and Z syndrome information extraction

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

## Development Notes

### Module Structure
- Main package: `src/color_code_stim/`
- Modular decoder system in `decoders/` with clear inheritance hierarchy
- Utility modules: `dem_utils/`, `simulation/`
- Asset files: Pre-computed cultivation circuits in `assets/cultivation_circuits/`

### Recent Refactoring (Phase 4 & 5)
The codebase has undergone modular refactoring with:
- Extracted circuit generation into CircuitBuilder
- Introduced NoiseModel class for systematic noise parameter handling
- Moved DEM generation to DEMManager with caching
- Created modular decoder architecture with BaseDecoder
- Implemented equivalence testing for performance optimization

### Superdense Circuit Implementation
The CircuitBuilder now supports superdense syndrome extraction:
- **Classical Controlled Gates**: Added after Z-type ancilla measurements using `circuit.append("CX", [stim.target_rec(-i), data_qid])`
- **Connection Tracking**: Tracks Z-ancilla to data qubit connections during 4-step superdense sequence
- **Spatial Routing Logic**: Determines ancilla type based on data qubit position relative to face center
- **Backward Compatibility**: Only applies when `superdense_circuit=True`

### Testing and Validation
- Test directory structure exists but may need population
- Equivalence tests validate modular refactoring maintains correctness
- Performance benchmarks ensure optimization improvements

### Performance Considerations
- DEMManager implements caching for expensive DEM operations
- Decoder equivalence testing prevents regression
- Cultivation circuits pre-computed and stored in assets/