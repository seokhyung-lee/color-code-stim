# color-code-stim
Python package for simulating &amp; decoding 2D color code circuits via the [concatenated MWPM decoder](https://quantum-journal.org/papers/q-2025-01-27-1609).

**Note**: _The [previous version](https://github.com/seokhyung-lee/color-code-stim/tree/53b60e9efb5a691ccdc0a8d1ecab2fb7b76cf301) of this package (used in [our paper](https://quantum-journal.org/papers/q-2025-01-27-1609/)) implemented the bit‑flip noise model incorrectly, leading to an overestimation of the logical failure rate. In that version, each qubit was subjected to bit‑flip noise twice, both before and after the syndrome extraction (see lines 612 and 656 of [`color_code_stim.py`](https://github.com/seokhyung-lee/color-code-stim/blob/53b60e9efb5a691ccdc0a8d1ecab2fb7b76cf301/color_code_stim.py)). This has been corrected in the latest version, where **the estimated bit‑flip noise threshold has been improved from 8.2% (presented in our paper) to 8.6%**, and the logical failure rate has been roughly halved. The circuit‑level results, which form the main focus of the paper, remain unaffected._

**Note**: _See also [ConcatMatching](https://github.com/seokhyung-lee/ConcatMatching) if you want to input your check matrix directly to the decoder instead of using pre-defined color code circuits._

## Features
- **Simulation of 2D color code circuits using [Stim](https://github.com/quantumlib/Stim) library.** <br> 
It currently supports the following circuit types: 
  * `circuit_type="tri"`: Memory experiment of a triangular patch with distance `d` (odd).
  * `circuit_type="rec"`: Memory experiment of a rectangular patch with distances `d` and `d2` (both even).
  * `circuit_type="rec_stability"`: Stability experiment of a rectangle-like patch with single-type boundaries. `d` and `d2` (both even) indicate the sizes of the patch, rather than code distances.
  * `circuit_type="growing"`: Growing operation of a triangular patch from distance `d` to `d2` (both odd).
  * `circuit_type="cult+growing"`: Cultivation on a triangular patch with distance `d` (3 or 5), followed by a growing operation to distance `d2` (odd). The cultivation circuits suggested in [arXiv:2409.17595 by Gidney, Shutty, and Jones](https://arxiv.org/abs/2409.17595) (excluding the grafting process) are used to construct this circuit.
- **Implementation of the Concatenated Minimum-Weight Perfect Matching (MWPM) Decoder for color codes.** <br>
The concatenated MWPM decoder is a decoder for color codes that functions by concatenation of two MWPM decoders per color, for a total of six matchings. See [Quantum 9, 1609 (2025)](https://doi.org/10.22331/q-2025-01-27-1609) for more details. The MWPM sub-routines of the decoder are implemented using [PyMatching](https://github.com/oscarhiggott/PyMatching) library.
- **Support of the [superdense syndrome extraction circuit](https://arxiv.org/abs/2312.08813).** <br>
Set `superdense_circuit=True` when initializing a `ColorCode` instance. By default, it is set to `False` and the space multiplexing circuit is used.
- **Comparative decoding \& calculation of the logical gap** <br>
By setting `comparative_decoding=True` (default is `False`) when defining a `ColorCode` object, the concatenated MWPM decoder can be executed multiple times over all distinct logical classes. The minimum-weight correction is chosen as the final correction, and the resulting **logical gap** quantifies its reliability, which can be used for post-selection. This feature was not discussed in our original [paper](https://doi.org/10.22331/q-2025-01-27-1609) but has been added for our following [paper](https://arxiv.org/abs/2409.07707) on color code magic state distillation.
- **Easy Monte-Carlo simulation to evaluate the decoder performance.** <br>

## Project Structure

```
color-code-stim/
├── src/color_code_stim/          # Main package source
│   ├── color_code.py             # Core ColorCode class & interface
│   ├── circuit_builder.py        # Circuit generation
│   ├── graph_builder.py          # Tanner graph construction
│   ├── decoders/                 # Modular decoder implementations
│   │   ├── base.py               # Base decoder interface
│   │   ├── concat_matching_decoder.py  # Main concatenated MWPM decoder
│   │   └── ...                   # Additional decoders
│   ├── dem_utils/                # Detector error model utilities
│   ├── simulation/               # Monte Carlo simulation tools
│   ├── assets/                   # Pre-computed circuits & data
│   └── ...                       # Utility modules
└── ...
```

**Key Components**:
- **ColorCode**: Main interface for circuit simulation & decoding
- **CircuitBuilder**: Generates quantum circuits for different geometries
- **TannerGraphBuilder**: Generates tanner graphs for different patch types
- **Decoders**: Modular decoder architecture with MWPM implementation
- **DemManager**: Manages detector error models
- **Simulator**: Handles error sampling and Monte Carlo simulations

## Install

Requires Python >= 3.11

### For Users (Direct Install)
```bash
pip install color-code-stim
```

### For Development (Editable Install)
```bash
git clone https://github.com/seokhyung-lee/color-code-stim.git
cd color-code-stim
pip install -e .
```

## Usage

- [Getting Started Notebook](getting_started.ipynb).
- [API Reference](https://seokhyung-lee.github.io/color-code-stim/)

### Quick Start

```python
from color_code_stim import ColorCode, NoiseModel

# Create noise model
noise = NoiseModel.uniform_circuit_noise(1e-3)

# Create color code instance
colorcode = ColorCode(
    d=5,                    # Code distance
    rounds=5,              # Syndrome extraction rounds
    circuit_type="tri",    # Circuit type
    noise_model=noise      # Noise configuration
)

# Run simulation
num_fails, info = colorcode.simulate(shots=10000, full_output=True)
```

## Citation
If you want to cite this package in an academic work, please cite the [paper](https://doi.org/10.22331/q-2025-01-27-1609):

```bibtex
@article{lee2025color,
  doi = {10.22331/q-2025-01-27-1609},
  url = {https://doi.org/10.22331/q-2025-01-27-1609},
  title = {Color code decoder with improved scaling for correcting circuit-level noise},
  author = {Lee, Seok-Hyung and Li, Andrew and Bartlett, Stephen D.},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {9},
  pages = {1609},
  month = jan,
  year = {2025}
}
```

## License
This package is distributed under the MIT license. Please see the LICENSE file for more details.

## Acknowledgements
This package is based upon work supported by the Australian Research Council via the Centre of Excellence in Engineered Quantum Systems (EQUS) project number CE170100009 and by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001122C0063.
