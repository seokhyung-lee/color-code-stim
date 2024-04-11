# color-code-stim
A small Python module for simulating &amp; decoding color code circuits.

## Features
- **Simulation of color code circuits using [Stim](https://github.com/quantumlib/Stim) library.** <br> 
It currently supports the logical idling operation of a triangular color code defined in the hexagonal (6-6-6) lattice. The code distance, number of syndrome extraction rounds, and CNOT schedule of the code are adjustable. Bit-flip and circuit-level noise models are supported and individual noise parameters are adjustable.
- **Implementation of the Concatenated Minimum-Weight Perfect Matching (MWPM) Decoder for color codes.** <br>
The concatenated MWPM decoder is a decoder for color codes that functions by concatenation of two MWPM decoders per color, for a total of six matchings. See arXiv:to_add for more details. The MWPM sub-routines of the decoder are implemented using [PyMatching](https://github.com/oscarhiggott/PyMatching) library.
- **Easy Monte-Carlo simulation to evaluate the decoder performance.** <br>

## Requirements
These specific versions have been used for testing the module. It is highly probable that newer versions will perform equally well.
- Python 3.9.16
- `numpy==1.23.5`
- `matplotlib==3.7.1`
- `stim==1.12.0`
- `pymatching==2.1.0`
- `python-igraph==0.11.3`
- `statsmodels==0.14.1`

## Usage
Just place [color_code_stim.py](color_code_stim.py) in your desired directory and import it.

See the [Getting Started Notebook](getting_started.ipynb).

## Citation
Paper in preparation.

## License
This module is distributed under the MIT license. Please see the LICENSE file for more details.

## Acknowledgements
This work is supported by the Australian Research Council via the Centre of Excellence in Engineered Quantum Systems (EQUS) project number CE170100009. This article is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001122C0063.
