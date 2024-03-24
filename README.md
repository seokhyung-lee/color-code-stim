# color-code-stim
A small Python module for simulating &amp; decoding color code circuits.

## Features
- **Simulation of color code circuits using [Stim](https://github.com/quantumlib/Stim) library.** <br> 
It currently supports the logical idling operation of a triangular color code in the hexagonal (6-6-6) lattice. The code distance, number of syndrome extraction rounds, and CNOT schedule of the code are adjustable. Bit-flip and circuit-level noise models are supported and individual noise parameters are adjustable.
- **Implementation of the Concatenated Minimum-weight Perfect Matching (MWPM) Decoder for color codes.** <br>
The paper on the Concatenated MWPM Decoder is now in preparation and will be uploaded on arXiv in the near future. The MWPM sub-routines of the decoder are implemented using [PyMatching](https://github.com/oscarhiggott/PyMatching) library.
- **Easy Monte-Carlo simulation to evaluate the decoder performance.** <br>

## Requirements
The specific versions have been used for testing the module. It is highly probable that newer versions will perform equally well.
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
It is distributed under the MIT license. Please see the LICENSE file for more details.

## Acknowledgements
This work is supported by the Australian Research Council via the Centre of Excellence in Engineered Quantum Systems (EQUS) project number CE170100009, by DARPA via project number HR001122C0101, and by the ARO under Grant Number: W911NF-21-1-0007.
