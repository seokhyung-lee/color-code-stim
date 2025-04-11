# color-code-stim
Python package for simulating &amp; decoding 2D color code circuits.

**Note**: _See also [ConcatMatching](https://github.com/seokhyung-lee/ConcatMatching) if you want to input your check matrix directly to the decoder instead of using pre-defined color code circuits._

## Features
- **Simulation of 2D color code circuits using [Stim](https://github.com/quantumlib/Stim) library.** <br> 
It currently supports the logical idling operation of a 2D triangular color code defined in the hexagonal (6-6-6) lattice. The code distance, number of syndrome extraction rounds, and CNOT schedule of the code are adjustable. Bit-flip and circuit-level noise models are supported and individual noise parameters are adjustable.
- **Implementation of the Concatenated Minimum-Weight Perfect Matching (MWPM) Decoder for color codes.** <br>
The concatenated MWPM decoder is a decoder for color codes that functions by concatenation of two MWPM decoders per color, for a total of six matchings. See [Quantum 9, 1609 (2025)](https://doi.org/10.22331/q-2025-01-27-1609) for more details. The MWPM sub-routines of the decoder are implemented using [PyMatching](https://github.com/oscarhiggott/PyMatching) library.
- **Comparative decoding \& calculation of the logical gap** <br>
By setting `comparative_decoding=True` (default is `False`) when defining a `ColorCode` object, the concatenated MWPM decoder can be executed multiple times over all distinct logical classes. The minimum-weight correction is chosen as the final correction, and the resulting **logical gap** quantifies its reliability, which can be used for post-selection. This feature was not discussed in our original [paper](https://doi.org/10.22331/q-2025-01-27-1609) but has been added for our following [paper](https://arxiv.org/abs/2409.07707) on color code magic state distillation.
- **Easy Monte-Carlo simulation to evaluate the decoder performance.** <br>
- **Integration with cultivation** (under development)

## Install

Requires Python >= 3.11

```bash
pip install git+https://github.com/seokhyung-lee/color-code-stim.git
```

## Usage

See the [Getting Started Notebook](getting_started.ipynb).

## Citation
If you want to cite this module in an academic work, please cite the [paper](https://doi.org/10.22331/q-2025-01-27-1609):

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
This module is distributed under the MIT license. Please see the LICENSE file for more details.

## Acknowledgements
This module is based upon work supported by the Australian Research Council via the Centre of Excellence in Engineered Quantum Systems (EQUS) project number CE170100009 and by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001122C0063.
