"""
Decoder modules for Color Code

This package provides modular decoder implementations for quantum error correction
with color codes. The decoders are designed to work with the DEMManager for
accessing decomposed detector error models.
"""

from .base import BaseDecoder
from .concat_matching_decoder import ConcatMatchingDecoder
from .bp_decoder import BPDecoder
from .belief_concat_matching_decoder import BeliefConcatMatchingDecoder

__all__ = ["BaseDecoder", "ConcatMatchingDecoder", "BPDecoder", "BeliefConcatMatchingDecoder"]