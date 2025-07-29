"""
Base decoder interface for Color Code decoders

This module provides a simple base class for decoder implementations.
Following the pragmatic approach, it's not overly abstract but provides
a clean interface for different decoding strategies.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseDecoder(ABC):
    """
    Simple base interface for quantum error correction decoders.
    
    This is a pragmatic base class that provides a common interface for
    different decoding strategies without being overly abstract. Each decoder
    implementation handles its specific requirements while maintaining a
    consistent API.
    """
    
    @abstractmethod
    def decode(self, detector_outcomes: np.ndarray, **kwargs) -> np.ndarray:
        """
        Decode detector outcomes to predict observables.
        
        Parameters
        ----------
        detector_outcomes : np.ndarray
            1D or 2D array of detector measurement outcomes.
            If 1D, interpreted as a single sample.
            If 2D, each row is a sample, each column a detector.
            
        **kwargs
            Additional decoder-specific parameters.
            
        Returns
        -------
        np.ndarray
            Predicted observable outcomes. Shape depends on the decoder
            implementation and input dimensions.
        """
        pass
    
    def supports_comparative_decoding(self) -> bool:
        """
        Check if this decoder supports comparative decoding.
        
        Returns
        -------
        bool
            True if the decoder can test multiple logical classes
            and return logical gaps for magic state distillation.
        """
        return False
    
    def supports_predecoding(self) -> bool:
        """
        Check if this decoder supports pre-decoding strategies.
        
        Returns
        -------
        bool
            True if the decoder supports erasure matching or
            belief propagation pre-decoding.
        """
        return False