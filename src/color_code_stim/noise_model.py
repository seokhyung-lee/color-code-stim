"""
Noise model for color code quantum error correction.

This module provides the NoiseModel class for systematic handling of noise
parameters in color code quantum circuits.
"""

from typing import Optional, Iterator, Tuple


class NoiseModel:
    """
    Noise model for color code quantum circuits with dictionary-like access.

    This class encapsulates all noise parameters used in color code circuits,
    providing a clean interface for setting.
    """

    def __init__(
        self,
        bitflip: float = 0.0,
        depol: float = 0.0,
        reset: float = 0.0,
        meas: float = 0.0,
        cnot: float = 0.0,
        idle: float = 0.0,
        cult: Optional[float] = None,
        initial_data_qubit_depol: float = 0.0,
    ):
        """
        Initialize noise model with individual parameters.

        All parameters must be non-negative floats representing error rates.

        Parameters
        ----------
        bitflip : float, default 0.0
            Bit-flip noise rate on data qubits at the start of each round.
        depol : float, default 0.0
            Depolarizing noise rate on data qubits at the start of each round.
        reset : float, default 0.0
            Error rate for reset operations (producing orthogonal state).
        meas : float, default 0.0
            Error rate for measurement operations (flipped outcome).
        cnot : float, default 0.0
            Two-qubit depolarizing noise rate for CNOT gates.
        idle : float, default 0.0
            Single-qubit depolarizing noise rate for idle operations.
        cult : float, optional
            Physical error rate during cultivation (for cult+growing circuits).
            If not provided, defaults to cnot rate when needed.
        initial_data_qubit_depol : float, default 0.0
            Depolarizing noise rate applied to all data qubits after the first
            syndrome extraction round (if perfect_first_syndrome_extraction=True)
            or after data qubit initialization (if perfect_first_syndrome_extraction=False).

        Examples
        --------
        >>> # Create noise model with specific parameters
        >>> noise = NoiseModel(depol=0.001, cnot=0.002, meas=0.001)
        >>> print(noise['depol'])  # 0.001
        >>> noise['idle'] = 0.0005

        >>> # Create uniform circuit-level noise
        >>> noise = NoiseModel.uniform_circuit_noise(0.001)
        >>> print(noise['cnot'])  # 0.001
        """
        # Store parameters in internal dict for easy access
        self._params = {
            "bitflip": float(bitflip),
            "depol": float(depol),
            "reset": float(reset),
            "meas": float(meas),
            "cnot": float(cnot),
            "idle": float(idle),
            "initial_data_qubit_depol": float(initial_data_qubit_depol),
        }

        # Handle cultivation noise separately since it can be None
        if cult is not None:
            self._params["cult"] = float(cult)
        else:
            self._params["cult"] = None

        # Validate all parameters
        self.validate()

    @classmethod
    def uniform_circuit_noise(cls, p_circuit: float) -> "NoiseModel":
        """
        Create noise model with uniform circuit-level noise.

        This is equivalent to the original p_circuit parameter, where
        reset, measurement, CNOT, and idle operations all have the same
        error rate.

        Parameters
        ----------
        p_circuit : float
            Circuit-level error rate to apply uniformly.

        Returns
        -------
        NoiseModel
            New noise model with uniform rates.

        Examples
        --------
        >>> noise = NoiseModel.uniform_circuit_noise(0.001)
        >>> noise['reset'] == noise['meas'] == noise['cnot'] == noise['idle']
        True
        """
        return cls(
            bitflip=0.0,  # Not included in circuit-level noise
            depol=0.0,  # Not included in circuit-level noise
            reset=p_circuit,
            meas=p_circuit,
            cnot=p_circuit,
            idle=p_circuit,
            cult=None,  # Will default to cnot when needed
            initial_data_qubit_depol=0.0,  # Not included in circuit-level noise
        )

    def __getitem__(self, key: str) -> float:
        """
        Enable dictionary-like access: noise_model['depol'].

        Parameters
        ----------
        key : str
            Parameter name to retrieve.

        Returns
        -------
        float
            Parameter value.

        Raises
        ------
        KeyError
            If parameter name is not recognized.
        """
        if key not in self._params:
            valid_keys = list(self._params.keys())
            raise KeyError(
                f"Unknown noise parameter '{key}'. Valid parameters: {valid_keys}"
            )

        # Handle special case of cult parameter
        value = self._params[key]
        if key == "cult" and value is None:
            # Default cult to cnot rate if not explicitly set
            return self._params["cnot"]

        return value

    def __setitem__(self, key: str, value: float) -> None:
        """
        Enable dictionary-like assignment: noise_model['depol'] = 0.01.

        Parameters
        ----------
        key : str
            Parameter name to set.
        value : float
            Parameter value to assign.

        Raises
        ------
        KeyError
            If parameter name is not recognized.
        ValueError
            If value is negative.
        """
        if key not in self._params:
            valid_keys = list(self._params.keys())
            raise KeyError(
                f"Unknown noise parameter '{key}'. Valid parameters: {valid_keys}"
            )

        # Convert to float and validate
        float_value = float(value)
        if float_value < 0:
            raise ValueError(
                f"Noise parameter '{key}' must be non-negative, got {float_value}"
            )

        # Special handling for cult parameter
        if key == "cult":
            self._params[key] = float_value if float_value > 0 else None
        else:
            self._params[key] = float_value

    def __contains__(self, key: str) -> bool:
        """
        Enable 'in' operator: 'depol' in noise_model.

        Parameters
        ----------
        key : str
            Parameter name to check.

        Returns
        -------
        bool
            True if parameter exists.
        """
        return key in self._params

    def keys(self) -> Iterator[str]:
        """
        Return parameter names.

        Returns
        -------
        Iterator[str]
            Iterator over parameter names.
        """
        return iter(self._params.keys())

    def values(self) -> Iterator[float]:
        """
        Return parameter values.

        Returns
        -------
        Iterator[float]
            Iterator over parameter values.
        """
        for key in self._params:
            yield self[key]  # Use __getitem__ to handle cult parameter

    def items(self) -> Iterator[Tuple[str, float]]:
        """
        Return (parameter, value) pairs.

        Returns
        -------
        Iterator[Tuple[str, float]]
            Iterator over (name, value) pairs.
        """
        for key in self._params:
            yield (key, self[key])  # Use __getitem__ to handle cult parameter

    def validate(self) -> None:
        """
        Validate all parameters are non-negative.

        Raises
        ------
        ValueError
            If any parameter is negative.
        """
        for key, value in self._params.items():
            if key == "cult" and value is None:
                continue  # cult can be None
            if value < 0:
                raise ValueError(
                    f"Noise parameter '{key}' must be non-negative, got {value}"
                )

    def is_noiseless(self) -> bool:
        """
        Check if all noise parameters are zero.

        Returns
        -------
        bool
            True if all parameters are zero (noiseless model).
        """
        for value in self.values():
            if value > 0:
                return False
        return True

    def __str__(self) -> str:
        """
        String representation for end users.

        Returns
        -------
        str
            Human-readable string representation.
        """
        if self.is_noiseless():
            return "NoiseModel(noiseless)"

        # Show only non-zero parameters
        nonzero_params = []
        for key, value in self.items():
            if value > 0:
                nonzero_params.append(f"{key}={value}")

        return f"NoiseModel({', '.join(nonzero_params)})"

    def __repr__(self) -> str:
        """
        Developer representation.

        Returns
        -------
        str
            Detailed string representation for debugging.
        """
        param_strs = []
        for key, value in self._params.items():
            if key == "cult" and value is None:
                param_strs.append(f"{key}=None")
            else:
                param_strs.append(f"{key}={value}")

        return f"NoiseModel({', '.join(param_strs)})"
