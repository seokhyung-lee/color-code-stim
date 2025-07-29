"""
Noise model for color code quantum error correction.

This module provides the NoiseModel class for systematic handling of noise
parameters in color code quantum circuits.
"""

from typing import Optional, Iterator, Tuple, Self


class NoiseModel:
    """
    Noise model for color code quantum circuits with dictionary-like access.

    This class encapsulates all noise parameters used in color code circuits,
    providing a clean interface for setting.


    Examples
    --------
    Uniform circuit-level noise model:
    >>> noise = NoiseModel.uniform_circuit_noise(0.001)
    >>> print(noise['cnot'])  # 0.001

    Code capacity noise model (depolarizing noise on data qubits before each round):
    >>> noise = NoiseModel(depol=0.01)

    Bit-flip noise on data qubits before each round:
    >>> noise = NoiseModel(bitflip=0.01)

    Phenomenological noise model:
    >>> noise = NoiseModel(depol=0.01, meas=0.01)

    Noise model with specific parameters:
    >>> noise = NoiseModel(cnot=0.002, idle=0.001, reset=0.005, meas=0.003)
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
        depol1_after_cnot: float = 0.0,
        idle_during_cnot: Optional[float] = None,
        idle_during_meas: Optional[float] = None,
        reset_data: Optional[float] = None,
        reset_anc_X: Optional[float] = None,
        reset_anc_Z: Optional[float] = None,
        meas_data: Optional[float] = None,
        meas_anc_X: Optional[float] = None,
        meas_anc_Z: Optional[float] = None,
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
        depol1_after_cnot : float, default 0.0
            Single-qubit depolarizing noise rate applied to each qubit participating
            in CNOT gates after the gates are applied. If provided and positive,
            DEPOLARIZE1 is added for each qubit involved in the CNOT operations.
        idle_during_cnot : float, optional
            Single-qubit depolarizing noise rate for idle qubits during CNOT operations.
            If None (default), uses the idle parameter. If set to any value (including 0),
            overrides idle for qubits not participating in CNOT gates.
        idle_during_meas : float, optional
            Single-qubit depolarizing noise rate for idle qubits during measurement operations.
            If None (default), uses the idle parameter. If set to any value (including 0),
            overrides idle for qubits not participating in measurement operations.
        reset_data : float, optional
            Error rate for reset operations on data qubits (producing orthogonal state).
            If None (default), uses the reset parameter.
        reset_anc_X : float, optional
            Error rate for reset operations on X-type ancilla qubits (producing orthogonal state).
            If None (default), uses the reset parameter.
        reset_anc_Z : float, optional
            Error rate for reset operations on Z-type ancilla qubits (producing orthogonal state).
            If None (default), uses the reset parameter.
        meas_data : float, optional
            Error rate for measurement operations on data qubits (flipped outcome).
            If None (default), uses the meas parameter.
        meas_anc_X : float, optional
            Error rate for measurement operations on X-type ancilla qubits (flipped outcome).
            If None (default), uses the meas parameter.
        meas_anc_Z : float, optional
            Error rate for measurement operations on Z-type ancilla qubits (flipped outcome).
            If None (default), uses the meas parameter.
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
            "depol1_after_cnot": float(depol1_after_cnot),
        }

        # Handle special parameters that can be None
        if cult is not None:
            self._params["cult"] = float(cult)
        else:
            self._params["cult"] = None

        if idle_during_cnot is not None:
            self._params["idle_during_cnot"] = float(idle_during_cnot)
        else:
            self._params["idle_during_cnot"] = None

        if idle_during_meas is not None:
            self._params["idle_during_meas"] = float(idle_during_meas)
        else:
            self._params["idle_during_meas"] = None

        # Handle granular reset parameters that can be None
        if reset_data is not None:
            self._params["reset_data"] = float(reset_data)
        else:
            self._params["reset_data"] = None

        if reset_anc_X is not None:
            self._params["reset_anc_X"] = float(reset_anc_X)
        else:
            self._params["reset_anc_X"] = None

        if reset_anc_Z is not None:
            self._params["reset_anc_Z"] = float(reset_anc_Z)
        else:
            self._params["reset_anc_Z"] = None

        # Handle granular measurement parameters that can be None
        if meas_data is not None:
            self._params["meas_data"] = float(meas_data)
        else:
            self._params["meas_data"] = None

        if meas_anc_X is not None:
            self._params["meas_anc_X"] = float(meas_anc_X)
        else:
            self._params["meas_anc_X"] = None

        if meas_anc_Z is not None:
            self._params["meas_anc_Z"] = float(meas_anc_Z)
        else:
            self._params["meas_anc_Z"] = None

        # Validate all parameters
        self.validate()

    @classmethod
    def uniform_circuit_noise(cls, p_circuit: float) -> Self:
        """
        Create noise model with uniform circuit-level noise. Equivalent to:
        >>> NoiseModel(reset=p_circuit, meas=p_circuit, cnot=p_circuit, idle=p_circuit)

        Parameters
        ----------
        p_circuit : float
            Circuit-level error rate to apply uniformly.

        Returns
        -------
        NoiseModel
            New noise model with uniform rates.
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
            depol1_after_cnot=0.0,  # Not included in circuit-level noise
            idle_during_cnot=None,  # Not included in circuit-level noise
            idle_during_meas=None,  # Not included in circuit-level noise
            reset_data=None,  # Will default to reset when needed
            reset_anc_X=None,  # Will default to reset when needed
            reset_anc_Z=None,  # Will default to reset when needed
            meas_data=None,  # Will default to meas when needed
            meas_anc_X=None,  # Will default to meas when needed
            meas_anc_Z=None,  # Will default to meas when needed
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

        # Handle special cases of parameters that can fallback to other parameters
        value = self._params[key]
        if key == "cult" and value is None:
            # Default cult to cnot rate if not explicitly set
            return self._params["cnot"]
        elif key == "idle_during_cnot" and value is None:
            # Default idle_during_cnot to idle rate if not explicitly set
            return self._params["idle"]
        elif key == "idle_during_meas" and value is None:
            # Default idle_during_meas to idle rate if not explicitly set
            return self._params["idle"]
        # Handle granular reset parameters fallback to base reset
        elif key == "reset_data" and value is None:
            return self._params["reset"]
        elif key == "reset_anc_X" and value is None:
            return self._params["reset"]
        elif key == "reset_anc_Z" and value is None:
            return self._params["reset"]
        # Handle granular measurement parameters fallback to base meas
        elif key == "meas_data" and value is None:
            return self._params["meas"]
        elif key == "meas_anc_X" and value is None:
            return self._params["meas"]
        elif key == "meas_anc_Z" and value is None:
            return self._params["meas"]

        return value

    def __setitem__(self, key: str, value: Optional[float]) -> None:
        """
        Enable dictionary-like assignment: noise_model['depol'] = 0.01.

        Parameters
        ----------
        key : str
            Parameter name to set.
        value : float or None
            Parameter value to assign. None is allowed for certain parameters.

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

        # Handle None values for special parameters
        if value is None:
            if key in {
                "cult",
                "idle_during_cnot",
                "idle_during_meas",
                "reset_data",
                "reset_anc_X",
                "reset_anc_Z",
                "meas_data",
                "meas_anc_X",
                "meas_anc_Z",
            }:
                self._params[key] = None
                return
            else:
                raise ValueError(
                    f"Noise parameter '{key}' cannot be None. "
                    f"Only 'cult', 'idle_during_cnot', 'idle_during_meas', "
                    f"and granular reset/measurement parameters can be None."
                )

        # Convert to float and validate
        float_value = float(value)
        if float_value < 0:
            raise ValueError(
                f"Noise parameter '{key}' must be non-negative, got {float_value}"
            )

        # Special handling for special parameters
        if key == "cult":
            self._params[key] = float_value if float_value > 0 else None
        elif key in {"idle_during_cnot", "idle_during_meas"}:
            # For idle context parameters, any explicit value (including 0) overrides idle
            self._params[key] = float_value
        elif key in {
            "reset_data",
            "reset_anc_X",
            "reset_anc_Z",
            "meas_data",
            "meas_anc_X",
            "meas_anc_Z",
        }:
            # For granular reset/measurement parameters, any explicit value (including 0) overrides base parameter
            self._params[key] = float_value
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
            if (
                key
                in {
                    "cult",
                    "idle_during_cnot",
                    "idle_during_meas",
                    "reset_data",
                    "reset_anc_X",
                    "reset_anc_Z",
                    "meas_data",
                    "meas_anc_X",
                    "meas_anc_Z",
                }
                and value is None
            ):
                continue  # These parameters can be None
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
            if (
                key
                in {
                    "cult",
                    "idle_during_cnot",
                    "idle_during_meas",
                    "reset_data",
                    "reset_anc_X",
                    "reset_anc_Z",
                    "meas_data",
                    "meas_anc_X",
                    "meas_anc_Z",
                }
                and value is None
            ):
                param_strs.append(f"{key}=None")
            else:
                param_strs.append(f"{key}={value}")

        return f"NoiseModel({', '.join(param_strs)})"
