# Color Code Circuit for Shuttling-based Distributed Quantum Computing (SDQC)

- A variant of a triangular patch memory experiment circuit with superdense syndrome extraction circuit
- `ColorCode(circuit_type="sdqc_memory", superdense_circuit=True)`
  * Raises an error if `superdense_circuit=False`
  * Other parameters behave in the same way as `circuit_type="tri"`.
- Differences from `circuit_type="tri"`:
  * Ancillary qubits undergo "shuttling" during each syndrome extraction round.
  * In the circuit, shuttling is specified as an identity operator `"I"` (this does nothing effectively, but include it in the circuit for convenience)
  * Shuttling is immediately followed by "DEPOLARIZE1" errors.
  * Faces are divided into two groups: "segmented" and "non-segmented".
  * Four types of shuttling error rates exist: `shuttling_seg_init`, `shuttling_non_seg_init`, `shuttling_seg_final`, `shuttling_non_seg_final`. Modify `NoiseModel` to take these as special parameters.
  * Importantly, the location of shuttling differs depending on whether faces are segmented or non-segmented:
    - For segmented faces, shuttlings on the relevant two ancillary qubits are placed just before the final CNOT gate between the two ancillary qubits (with `shuttling_seg_final` error rate)
    - For non-segmented faces, shuttlings are placed just before the final measurements of the ancillary qubits (with `shuttling_non_seg_final` error rate)
    - For both types of faces, shuttlings are also placed just after the first CNOT gate between the two ancillary qubits. (with `shuttling_seg_init` or `shuttling_non_seg_init` error rate)
  * The location of segmented faces depend on the code distance `d`.
    - d=3, d=5: no segmented faces
    - d=7: face_x = 20
    - d=9: face_x = 20, 32
    - d=11: face_x = 20, 32, 44, 
    - d=13: face_x = 20, 32, 38, 44, 56
    - d>=15: raise `NotImplementedError`
  * Add an optional parameter `set_all_faces_segmented=False`. If True, ignore the above and just assume all faces are segmented. (so d>=15 is permitted in this case.)
  * For visual aid, in `draw_lattice` method, add a parameter `highlight_segmented_faces=True`. If True, this is equivalent to set `highlight_faces` to be segmented faces.
  * Other changes not related to shuttling errors:
    - Do **NOT** add idling errors (DEPOLARIZE1) on data qubits during CNOT gates between ancillar qubits and during resets/measurements of ancillary qubits. Namely, idling errors on data qubits only occur when other data qubits are involved in CNOT gates.
    - `depol1_after_cnot` error rate in `NoiseModel` now affects only CNOT gates between data and ancillary qubits. (since CNOT gates between ancillary qubits already have shuttling errors.)

## Other changes applied universally

The following changes are applied to all circuit types not limited to "sdqc_memory"

- Add a new error rate `depol1_on_anc_before_cnot` to `NoiseModel`. This adds a DEPOLARIZE1 error ONLY on the ancillary qubit before each CNOT gate between data and ancillary qubits.