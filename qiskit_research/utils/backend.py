# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for dealing with backends and their coupling maps."""


from typing import Optional

from qiskit import BasicAer
from qiskit.providers import Backend, Provider
from qiskit_aer import AerSimulator


def get_backend(
    name: str, provider: Optional[Provider], seed_simulator: Optional[int] = None
) -> Backend:
    """Retrieve a backend."""
    if provider is not None:
        return provider.get_backend(name)
    if name == "aer_simulator":
        return AerSimulator(seed_simulator=seed_simulator)
    if name == "statevector_simulator":
        return BasicAer.get_backend("statevector_simulator")
    raise ValueError("The given name does not match any supported backends.")

def get_coupling_map_from_init_layout(coupling_map, init_layout):
    """
    TODO: write this function
    """

def get_outward_coupling_map(coupling_map, ent_map, start_qubits):
    """
    Recursive method for rearranging a coupling map to generate an entanglement
    map outward from a set of starting qubit(s). The new entanglement map is then
    suitable for use by canned algorithms such as TwoLocal in the entanglement= kwarg.

    Args:
        coupling_map: coupling map of backend (TODO: need to currently deepcopy this before calling method)
        ent_map: rearranged coupling map used for generating entanglement in correct order (TODO: currently take empty list)
        start_qubits: qubits to find the pairs containing (TODO: currently takes [middle_qubit])
    """
    next_qubits = []
    for pair in [pair for pair in coupling_map if pair[0] in start_qubits]:
        next_qubits.append(pair[1])
        ent_map.append(pair)
        coupling_map.remove(pair)
        coupling_map.remove(list(reversed(pair)))

    if next_qubits:
        get_outward_coupling_map(coupling_map, ent_map, next_qubits)


def get_layered_ansatz_coupling_map(coupling_map):
    """
    Method to layer entangling gates of an ansatz type into those that can be executed
    simultaneously. Currently assumes spacing of 1 qubit within a layer.
    TODO: extend to more than single-qubit spacing
    """
    # TODO: may want to make the below directional to support certain backends, i.e. ibm_sherbrooke
    ordered_cm = [pair for pair in coupling_map if pair[0] < pair[1]]

    # split entanglement map into sets of non-overlapping qubits
    ent_map = []
    for pair in ordered_cm:
        if ent_map == []:
            ent_map.append([pair])
        elif all([any([pair[0] in epair or pair[1] in epair for epair in emap]) for emap in ent_map]):
            ent_map.append([pair])
        else:
            for emap in ent_map:
                if all([pair[0] not in epair and pair[1] not in epair for epair in emap]):
                    emap.append(pair)
                    break

    return ent_map