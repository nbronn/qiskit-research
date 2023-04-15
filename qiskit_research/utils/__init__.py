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

"""
======================================================
Utilities for running research experiments with Qiskit
======================================================
"""

from qiskit_research.utils.backend import (
    get_backend,
    convert_list_map_to_dict,
    convert_dict_to_list,
    matrix_to_dict,
    get_outward_coupling_map,
    get_layered_ansatz_coupling_map,
    PopulateCouplingMapDictAndMatrixDict,
    GetEntanglingMapFromInitLayout,
    PlotLayerData,
)
from qiskit_research.utils.dynamical_decoupling import (
    add_pulse_calibrations,
    dynamical_decoupling_passes,
)
from qiskit_research.utils.gate_decompositions import (
    RZXtoEchoedCR,
    XXMinusYYtoRZX,
    XXPlusYYtoRZX,
    RZXWeylDecomposition,
)
from qiskit_research.utils.pauli_twirling import (
    PauliTwirl,
    pauli_transpilation_passes,
)
from qiskit_research.utils.pulse_scaling import (
    BindParameters,
    CombineRuns,
    SECRCalibrationBuilder,
    cr_scaling_passes,
    pulse_attaching_passes,
)
from qiskit_research.utils.periodic_dynamical_decoupling import (
    PeriodicDynamicalDecoupling,
)

from qiskit_research.utils.circuit_layering import (
    get_entanglement_map,
    FindBlockTrotterEvolution,
    AddBarriersForGroupOfLayers,
    ExpandBlockOperators,
    LayerBlockOperators,
)

__all__ = [
    "get_backend",
    "convert_list_map_to_dict",
    "convert_dict_to_list",
    "matrix_to_dict",
    "PopulateCouplingMapDictAndMatrixDict",
    "GetEntanglingMapFromInitLayout",
    "PlotLayerData",
    "get_outward_coupling_map",
    "get_layered_ansatz_coupling_map",
    "add_pulse_calibrations",
    "dynamical_decoupling_passes",
    "RZXtoEchoedCR",
    "XXMinusYYtoRZX",
    "XXPlusYYtoRZX",
    "RZXWeylDecomposition",
    "BindParameters",
    "CombineRuns",
    "SECRCalibrationBuilder",
    "cr_scaling_passes",
    "pulse_attaching_passes",
    "PauliTwirl",
    "pauli_transpilation_passes",
    "get_entanglement_map",
    "PeriodicDynamicalDecoupling",
    "FindBlockTrotterEvolution",
    "LayerBlockOperators",
    "ExpandBlockOperators",
    "AddBarriersForGroupOfLayers",
]
