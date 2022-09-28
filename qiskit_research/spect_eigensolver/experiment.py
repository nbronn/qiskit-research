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

from ast import Param
from typing import List, Optional, Union

from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.opflow import I, X, Z, OperatorBase, SummedOp
from qiskit.providers.backend import Backend
from qiskit_research.utils.convenience import attach_cr_pulses
from qiskit_research.spect_eigensolver.utils import stringify

import numpy as np

def build_resonance_ham(h0: OperatorBase, coupling_param: Parameter, energy_param: Parameter) -> SummedOp:
    nq = h0.num_qubits
    h_jw = []
    for pop in h0:
        #for pop in op:
        h_jw.append((pop^I).to_pauli_op())
    oplist = [-0.5*energy_param*((I^(nq))^Z), coupling_param*((I^(nq-1))^X^X)]
    oplist += h_jw
    return SummedOp(oplist)

def build_sweep(
    trot_unit: Union[QuantumCircuit, List[QuantumCircuit]],
    energy_range: Union[List[float], np.array],
    sweep_range: Union[List[float], np.array],
    model_params: dict,
    backend: Optional[Backend]=None,
) -> List[QuantumCircuit]:
    my_layout = get_layout(trot_unit)
    qr = QuantumRegister(len(my_layout), 'q')
    cr = ClassicalRegister(1, 'c')
    cc = Parameter('c')
    tt = Parameter('t')
    ww = Parameter('ω')
    circs_w = []
    for sparam in sweep_range:
        (model_params, exp_str) = set_missing_param(model_params, sparam)
        param_bind = invert_params(model_params)
        param_bind[cc] = model_params['c_set']
        param_bind[tt] = model_params['dt_set'] # unit of each time step
        num_steps = int(model_params['t_set']/model_params['dt_set'])

        for w_set in energy_range:
            param_bind[ww] = w_set
            metadata_circ = {
                'experiment': exp_str,
                'layout': my_layout, # TODO: extract this from base circuit
                'trotter': num_steps,
                **param_bind,
            }
            circ = QuantumCircuit(qr, cr, metadata=stringify(metadata_circ))
            for _ in range(num_steps):
                circ.append(trot_unit.to_instruction(), qr)

            circ = circ.decompose()
            circ.measure(my_layout[0], 0)
            if backend is None:
                circs_w.append(circ.bind_parameters(param_bind))
            else:
                circs_w.append(attach_cr_pulses(circ, backend, param_bind))

    return circs_w

def get_layout(circ: QuantumCircuit) -> list:
    dag = circuit_to_dag(circ)
    qubits = circ.qubits

    for wire in dag.idle_wires(ignore=["barrier", "delay"]):
        if wire in qubits:
            qubits.remove(wire)

    return [qubit.index for qubit in qubits]

def set_missing_param(model_params: dict, param: float):
    if 't_set' not in model_params:
        model_params['t_set'] = param
        exp_str = 't_sweep'
    elif 'dt_set' not in model_params:
        model_params['dt_set'] = param
        exp_str = 'dt_sweep'
    elif 'c_set' not in model_params:
        model_params['c_set'] = param
        exp_str = 'c_sweep'
    elif 'm_set' not in model_params:
        model_params['m_set'] = param
        exp_str = 'm_sweep'
    elif 'x_set' not in model_params:
        model_params['x_set'] = param
        exp_str = 'x_sweep'
    elif 'y_set' not in model_params:
        model_params['y_set'] = param
        exp_str = 'y_sweep'
    elif 'z_set' not in model_params:
        model_params['z_set'] = param
        exp_str = 'z_sweep'

    return (model_params, exp_str)

def invert_params(model_params: dict) -> dict:
    TT = Parameter('T')
    DD = Parameter('Δ')
    UU = Parameter('U')
    mu = Parameter('μ')
    T_set = model_params['x_set'] + model_params['y_set']
    D_set = model_params['x_set'] - model_params['y_set']
    U_set = 4*model_params['z_set']
    mu_set = -2*(model_params['m_set'] + model_params['z_set'])
    
    return {TT: T_set, DD: D_set, UU: U_set, mu: mu_set}