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

from typing import Any, Iterable, List, Optional, Union, cast

from copy import deepcopy
import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IGate, XGate, YGate, ZGate, CXGate, RZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOpNode
from qiskit.providers.backend import Backend
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import ALAPSchedule
from qiskit.transpiler import PassManager
# from qiskit_research.utils import get_instruction_durations

def stringify(param_bind: dict) -> dict:
    param_bind_str = {}
    for key in param_bind.keys():
        param_bind_str[str(key)] = param_bind[key]

    return param_bind_str

def cost_func(circ, layouts, backend):
    """
    A custom cost function that includes T1 and T2 computed during idle periods

    Parameters:
        circ (QuantumCircuit): circuit of interest
        layouts (list of lists): List of specified layouts
        backend (IBMQBackend): An IBM Quantum backend instance

    Returns:
        list: Tuples of layout and cost
    """
    out = []
    props = backend.properties()
    dt = backend.configuration().dt
    num_qubits = backend.configuration().num_qubits
    t1s = [props.qubit_property(qq, 'T1')[0] for qq in range(num_qubits)]
    t2s = [props.qubit_property(qq, 'T2')[0] for qq in range(num_qubits)]
    for layout in layouts:
        # sch_circ = transpile(circ, backend, initial_layout=layout, basis_gates=['rz', 'sx', 'x', 'cx', 'rzx'],
        #                      optimization_level=0, scheduling_method='alap')
        sch_circ = circ
        # pass_ = ALAPSchedule(get_instruction_durations(backend))
        # pm = PassManager(pass_)
        # sch_circ = pm.run(circ)
        error = 0
        fid = 1
        touched = set()
        for item in sch_circ._data:
            if item[0].name == 'cx':
                q0 = sch_circ.find_bit(item[1][0]).index
                q1 = sch_circ.find_bit(item[1][1]).index
                fid *= (1-props.gate_error('cx', [q0, q1]))
                touched.add(q0)
                touched.add(q1)

            # if it is a scaled pulse derived from cx
            elif item[0].name == 'rzx':
                q0 = sch_circ.find_bit(item[1][0]).index
                q1 = sch_circ.find_bit(item[1][1]).index

                cr_error = (float(item[0].params[0])/(np.pi/2)) * props.gate_error('cx', [layout[q0], layout[q1]])

                # assumes control qubit is actually control for cr
                echo_error = props.gate_error('x', layout[q0])

                fid *= (1 - max(cr_error, echo_error))

            elif item[0].name in ['sx', 'x']:
                q0 = sch_circ.find_bit(item[1][0]).index
                fid *= 1-props.gate_error(item[0].name, q0)
                touched.add(q0)

            elif item[0].name == 'measure':
                q0 = sch_circ.find_bit(item[1][0]).index
                fid *= 1-props.readout_error(q0)
                touched.add(q0)

            # elif item[0].name == 'delay':
            #     q0 = sch_circ.find_bit(item[1][0]).index
            #     # Ignore delays that occur before gates
            #     # This assumes you are in ground state and errors
            #     # do not occur.
            #     if q0 in touched:
            #         time = item[0].duration * dt
            #         fid *= 1-idle_error(time, t1s[q0], t2s[q0])

        error = 1-fid
        out.append((layout, error))
    return out


def idle_error(time, t1, t2):
    """Compute the approx. idle error from T1 and T2
    Parameters:
        time (float): Delay time in sec
        t1 (float): T1 time in sec
        t2, (float): T2 time in sec
    Returns:
        float: Idle error
    """
    t2 = min(t1, t2)
    rate1 = 1/t1
    rate2 = 1/t2
    p_reset = 1-np.exp(-time*rate1)
    p_z = (1-p_reset)*(1-np.exp(-time*(rate2-rate1)))/2
    return p_z + p_reset
