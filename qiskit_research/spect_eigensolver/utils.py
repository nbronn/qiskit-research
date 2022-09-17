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
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IGate, XGate, YGate, ZGate, CXGate, RZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOpNode
from qiskit.opflow import I, X, Z, OperatorBase, SummedOp
from qiskit.providers.backend import Backend
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import ALAPSchedule
from qiskit.transpiler import PassManager
from qiskit_research.utils import get_instruction_durations

def build_resonance_ham(h0: OperatorBase, coupling_param: Parameter, energy_param: Parameter) -> SummedOp:
    nq = h0.num_qubits
    h_jw = []
    for pop in h0:
        #for pop in op:
        h_jw.append((pop^I).to_pauli_op())
    oplist = [-0.5*energy_param*((I^(nq))^Z), coupling_param*((I^(nq-1))^X^X)]
    oplist += h_jw
    return SummedOp(oplist)

def get_zz_temp_sub() -> QuantumCircuit:
    rzx_dag = circuit_to_dag(deepcopy(rzx_templates.rzx_templates(['zz3'])['template_list'][0]))
    temp_cx1_node = rzx_dag.front_layer()[0]
    for gp in rzx_dag.bfs_successors(temp_cx1_node):
        if gp[0] == temp_cx1_node:
            if isinstance(gp[1][0].op, CXGate) and isinstance(gp[1][1].op, RZGate):
                temp_rz_node = gp[1][1]
                temp_cx2_node = gp[1][0]

                rzx_dag.remove_op_node(temp_cx1_node)
                rzx_dag.remove_op_node(temp_rz_node)
                rzx_dag.remove_op_node(temp_cx2_node)

    return dag_to_circuit(rzx_dag).inverse()

def sub_zz_in_dag(dag: DAGCircuit, cx1_node: DAGNode, rz_node: DAGNode, cx2_node: DAGNode) -> DAGCircuit:
    zz_temp_sub = get_zz_temp_sub().assign_parameters({get_zz_temp_sub().parameters[0]: rz_node.op.params[0]})
    dag.remove_op_node(rz_node)
    dag.remove_op_node(cx2_node)

    qr = QuantumRegister(2, 'q')
    mini_dag = DAGCircuit()
    mini_dag.add_qreg(qr)
    for idx, (instr, qargs, cargs) in enumerate(zz_temp_sub.data):
        mini_dag.apply_operation_back(instr, qargs=qargs)

    dag.substitute_node_with_dag(node=cx1_node, input_dag=mini_dag, wires=[qr[0], qr[1]])
    return dag

def forced_zz_temp_sub(dag: DAGCircuit) -> DAGCircuit:
    cx_runs = dag.collect_runs('cx')
    for run in cx_runs:
        cx1_node = run[0]
        gp = next(dag.bfs_successors(cx1_node))
        if isinstance(gp[0].op, CXGate): # dunno why this is needed
            if isinstance(gp[1][0], DAGOpNode) and isinstance(gp[1][1], DAGOpNode):
                if isinstance(gp[1][0].op, CXGate) and isinstance(gp[1][1].op, RZGate):
                    rz_node = gp[1][1]
                    cx2_node = gp[1][0]
                    gp1 = next(dag.bfs_successors(rz_node))
                    if cx2_node in gp1[1]:
                        if ((cx1_node.qargs[0].index == cx2_node.qargs[0].index) and
                            (cx1_node.qargs[1].index == cx2_node.qargs[1].index) and
                            (cx2_node.qargs[1].index == rz_node.qargs[0].index)):

                            dag = sub_zz_in_dag(dag, cx1_node, rz_node, cx2_node)

    return dag

def combine_runs(dag: DAGNode, gate_str: str) -> DAGCircuit:
    runs = dag.collect_runs([gate_str])
    for run in runs:
        partition = []
        chunk = []
        for ii in range(len(run)-1):
            chunk.append(run[ii])

            qargs0 = run[ii].qargs
            qargs1 = run[ii+1].qargs

            if qargs0 != qargs1:
                partition.append(chunk)
                chunk = []

        chunk.append(run[-1])
        partition.append(chunk)

        # simplify each chunk in the partition
        for chunk in partition:
            theta = 0
            for ii in range(len(chunk)):
                theta += chunk[ii].op.params[0]

            # set the first chunk to sum of params
            chunk[0].op.params[0] = theta

            # remove remaining chunks if any
            if len(chunk) > 1:
                for nn in chunk[1:]:
                    dag.remove_op_node(nn)
    return dag

def get_avg_gate_error(backend: Backend, initial_layout: List[int]) -> float:
    avg_gate_error = 0
    for ii in range(len(initial_layout)-1):
        q0 = initial_layout[ii]
        q1 = initial_layout[ii+1]
        avg_gate_error += backend.properties().gate_property('cx')[(q0, q1)]['gate_error'][0]

    avg_gate_error /= len(initial_layout)-1
    return avg_gate_error

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
