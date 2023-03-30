# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Transpiler Passes for Circuit Layering."""

from qiskit.circuit import Instruction, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate, RXXGate, RYYGate, RZZGate
from qiskit.transpiler import CouplingMap, TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import Pauli

import numpy as np
from typing import List, Tuple

class FindBlockTrotterEvolution(TransformationPass):
    def __init__(
        self, 
        block_ops: List[str] = None,
    ):
        super().__init__()
        self._block_ops = block_ops
    
    def run(self, dag: DAGCircuit):
        for node in dag.op_nodes(): # let's take in PauliTrotterEvolutionGates to start
            if isinstance(node.op, PauliEvolutionGate):
                dag = self._decompose_to_block_ops(dag, node)
    
        return dag
                                
    def _decompose_to_block_ops(self, dag: DAGCircuit, node: DAGOpNode) -> DAGCircuit:
        """Decompose the PauliSumOp into two-qubit.
        Args:
            dag: The dag needed to get access to qubits.
            op: The operator with all the Pauli terms we need to apply.
        Returns:
            A dag made of two-qubit :class:`.PauliEvolutionGate`.
        """
        sub_dag = dag.copy_empty_like()
        required_paulis = {self._pauli_to_edge(pauli): {} for pauli in node.op.operator.paulis}
        for pauli, coeff in zip(node.op.operator.paulis, node.op.operator.coeffs):
            required_paulis[self._pauli_to_edge(pauli)][pauli] = coeff
        for edge, pauli_dict in required_paulis.items():         
            params = np.zeros(len(self._block_ops), dtype=object)
            for pauli, coeff in pauli_dict.items():
                qubits = [dag.qubits[edge[0]], dag.qubits[edge[1]]]
                for pidx, pstr in enumerate(self._block_ops):
                    if pauli.to_label().replace("I", "") == pstr:
                        params[pidx] = node.op.time * coeff
            block_op = Instruction('xx+yy+zz', num_qubits=2, num_clbits=0, params=params)
            sub_dag.apply_operation_back(block_op, qubits)
                
        dag.substitute_node_with_dag(node, sub_dag)

        return dag
    
    @staticmethod
    def _pauli_to_edge(pauli: Pauli) -> Tuple[int, ...]:
        """Convert a pauli to an edge.
        Args:
            pauli: A pauli that is converted to a string to find out where non-identity
                Paulis are.
        Returns:
            A tuple representing where the Paulis are. For example, the Pauli "IZIZ" will
            return (0, 2) since virtual qubits 0 and 2 interact.
        Raises:
            QiskitError: If the pauli does not exactly have two non-identity terms.
        """
        edge = tuple(np.logical_or(pauli.x, pauli.z).nonzero()[0])

        if len(edge) != 2:
            raise QiskitError(f"{pauli} does not have length two.")

        return edge


class LayerBlockOperators(TransformationPass):
    def __init__(
        self, 
        block_ops: List[str] = None,
        coupling_map: CouplingMap = None,
    ):
        super().__init__()
        self._block_ops = block_ops
        self._block_str = "+".join(block_ops).lower()
        self._coupling_map = coupling_map
        self._ent_map = self._get_entanglement_map()
        
    def run(self, dag: DAGCircuit):
        for front_node in dag.front_layer():
            self._find_consecutive_block_nodes(dag, front_node)
            
        return dag
    
    def _find_consecutive_block_nodes(self, dag, node0):
        for node1 in dag.successors(node0):
            if isinstance(node1, DAGOpNode):
                self._find_consecutive_block_nodes(dag, node1)
                if node1.op.name == self._block_str:
                    if node0.op.name == self._block_str:
                        self._layer_block_op_nodes(dag, node0, node1)
    
    def _get_entanglement_map(self) -> List[List[int]]:
        
        ent_map = [pair for pair in [pair for pair in self._coupling_map if pair[0] < pair[1]]]

        # split entanglement map into sets of non-overlapping qubits
        ent_maps = []
        for pair in ent_map:
            if ent_maps == []:
                ent_maps.append([pair])
            elif all([any([pair[0] in epair or pair[1] in epair for epair in emap]) for emap in ent_maps]):
                ent_maps.append([pair])
            else:
                for emap in ent_maps:
                    if all([pair[0] not in epair and pair[1] not in epair for epair in emap]):
                        emap.append(pair)
                        break
        
        return ent_maps
    
    @staticmethod
    def _get_pair_from_node(node):
        return [node.qargs[0].index, node.qargs[1].index]

    @staticmethod
    def _get_layer_index(pair, ent_maps):
        for lidx, ent_map in enumerate(ent_maps):
            if pair in ent_map or list(reversed(pair)) in ent_map:
                return lidx

    @staticmethod
    def _get_ordered_qreg(pair0, pair1):
        if pair0[0] in pair1:
            q1 = pair0[0]
            q0 = pair0[1]
        else:
            q1 = pair0[1]
            q0 = pair0[0]

        if q1 == pair1[0]:
            q2 = pair1[1]
        else:
            q2 = pair1[0]
        return (q0, q1, q2)
    
    def _layer_block_op_nodes(self, dag, node0, node1):
        pair0 = self._get_pair_from_node(node0)
        lidx0 = self._get_layer_index(pair0, self._ent_map)
        pair1 = self._get_pair_from_node(node1)
        lidx1 = self._get_layer_index(pair1, self._ent_map)
        
        if lidx0 < lidx1:
            return dag
        elif lidx1 < lidx0:
            mini_dag = DAGCircuit()
            qr = QuantumRegister(3, 'q_{md}')
            mini_dag.add_qreg(qr)

            (q0, q1, q2) = self._get_ordered_qreg(pair0, pair1)

            qargs = list(set(node0.qargs+node1.qargs)) # should share exactly one qubit
            qreg = qargs[0].register  
            
            mini_dag.apply_operation_back(node1.op, [qr[2], qr[1]])
            mini_dag.apply_operation_back(node0.op, [qr[0], qr[1]])

            fake_op = Instruction("commutings blocks", num_qubits=3, num_clbits=0, params=[])
            new_node = dag.replace_block_with_op([node0, node1], fake_op, wire_pos_map={qargs[0]: q0, qargs[1]: q1, qargs[2]: q2})
            dag.substitute_node_with_dag(new_node, mini_dag, wires={qr[0]: qreg[q0], qr[1]: qreg[q1], qr[2]: qreg[q2]})


class ExpandBlockOperators(TransformationPass):
    def __init__(
        self, 
        block_ops: List[str] = None,
    ):
        super().__init__()
        self._block_ops = block_ops
        self._block_str = "+".join(block_ops).lower()

    def run(self, dag: DAGCircuit):
        for node in dag.op_nodes(): 
            if node.op.name == self._block_str:
                dag = self._expand_block_ops(dag, node)
    
        return dag
    
    def _expand_block_ops(self, dag, node):
        mini_dag = DAGCircuit()
        qr = QuantumRegister(2)
        mini_dag.add_qreg(qr)

        for oidx, op in enumerate(self._block_ops):
            if op == 'XX':
                mini_dag.apply_operation_back(RXXGate(node.op.params[oidx]), [qr[0], qr[1]])
            elif op == 'YY':
                mini_dag.apply_operation_back(RYYGate(node.op.params[oidx]), [qr[0], qr[1]])
            elif op == 'ZZ':
                mini_dag.apply_operation_back(RZZGate(node.op.params[oidx]), [qr[0], qr[1]])

        dag.substitute_node_with_dag(node, mini_dag)

        return dag