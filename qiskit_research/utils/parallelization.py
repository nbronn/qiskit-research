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

"""Tools for parallelizing circuits."""

from qiskit.circuit import Instruction
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import CouplingMap, TransformationPass
# from qiskit.transpiler.passes import Trans

def layer_qubit_pairs(coupling_map: CouplingMap) -> List[List[int]]:
    """
    From the coupling map of a physical backend, layer them into lists that can be
    operated on simultaneously (i.e., no overlapping qubits). 
    """
    # create ordered coupling map
    ordered_cm = [pair for pair in [pair for pair in coupling_map if pair[0] < pair[1]]]

    # layer coupling map into entanglement maps of non-overlapping qubits
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

def LayerGateSets(TransformationPass):

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:

        return 0

def CollectGateSets(TransformationPass):

    def __init__(gate_set: list[str]):
        self._gate_set = gate_set

    def get_qubits_from_node(node):
        return [qarg.index for qarg in node.qargs]

    def next_node_in_set(
        curr_node: DAGOpNode, 
        next_node: DAGOpNode,
    ) -> bool:
        if self.get_qubits_from_node(curr_node) == self.get_qubits_from_node(next_node):
            if next_node.op.name in self._gate_set:
                if self._gate_set.index(curr_node.op.name) == self._gate_set.index(next_node.op.name) - 1:
                    return True

        return False

    def collect_gate_sets(dag: DAGCircuit, curr_node: DAGOpNode, block):
        if isinstance(curr_node, DAGOpNode):
            if curr_node.op.name == self._gate_set[0]:
                block.append(curr_node)
                for next_node in dag.successors(curr_node):
                    if self.next_node_in_set(curr_node, next_node):
                        block.append(next_node)
                        if next_node.op.name == self._gate_set[-1]:
                            block_node = Instruction(
                                "+".join(self._gate_set), 
                                num_qubits=len(self._gate_set), 
                                num_clbits=0
                                params=[node.op.params[0] for node in block]
                            )
                            dag.replace_block_with_op(block, block_node, wire_pos_map=get_qubits_from_node(block[0]))
                        else:
                            collect_gate_sets(dag, next_node, block)

                block = []



    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:

        for front_node in dag.front_layer():
            block = []
            collect_gate_sets(front_node, block)
