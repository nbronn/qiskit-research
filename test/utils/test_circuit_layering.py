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

"""Test circuit layering."""

import unittest
import pdb

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import PauliSumOp
from qiskit.providers.fake_provider import FakeKolkata
from qiskit.providers.fake_provider import FakeWashington
from qiskit.transpiler import PassManager

from qiskit_research.utils.circuit_layering import (
    get_entanglement_map,
    ExpandBlockOperators,
    FindBlockTrotterEvolution,
    LayerBlockOperators,
    AddBarriersForGroupOfLayers,
)

from qiskit_research.utils import (
    GetEntanglingMapFromInitLayout,
)


class TestEntanglingMap(unittest.TestCase):
    """Test passes using two different qubit_lengths.
    Use  # Then compare to expected results using https://docs.python.org/3/library/unittest.html.
    """

    def setUp(self):
        """Setup unit test runs before every individual test method."""

        ###########  Use simulated backend.  ###############
        self.backend = FakeWashington()  # 127 qubit device.
        self.config = self.backend.configuration()
        # the_backend_plot = plot_error_map(backend)
        self.the_coupling_map_list = self.config.coupling_map

        self.the_initial_layout_set_wrong = {
            0,
            1,
            2,
            3,
            400,
        }  # This was test for raise error.

        # The list is out of order on purpose for testing.
        self.the_initial_layout_set = [
            20,
            21,
            22,
            23,
            24,
            34,
            43,
            42,
            41,
            40,
            39,
            33,
            19,
            15,
            25,
            44,
            53,
            38,
        ]  # Meaningful list for FakeWashington

        # For index 15
        self.n2_combined_layers_min = [
            [[15, 22], [20, 21], [24, 34], [33, 39], [40, 41], [42, 43]],
            [[19, 20], [21, 22], [23, 24], [34, 43], [39, 40], [41, 42]],
            [[20, 33], [38, 39], [43, 44], [41, 53], [22, 23], [24, 25]],
        ]

        # For index 5
        self.n3_combined_layers_min = [
            [[15, 22], [19, 20], [24, 25], [39, 40], [42, 43]],
            [[20, 21], [23, 24], [38, 39], [41, 53], [43, 44]],
            [[20, 33], [22, 23], [34, 43], [40, 41]],
            [[21, 22], [24, 34], [33, 39], [41, 42]],
        ]

    def tearDown(self):
        """Tie any loose ends after each test."""

    def test_get_entangling_map_2(self, distance: int = 0):
        """Ensure code changes don't change basic output. This is just a sanity check.

        Args:
            qubit_distance (int, optional): Relates to desired distance between pairs.
                        The value of 0 means pairs can be adjacent to each other or further apart.
                        Defaults to 0.
        """

        new_layers = GetEntanglingMapFromInitLayout(
            self.the_coupling_map_list,
            self.the_initial_layout_set,
            distance=distance,
        )
        (
            dict_of_layers_of_pairs,
            unique_layers_of_pairs,
            min_layer_unique_layer_of_pairs,
            combined_layers_min,
        ) = new_layers.pairs_from_n_and_reduced_coupling_map()
        self.assertEqual(len(combined_layers_min), 16)
        self.assertEqual(len(min_layer_unique_layer_of_pairs), 11)
        self.assertEqual(len(unique_layers_of_pairs), 17)
        self.assertEqual(len(dict_of_layers_of_pairs), 18)
        self.assertListEqual(self.n2_combined_layers_min, combined_layers_min[15])

    def test_get_entangling_map_3(self, distance: int = 1):
        """Ensure code changes don't change basic output. This is just a sanity check.

        Args:
            qubit_distance (int, optional): Relates to desired distance between pairs.
                        The value of 1 means pairs can be 1 qubit, or greater apart.
        """

        new_layers = GetEntanglingMapFromInitLayout(
            self.the_coupling_map_list,
            self.the_initial_layout_set,
            distance=distance,
        )
        (
            dict_of_layers_of_pairs,
            unique_layers_of_pairs,
            min_layer_unique_layer_of_pairs,
            combined_layers_min,
        ) = new_layers.pairs_from_n_and_reduced_coupling_map()

        self.assertEqual(len(combined_layers_min), 6)
        self.assertEqual(len(min_layer_unique_layer_of_pairs), 6)
        self.assertEqual(len(unique_layers_of_pairs), 16)
        self.assertEqual(len(dict_of_layers_of_pairs), 18)
        self.assertListEqual(self.n3_combined_layers_min, combined_layers_min[5])

    @unittest.expectedFailure
    def test_bad_coupling_map(self, distance: int = 1):
        """Have code to catch bad input data.
        This case will ask for qubit that is is not in the backend.

        Args:
            distance (int, optional): Desire qubit pairs to be 1 or greater distance apart.
        """
        _ = GetEntanglingMapFromInitLayout(
            self.the_coupling_map_list,
            self.the_initial_layout_set_wrong,
            distance=distance,
        )


# TODO: ceate data for different params & distances
class TestLayeredPauliGates(unittest.TestCase):
    """_summary_

    Args:
        unittest (_type_): For unit testing.
    """

    def test_layered_pauli_evol_gate(self):
        num_qubits = 7
        op = PauliSumOp.from_list(
            [
                ("I" * idx + pair + "I" * (7 - idx - 2), 1)
                for idx in range(num_qubits - 2)
                for pair in ["XX", "YY", "ZZ"]
            ]
        )
        qc = QuantumCircuit(num_qubits)
        qc.append(PauliEvolutionGate(op, 1.3), range(num_qubits))

        block_ops = ["XX", "YY", "ZZ"]
        qc_fbte = PassManager(FindBlockTrotterEvolution(block_ops=block_ops)).run(qc)

        backend = FakeKolkata()
        coupling_map = backend.configuration().coupling_map
        # Can also give a list of "good" well-behaved qubits.
        max_qubits = backend.configuration().num_qubits
        initial_layout = range(max_qubits)
        entanglement_map = get_entanglement_map(
            coupling_map=coupling_map,
            init_layout=initial_layout,
            distance=0,
            ent_map_index=0,
        )
        qc_l = transpile(qc_fbte, coupling_map=coupling_map, seed_transpiler=12345)

        qc_layered = PassManager(
            [
                LayerBlockOperators(
                    block_ops=block_ops, entanglement_map=entanglement_map
                ),
                ExpandBlockOperators(block_ops=block_ops),
                AddBarriersForGroupOfLayers(entanglement_map=entanglement_map),
            ]
        ).run(qc_l)

        qc_unlayered = PassManager(ExpandBlockOperators(block_ops=block_ops)).run(qc_l)

        # pdb.set_trace()
        # qc_layered.draw(idle_wires=False)
        self.assertLess(qc_layered.depth(), qc_unlayered.depth())

    def test_barrier_pauli_evol_gate(self):
        op = PauliSumOp.from_list([[pair, 1] for pair in ["XX", "YY", "ZZ"]])
        layers = [[[0, 1], [4, 5]], [[2, 3], [6, 7]], [[1, 2], [5, 6]], [[3, 4]]]
        qc = QuantumCircuit(8)

        for layer in layers:
            [qc.append(PauliEvolutionGate(op, 0.7), pair) for pair in layer]
            qc.barrier()

        # pdb.set_trace()
        # qc.draw()


if __name__ == "__main__":
    unittest.main(verbosity=2)
