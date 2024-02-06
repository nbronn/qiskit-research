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
import warnings
from typing import Optional
from collections import defaultdict
from collections import deque
from qiskit import BasicAer
from qiskit.providers import Backend, Provider

# To plot each layer for a grouping.
from qiskit.visualization import plot_circuit_layout
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


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


def get_outward_coupling_map(coupling_map, ent_map, start_qubits):
    """
    Recursive method for rearranging a coupling map to generate an entanglement
    map outward from a set of starting qubit(s). The new entanglement map is then
    suitable for use by canned algorithms such as TwoLocal in the entanglement= kwarg.

    Args:
        coupling_map: coupling map of backend
                    (TODO: need to currently deepcopy this before calling method)
        ent_map: rearranged coupling map used for generating entanglement in correct order
                (TODO: currently take empty list)
        start_qubits: qubits to find the pairs containing
                        TODO: currently takes [middle_qubit])
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
        elif all(
            [
                any([pair[0] in epair or pair[1] in epair for epair in emap])
                for emap in ent_map
            ]
        ):
            ent_map.append([pair])
        else:
            for emap in ent_map:
                if all(
                    [pair[0] not in epair and pair[1] not in epair for epair in emap]
                ):
                    emap.append(pair)
                    break

    return ent_map


def convert_list_map_to_dict(list_map: list) -> defaultdict:
    """Reorganize the coupling map since qubits may not be symmetric.

    Args:
        list_map (list): The map obtained from the backend.

    Raises:
        ValueError: Each sublist-pair within coupling_map should be a start and end qubit integers.

    Returns:
        defaultdict: Each key is a start qubit, the value hold a list of qubits that can be
                    be second qubit.  This accounts for if the qubits are non-symmetric.
    """

    if list_map:  # If there is something inside the list_map.
        map_dict = defaultdict(list)
    else:
        warnings.warn("The list_map is empty. No dict will be returned.")
        return None

    for pair in list_map:
        len_sublist = len(pair)
        if len_sublist == 2:
            first_qubit, second_qubit = pair
            map_dict[first_qubit].append(second_qubit)
        else:
            error_string = (
                f"The length of each sublist within list map should contain "
                f"only 2 integers. You have {len_sublist},  for pair: {pair}"
            )
            raise ValueError(error_string)
    return map_dict


def convert_dict_to_list(coupling_map_dict: defaultdict) -> list:
    """Change the format from dict to list of potential qubits.

    Args:
        coupling_map_dict (defaultdict): A dict where the key is fist_qubit, and the value is the
                                        list of potential second qubits.

    Returns:
        list: Reformatted list of qubit pairs.
    """
    coupling_map_list = []
    for first_qubit, second_qubit_list in coupling_map_dict.items():
        for second_qubit in second_qubit_list:
            coupling_map_list.append([first_qubit, second_qubit])

    return coupling_map_list


def matrix_to_dict(
    entangled_result: "np.ndarray",
    sorted_init_layout: list,
) -> dict:
    """Extract qubit ids for when the matrix is equal to 1 and formats the data
    with key=start_qubit with a value(list) of the second_qubit.

    The number in the matrix corresponds to how many paths in the graphs connect
    the two qubits. For a value of 3 then, the qubits could be next to each other
    but you can find paths that double-back to the qubit in question. For this reason,
    it is important to only consider the pairs whose value is 1, along with not being
    on the diagonal.

    Args:
        entangled_result (numpy.ndarray): 2-D square array with each index corresponding to
                                        qubits in the sorted_init_layout.
        sorted_init_layout (list): Sorted list of the user-provided init_layout.  The init_layout,
                                ideally denotes the "good" qubits.

    Returns:
        dict: Using the entangled_result, provide key=start_qubit with a value(list)
            of the second_qubit
    """

    entangling_dict = defaultdict(list)
    for first_index, row in enumerate(entangled_result):
        for second_index, value in enumerate(row):
            if value == 1 and first_index != second_index:
                entangling_dict[sorted_init_layout[first_index]].append(
                    sorted_init_layout[second_index]
                )
    return entangling_dict


class PopulateCouplingMapDictAndMatrixDict:
    """Return entangling_map_dict and reduced_coupling_list."""

    def __init__(self, coupling_map: list, init_layout: list, qubit_distance: int = 2):
        """Prepare the data so that logic to pair the qubits can be implemented.

        Args:
            coupling_map (list): From provider's backend.
            init_layout (list): Qubit_ids which are desired and a subset of available
                                qubits from coupling map.
            qubit_distance (int, optional): Determines exponent for matrix multiplication
                                of the adjacency matrix. Defaults to 2.
        """
        self.coupling_map = coupling_map
        # Used to determine self.the_diff
        self.coupling_set = None
        # coupling_map_dict (defaultdict): Reformatted coupling_map key=first_qubit,
        #                                     value=list of all qubits connected to it.
        self.coupling_map_dict = None

        self.init_layout = init_layout
        # sorted_init_layout (list): Can use the index of the qubits to determine
        #                               layout of matrix axis.
        self.sorted_init_layout = None

        # the_diff (set): Qubits that are within the coupling_map VS init_layout.
        self.the_diff = None

        self.qubit_distance = qubit_distance
        self.populate_coupling_map_dict()

        self.initial_layout_lookup = (
            None  # Denotes the qubits for entangle_result and entangle_matrix.
        )

        # Set up matrix before multiplying to exponent=self.qubit_distance.
        self.entangle_matrix = None

        # Hold intermediate results for matrix multiplication.
        self.list_entangle_results = []

        # Hold result of self.entangle_matrix to the exponent self.qubit_distance
        self.entangled_result = None  # The result of matrix multiplication.

        # Populate a dict to contain the information about the reduced coupling map.
        self.reduced_coupling_map = defaultdict(list)
        self.create_entangle_matrix()
        self.entangling_dict = self._matrix_to_get_entangle_dict()  # defaultdict(list)

    def populate_coupling_map_dict(self):
        """Do some error checking and convert the coupling map to a dict.

        Raises:
            ValueError: User requested a qubit with does not exist in the coupling map.
        """
        self.sorted_init_layout = sorted(self.init_layout)
        # Working on this.
        answer, self.coupling_set = self._confirm_init_layout_qubits_in_coupling_map()
        if not answer:
            error_string = "A request to use qubit which does not exist in backend."
            raise ValueError(error_string)

        self.the_diff = self.coupling_set.symmetric_difference(self.sorted_init_layout)
        self.coupling_map_dict = convert_list_map_to_dict(self.coupling_map)

    def create_entangle_matrix(self):
        """Give an equal or subset of desired qubits denoted in self.init_layout, which should be
        limited by qubits within the coupling map, generate a new dict of entangling qubits.
        The entangling qubits is equal or a subset of available qubits from the self.coupling map
        that are apart by self.qubit_distance.
        """

        if self.the_diff:
            for qubit_id in self.the_diff:
                self.coupling_map_dict.pop(qubit_id)
        # Note: coupling_map_dict has been reduced by init_layout just for ONLY the key.
        # The value will be reduced later and put into reduced_coupling_map.

        # Interim solution, still need to address self.qubit_distance.
        size_of_matrix = len(self.sorted_init_layout)

        # Populate the matrix with zero.
        self.entangle_matrix = np.zeros([size_of_matrix, size_of_matrix], dtype=int)

    def _confirm_init_layout_qubits_in_coupling_map(self) -> tuple[bool, set]:
        """Confirm that init_layout has qubits that are equal or less than the coupling map.

        Returns:
            bool: If self.init_layout has qubits within coupling_map.
            set: The qubits from coupling map.
        """
        cm_set = set()
        il_set = set(self.init_layout)
        for item in self.coupling_map:
            subset = set(item)
            cm_set.update(subset)

        a_subset = il_set.issubset(cm_set)
        if not a_subset:
            message = (
                f"The qubits in init_layout: {il_set} are not in coupling_map: {cm_set}"
            )
            warnings.warn(message)
        return a_subset, cm_set

    def _matrix_to_get_entangle_dict(
        self,
    ) -> dict:
        """Give an equal or subset of desired qubits denoted in self.init_layout,
        which should be limited by qubits within the coupling map,
        generate a new dict of entangling qubits.  The entangling qubits
        is equal or a subset of available qubits from the self.coupling map
        that are apart by self.qubit_distance.

        Returns:
            defaultdict(list): Contains only qubits which are desired from init_layout.
                The list has been sorted by both the first and second qubits pairs.
                Then put desired qubits formatted within a matrix and multiplied
                by self.qubit_distance times. The number in the matrix corresponds to how many
                paths in the graphs connect the two qubits.
                Within the result of matrix multiplication, use the qubits with "1"  entry
                within the matrix, which is not on the diagonal.

        """

        self.initial_layout_lookup = defaultdict(int)

        # For each qubit, denote the index on the matrix axis.
        for index, qubit in enumerate(self.sorted_init_layout):
            self.initial_layout_lookup[qubit] = index

        # For every connection between first and second qubits,
        # fill the connections with value of 1.

        # Sort just the keys of dict which represents the first_qubit of pair.
        for first_qubit, connection in sorted(self.coupling_map_dict.items()):
            # The value is a list of connections for second_qubit, so sort that separately.
            for second_qubit in sorted(connection):
                # Rebuild the reduced list map for
                # qubits that user denoted in self.sorted_init_layout.
                if second_qubit in self.sorted_init_layout:
                    self.entangle_matrix[
                        self.initial_layout_lookup[first_qubit],
                        self.initial_layout_lookup[second_qubit],
                    ] = 1

                    # This dict has both the key and value limited by init_layout.
                    self.reduced_coupling_map[first_qubit].append(second_qubit)

        self._build_intermediate_matrices()
        entangling_dict = matrix_to_dict(self.entangled_result, self.sorted_init_layout)

        return entangling_dict

    def _build_intermediate_matrices(self):
        """Keep a list of intermediate matrices since we need to use the intermediate matrices
        to check if coupling pair is spaced apart as desired."""

        # Index 0
        self.list_entangle_results.append(self.entangle_matrix)

        for each_mult in range(1, self.qubit_distance):
            # Places matrix in index each_mult.
            self.list_entangle_results.append(
                np.matmul(
                    self.list_entangle_results[each_mult - 1], self.entangle_matrix
                )
            )
        # test_to_compare = matrix_power(self.entangle_matrix, self.qubit_distance)
        self.entangled_result = self.list_entangle_results[self.qubit_distance - 1]


class GetEntanglingMapFromInitLayout(PopulateCouplingMapDictAndMatrixDict):
    """Use the parent class to gather the data for sorting. Then this class will sort
    according to desired grouping for layers.
    """

    def __init__(self, coupling_map: list, init_layout: list, distance: int = 0):
        """Pass arguments to parent init and denote variable to hold result of sorting.

        Args:
            coupling_map (list): From provider's backend.
            init_layout (list): Qubit_ids which are desired and a subset of available
                                qubits from coupling map.
            distance (int, optional): How many qubits are guaranteed to be minimally between pairs.
                                This number is connected to the exponent of the adjacency matrix.
                                The exponent for matrix multiplication = distance+2.
                                Defaults to 0.
                                Zero means the pairs can be adjacent to each other.
                                One means there is minimum of 1 qubit between pairs, etc.
        """
        PopulateCouplingMapDictAndMatrixDict.__init__(
            self,
            coupling_map=coupling_map,
            init_layout=init_layout,
            # The distance+2 is used for the exponent for matrix multiplication.
            qubit_distance=distance + 2,
        )
        self.distance = distance
        self.dict_of_layers_of_pairs = defaultdict(lambda: defaultdict(list))
        self.unique_layers_of_pairs = []  # Will be a lists of lists which is sorted.
        # Will be list of lists with minimum number of layers.
        self.min_layer_unique_layer_of_pairs = []
        self.combined_layers = []
        self.combined_layers_min = []
        self.temp_sorted_by_len = []
        # Keeps track of the smallest len within self.temp_sorted_by_len
        self.min_number_compress = None

        self.reduced_coupling_list_to_del = None
        self.tally_used_qubits = []
        self.min_number_of_layers = None

    def pairs_from_n_and_reduced_coupling_map(self):
        """By using the reduced_coupling_map to look at qubit_distance within self.entangling_dict,
        create list of layers.  Each layer is a grouping of tuples of qubit pairs.
        A grouping of layers is made by starting with each qubit identified from
        the reduced_coupling_map.

        Returns:

            defaultdict[defaultdict]]:  self.dict_of_layers_of_pairs gets populated now.
                                        The key is the qubit which is used
                                        to start the search.
                                        There is a second key which denotes the unused
                                        pairs within the search.
            list[list(tuple)]: Each sublist are lists of pairs which are not repeated.
                                Each sublist should be unique.
                                Each sublist should have number of layers to be the minimum.
        """

        reduced_coupling_deque = deque(self.reduced_coupling_map.items())
        self.min_number_of_layers = None

        # Rotate reduced_coupling_deque until going through each qubit.  Each start-qubit can
        # give potentially a different set of groupings.
        for index in range(len(reduced_coupling_deque)):
            reduced_coupling_deque.rotate(1)  # Rotate just once for each index.
            key = reduced_coupling_deque[0][0]
            list_of_layers_of_pairs = []
            tally_used_pairs = []

            self.reduced_coupling_list_to_del = convert_dict_to_list(
                self.reduced_coupling_map
            )
            for first_qubit, second_qubit_list in reduced_coupling_deque:
                for second_qubit in second_qubit_list:
                    grouping_pair = []

                    grouping_pair = self.handle_pair(
                        first_qubit, second_qubit, grouping_pair
                    )
                    if grouping_pair:
                        grouping_pair.sort()
                        # If grouping_pair not in list_of_layers_of_pairs:
                        list_of_layers_of_pairs.append(grouping_pair)
                        tally_used_pairs.extend(grouping_pair)

                        # Reset after each grouping.
                        self.tally_used_qubits = []

                        # Need to rebuild a further reduced_coupling_map by tally_used_pairs.
                        self.reduced_coupling_list_to_del = convert_dict_to_list(
                            self.reduced_coupling_map
                        )
                        self.reduced_coupling_list_to_del = [
                            [q1, q2]
                            for q1, q2 in self.reduced_coupling_list_to_del
                            if not (
                                [q1, q2] in tally_used_pairs
                                or [q2, q1] in tally_used_pairs
                            )
                        ]

            self.dict_of_layers_of_pairs[key]["useful"] = list_of_layers_of_pairs

            # Gather the information if there were any unused pairs.
            self.dict_of_layers_of_pairs[key][
                "unused"
            ] = self.reduced_coupling_list_to_del
            len_of_listoflayersofpairs = len(list_of_layers_of_pairs)
            # Check to see if the list_of_layer_pairs are unique and
            # gather them in self.unique_layers_of_pairs.
            if list_of_layers_of_pairs:
                if list_of_layers_of_pairs not in self.unique_layers_of_pairs:
                    self.unique_layers_of_pairs.append(list_of_layers_of_pairs)
                    if not self.min_number_of_layers:
                        self.min_number_of_layers = len_of_listoflayersofpairs
                    elif self.min_number_of_layers > len_of_listoflayersofpairs:
                        self.min_number_of_layers = len_of_listoflayersofpairs
                else:
                    pass
                    # a = 5  # For breakpoint

            # Each set of layers should be at the minimum amount, or we don't want it.
            # Within self.combine_layers_populate() we remove any copies when determining
            # self.combined_layers_min.  So, thinking no-one will want to use
            # self.min_layer_unique_layer_of_pairs for other than looking at
            # what happens along the way of getting self.combined_layers_min.
            self.min_layer_unique_layer_of_pairs = [
                layer_set
                for layer_set in self.unique_layers_of_pairs
                if len(layer_set) == self.min_number_of_layers
            ]

        self.combine_layers_populate()

        return (
            self.dict_of_layers_of_pairs,
            self.unique_layers_of_pairs,
            self.min_layer_unique_layer_of_pairs,
            self.combined_layers_min,
        )

    def handle_pair(
        self, first_qubit: int, second_qubit: int, grouping_pair: list
    ) -> list:
        """This is the beginning of a recursive algorithm. Return a list of pair that are n
        away from the first and second qubit. Continue searching through all the options
        and then return a group of pairs.

        Args:
            first_qubit (int): A qubit in a pair from the reduced_coupling_map.
            second_qubit (int): Another qubit in a pair from the reduced_coupling_map.
            grouping_pair (list): Holds a grouping of pairs for which are n away from each other.
                                The pairs are sorted when entered into this link.
                                This is so we can compare with other sets that are
                                created with different start qubits.

            Returns:
            list: List of lists which are a pair that are n away from each other.
        """
        a_pair = [first_qubit, second_qubit]
        a_pair_flipped = [second_qubit, first_qubit]
        a_pair_sorted = sorted(a_pair)
        n_away_first_qubit_list = self.entangling_dict[first_qubit]
        n_away_second_qubit_list = self.entangling_dict[second_qubit]

        # find the pairs from reduced coupling map that are n away from each other.
        if (
            a_pair_flipped in self.reduced_coupling_list_to_del
            or a_pair in self.reduced_coupling_list_to_del
        ):
            if (
                self.qubit_distance > 2
                and self.check_for_adjacent_qubits(
                    grouping_pair, first_qubit, second_qubit
                )
                or self.qubit_distance == 2
            ):
                grouping_pair.append(a_pair_sorted)
                self.tally_used_qubits.append(first_qubit)
                self.tally_used_qubits.append(second_qubit)
                # Reduce to a new list without qubits used for a_pair_sorted
                self.reduced_coupling_list_to_del = [
                    [q1, q2]
                    for q1, q2 in self.reduced_coupling_list_to_del
                    if not (
                        first_qubit == q1
                        or first_qubit == q2
                        or second_qubit == q1
                        or second_qubit == q2
                    )
                ]

        # If the qubit has already been used, no point is testing pairs with it.
        if self.tally_used_qubits:
            n_away_first_qubit_list = [
                q1 for q1 in n_away_first_qubit_list if not q1 in self.tally_used_qubits
            ]
            n_away_second_qubit_list = [
                q2
                for q2 in n_away_second_qubit_list
                if not q2 in self.tally_used_qubits
            ]

        if a_pair_sorted in grouping_pair or a_pair in grouping_pair:
            for qubit_start in n_away_first_qubit_list:
                for qubit_test in n_away_second_qubit_list:
                    a_pair_test = [qubit_start, qubit_test]
                    a_pair_test_flipped = [qubit_test, qubit_start]
                    a_pair_test_sorted = sorted(a_pair_test)

                    if (
                        a_pair_test_flipped in self.reduced_coupling_list_to_del
                        or a_pair_test in self.reduced_coupling_list_to_del
                    ):
                        if (
                            self.qubit_distance > 2
                            and self.check_for_adjacent_qubits(
                                grouping_pair, qubit_start, qubit_test
                            )
                            or self.qubit_distance == 2
                        ):
                            grouping_pair.append(a_pair_test_sorted)
                            self.tally_used_qubits.append(qubit_start)
                            self.tally_used_qubits.append(qubit_test)

                            # Reduce to a new list without qubits used for a_pair
                            self.reduced_coupling_list_to_del = [
                                [q1, q2]
                                for (q1, q2) in self.reduced_coupling_list_to_del
                                if not (
                                    qubit_start == q1
                                    or qubit_start == q2
                                    or qubit_test == q1
                                    or qubit_test == q2
                                )
                            ]

                            # Need the if statement for the last case of double for loops.
                            if a_pair_sorted in grouping_pair:
                                grouping_pair = self.handle_pair(
                                    qubit_start, qubit_test, grouping_pair
                                )

                            break

        return grouping_pair

    def check_for_adjacent_qubits(self, grouping_pair: list, q1: int, q2: int) -> bool:
        """This is a check when creating layers and self.qubit_distance > 2.
        Since we are searching on all possible directions, we need to confirm we don't
        select an adjacent qubit based on self.qubit_distance

        Args:
            grouping_pair (list): The layer which (q1,q2) could be placed in.
            q1 (int): One qubit of the coupling pair.
            q2 (int): Second qubit of the coupling pair.

        Returns:
            bool: This only give a true or false after checking the matrix of coupling pairs.
        """
        if not grouping_pair:
            # Since grouping pair is empty, add the pair to grouping pair.
            return True

        for index in range(self.qubit_distance - 2):
            # Check for q1.
            for used_qubit in self.tally_used_qubits:
                if (
                    self.list_entangle_results[index][
                        self.initial_layout_lookup[used_qubit],
                        self.initial_layout_lookup[q1],
                    ]
                    == 1
                ):
                    return False

            # Check for q2.
            for used_qubit in self.tally_used_qubits:
                if (
                    self.list_entangle_results[index][
                        self.initial_layout_lookup[used_qubit],
                        self.initial_layout_lookup[q2],
                    ]
                    == 1
                ):
                    return False

        return True

    def can_not_combine_layers(self, start_with_qubits: set, last_qubits: set) -> bool:
        """Are qubits being too close to each other between the two lists?  If they are too close,
        they can not be combined.

        Args:
            start_with_qubits (set): Qubits from one layer
            last_qubits (set): Qubits from a different layer to compare with.

        Returns:
            bool:   True if qubits are too close.
                    False if the two sets are islands with enough distance.
        """
        # Check every matrix that for spacing.
        #  If too close, they can not be combined.
        for index in range(self.qubit_distance - 2):
            for start_qubit in start_with_qubits:
                for last_qubit in last_qubits:
                    if (
                        self.list_entangle_results[index][
                            self.initial_layout_lookup[start_qubit],
                            self.initial_layout_lookup[last_qubit],
                        ]
                        == 1
                    ):
                        return True

        return False

    def combine_layers_populate(self):
        """Given a set of layers, one could combine the layers; like combining islands.
        Use self.unique_layers_of_pairs.  For each set, if possible, compress them.
        Keep track of the size of sets and keep the smallest sized sets.
        Also, remove any duplicate sets.
        """

        for set_layers in self.unique_layers_of_pairs:
            self.temp_sorted_by_len = sorted(set_layers, key=len, reverse=True)

            len_list = len(self.temp_sorted_by_len)

            # put the compressed list into self.temp_sorted_by_len
            if self.qubit_distance == 2:
                self.n2_compress(check_index=len_list - 1, length=len_list)
            else:
                self.matrix_compress(check_index=len_list - 1, length=len_list)

            len_compressed = len(self.temp_sorted_by_len)
            if not self.min_number_compress:
                # This is for the first time through.
                self.min_number_compress = len_compressed
            elif self.min_number_compress > len_compressed:
                self.min_number_compress = len_compressed

            if self.temp_sorted_by_len not in self.combined_layers:
                self.combined_layers.append(self.temp_sorted_by_len)

        # Reduce self.combined_layers to have len of self.min_number_compress only.
        self.combined_layers_min = [
            layer_set
            for layer_set in self.combined_layers
            if len(layer_set) == self.min_number_compress
        ]

    def matrix_compress(self, check_index: int, length: int):
        """Use this to see if the layers can be combined using the result of matrix multiplication
        based on self.qubit_distance.

        Args:
            check_index (int): If compressed, the size of set changes.
                            Pass updated information to next recursive call.
            length (int): If compressed, the size of set changes.
                            Pass updated information to next recursive call.
        """
        for compare_index in reversed(range(check_index + 1)):
            last_qubits = set()
            # Gather qubits for the last layer which is compared to layers before it.
            for q3, q4 in self.temp_sorted_by_len[compare_index]:
                last_qubits.add(q3)
                last_qubits.add(q4)

            for start_index in range(compare_index):
                start_with_qubits = set()
                # Gather the qubits for start layer.
                for q1, q2 in self.temp_sorted_by_len[start_index]:
                    start_with_qubits.add(q1)
                    start_with_qubits.add(q2)

                # Check matrixes for each pair comparison.

                if start_with_qubits.intersection(last_qubits):
                    # There is/are overlap of qubit(s).
                    pass

                elif self.can_not_combine_layers(start_with_qubits, last_qubits):
                    # They  qubits are too close to each other, so can not combine layers.
                    pass

                else:
                    # Qubits in the two layers do not overlap so can add the two layers.
                    # Also, remove one of the layers.
                    self.temp_sorted_by_len[start_index] = (
                        self.temp_sorted_by_len[start_index]
                        + self.temp_sorted_by_len[compare_index]
                    )
                    del self.temp_sorted_by_len[compare_index]

                    self.matrix_compress(check_index=length - 2, length=length - 1)

                    # Iterating anymore would be working wrong self.temp_sorted_by_len.
                    return

    def n2_compress(self, check_index: int, length: int):
        """This is meant to be used recursively for each set of layers.
        Sort each layer from largest to smallest.  Then look at the smallest and
        sequentially check the other layers to see if they can be combined.  If so, then
        restart the method again.

        Args:
            check_index (int): If compressed, the size of set changes.
                            Pass updated information to next recursive call.
            length (int): If compressed, the size of set changes.
                            Pass updated information to next recursive call.
        """
        for compare_index in reversed(range(check_index + 1)):
            for start_index in range(compare_index):
                start_with_qubits = set()
                last_qubits = set()

                for q1, q2 in self.temp_sorted_by_len[start_index]:
                    start_with_qubits.add(q1)
                    start_with_qubits.add(q2)
                for q3, q4 in self.temp_sorted_by_len[compare_index]:
                    last_qubits.add(q3)
                    last_qubits.add(q4)

                if start_with_qubits.intersection(last_qubits):
                    # There are overlap of qubits.
                    pass
                else:
                    # Qubits in the two layers do not overlap so can add the two layers.
                    # Also, remove one of the layers.
                    self.temp_sorted_by_len[start_index] = (
                        self.temp_sorted_by_len[start_index]
                        + self.temp_sorted_by_len[compare_index]
                    )
                    del self.temp_sorted_by_len[compare_index]

                    self.n2_compress(check_index=length - 2, length=length - 1)

                    # Iterating anymore would be working wrong self.temp_sorted_by_len.
                    return
        return


class PlotLayerData:
    """Visualize the layers for Ansatz."""

    def __init__(self, backend) -> None:
        """Provide the backend.

        Args:
            backend (_type_): _description_
        """
        self.backend = backend
        self.config = backend.configuration()
        self.n_qubits = self.config.n_qubits
        self.init_layout = list(range(self.n_qubits))
        self.list_of_layer_plots = []

    def plot_one_set_of_layers(self, a_set: list) -> list:
        """Give a set of layers.  A plot for each layer
        using self.backend will be generated.

        Args:
            a_set (list): List of lists.  Each list has tuple of coupling pair.

        Returns:
            list: Reference to each plot within the list.
        """

        for layer in a_set:
            layer_qubits = []
            for q1, q2 in layer:
                layer_qubits.append(q1)
                layer_qubits.append(q2)

            n_layer_qubits = len(layer_qubits)
            qc = QuantumCircuit(n_layer_qubits)

            idx_0 = 0
            idx_1 = 1
            for q1, q2 in layer:
                qc.cx(idx_0, idx_1)
                idx_0 += 2
                idx_1 += 2

            new_qc_lv3 = transpile(
                qc,
                initial_layout=layer_qubits,
                backend=self.backend,
                optimization_level=0,
            )

            a_plot = plot_circuit_layout(
                new_qc_lv3, backend=self.backend, view="physical"
            )
            self.list_of_layer_plots.append(a_plot)

        return self.list_of_layer_plots
