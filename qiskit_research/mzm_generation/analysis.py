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

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple, Union

import mthree
import numpy as np
from matplotlib.figure import Figure
from qiskit_experiments.framework import (
    AnalysisResultData,
    BaseAnalysis,
    ExperimentData,
)
from qiskit_research.mzm_generation.experiment import (
    CircuitParameters,
    KitaevHamiltonianExperiment,
)
from qiskit_research.mzm_generation.utils import (
    compute_correlation_matrix,
    compute_parity,
    counts_to_quasis,
    edge_correlation_op,
    expectation_from_correlation_matrix,
    fidelity_witness,
    kitaev_hamiltonian,
    number_op,
    post_select_quasis,
    purify_idempotent_matrix,
)

if TYPE_CHECKING:
    from mthree.classes import QuasiDistribution
    from qiskit_research.mzm_generation.utils import _CovarianceDict


class KitaevHamiltonianAnalysis(BaseAnalysis):
    "Analyze Kitaev Hamiltonian experimental data."

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List[Figure]]:
        # reconstruct experiment
        experiment_id = experiment_data.metadata["experiment_id"]
        qubits = experiment_data.metadata["qubits"]
        tunneling_values = experiment_data.metadata["tunneling_values"]
        superconducting_values = experiment_data.metadata["superconducting_values"]
        chemical_potential_values = experiment_data.metadata[
            "chemical_potential_values"
        ]
        occupied_orbitals_list = [
            tuple(occupied_orbitals)
            for occupied_orbitals in experiment_data.metadata["occupied_orbitals_list"]
        ]
        experiment = KitaevHamiltonianExperiment(
            experiment_id=experiment_id,
            qubits=qubits,
            tunneling_values=tunneling_values,
            superconducting_values=superconducting_values,
            chemical_potential_values=chemical_potential_values,
            occupied_orbitals_list=occupied_orbitals_list,
        )

        # put data into dictionary for easier handling
        data = {}
        for result in experiment_data.data():
            (
                tunneling,
                superconducting,
                chemical_potential,
                occupied_orbitals,
                permutation,
                measurement_label,
            ) = result["metadata"]["params"]
            params = CircuitParameters(
                tunneling=tunneling,
                superconducting=superconducting
                if isinstance(superconducting, float)
                else complex(*superconducting),
                chemical_potential=chemical_potential,
                occupied_orbitals=tuple(occupied_orbitals),
                permutation=tuple(permutation),
                measurement_label=measurement_label,
            )
            data[params] = result

        # load readout calibration
        mit = mthree.M3Mitigation()
        mit.cals_from_file(
            os.path.join("data", experiment.experiment_id, f"readout_calibration.json")
        )

        # get results
        results = list(self._compute_analysis_results(experiment, data, mit))
        return results, []

    def _compute_analysis_results(
        self,
        experiment: KitaevHamiltonianExperiment,
        data: Dict[CircuitParameters, Dict],
        mit: mthree.M3Mitigation,
    ) -> Iterable[AnalysisResultData]:
        # fix tunneling and superconducting
        tunneling = -1.0
        superconducting = 1.0

        # get simulation results
        yield from self._compute_simulation_results(
            tunneling, superconducting, experiment
        )

        # create data storage objects
        corr_raw = {}
        corr_mem = {}
        corr_ps = {}
        corr_pur = {}
        quasi_dists_raw = {}
        quasi_dists_mem = {}
        quasi_dists_ps = {}
        ps_removed_masses = {}

        # calculate results
        for chemical_potential in experiment.chemical_potential_values:
            # create Hamiltonian
            hamiltonian_quad = kitaev_hamiltonian(
                experiment.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # diagonalize
            (
                transformation_matrix,
                _,
                _,
            ) = hamiltonian_quad.diagonalizing_bogoliubov_transform()
            # compute parity
            W1 = transformation_matrix[:, : experiment.n_modes]
            W2 = transformation_matrix[:, experiment.n_modes :]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            hamiltonian_parity = np.sign(
                np.real(np.linalg.det(full_transformation_matrix))
            )
            # compute quasis and correlation matrices
            for occupied_orbitals in experiment.occupied_orbitals_list:
                exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity
                quasis_raw = {}  # Dict[Tuple[Tuple[int, ...], str], QuasiDistribution]
                quasis_mem = {}  # Dict[Tuple[Tuple[int, ...], str], QuasiDistribution]
                quasis_ps = {}  # Dict[Tuple[Tuple[int, ...], str], QuasiDistribution]
                ps_removed_mass = {}  # Dict[Tuple[Tuple[int, ...], str], float]
                for permutation, label in experiment.measurement_labels():
                    params = CircuitParameters(
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        permutation,
                        label,
                    )
                    counts = data[params]["counts"]
                    # raw quasis
                    quasis_raw[permutation, label] = counts_to_quasis(counts)
                    # measurement error mitigation
                    quasis_mem[permutation, label] = mit.apply_correction(
                        counts,
                        experiment.qubits,
                        return_mitigation_overhead=True,
                    )
                    # post-selection
                    new_quasis, removed_mass = post_select_quasis(
                        quasis_mem[permutation, label],
                        lambda bitstring: (-1) ** sum(1 for b in bitstring if b == "1")
                        == exact_parity,
                    )
                    quasis_ps[permutation, label] = new_quasis
                    ps_removed_mass[permutation, label] = removed_mass
                # save data
                quasi_dists_raw[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ] = quasis_raw
                quasi_dists_mem[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ] = quasis_mem
                quasi_dists_ps[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ] = quasis_ps
                ps_removed_masses[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ] = ps_removed_mass
                # compute correlation matrices
                corr_raw[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ] = compute_correlation_matrix(quasis_raw, experiment)
                corr_mem[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ] = compute_correlation_matrix(quasis_mem, experiment)
                corr_mat_ps, cov_ps = compute_correlation_matrix(quasis_ps, experiment)
                corr_ps[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ] = (corr_mat_ps, cov_ps)
                corr_pur[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ] = (purify_idempotent_matrix(corr_mat_ps), cov_ps)

        yield from self._compute_fidelity_witness(
            "raw",
            corr_raw,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_fidelity_witness(
            "mem",
            corr_mem,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_fidelity_witness(
            "ps",
            corr_ps,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_fidelity_witness(
            "pur",
            corr_pur,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )

        yield from self._compute_energy(
            "raw",
            corr_raw,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_energy(
            "mem",
            corr_mem,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_energy(
            "ps",
            corr_ps,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_energy(
            "pur",
            corr_pur,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_edge_correlation(
            "raw",
            corr_raw,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_edge_correlation(
            "mem",
            corr_mem,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_edge_correlation(
            "ps",
            corr_ps,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_edge_correlation(
            "pur",
            corr_pur,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_number(
            "raw",
            corr_raw,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_number(
            "mem",
            corr_mem,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_number(
            "ps",
            corr_ps,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_number(
            "pur",
            corr_pur,
            experiment.n_modes,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_parity(
            "raw",
            quasi_dists_raw,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_parity(
            "mem",
            quasi_dists_mem,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )
        yield from self._compute_parity(
            "ps",
            quasi_dists_ps,
            tunneling,
            superconducting,
            experiment.chemical_potential_values,
            experiment.occupied_orbitals_list,
        )

    def _compute_simulation_results(
        self,
        tunneling: float,
        superconducting: Union[float, complex],
        experiment: KitaevHamiltonianExperiment,
    ) -> Iterable[AnalysisResultData]:
        # set chemical potential values to the experiment range but with fixed resolution
        chemical_potential_values = np.linspace(
            experiment.chemical_potential_values[0],
            experiment.chemical_potential_values[-1],
            num=50,
        )

        # construct operators
        edge_correlation = edge_correlation_op(experiment.n_modes)
        number = number_op(experiment.n_modes)

        # create data storage objects
        energy_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        edge_correlation_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        parity_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        number_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]

        for chemical_potential in chemical_potential_values:
            # create Hamiltonian
            hamiltonian_quad = kitaev_hamiltonian(
                experiment.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # compute energy
            (
                transformation_matrix,
                orbital_energies,
                constant,
            ) = hamiltonian_quad.diagonalizing_bogoliubov_transform()
            energy_shift = -0.5 * np.sum(orbital_energies) - constant
            # compute parity
            W1 = transformation_matrix[:, : experiment.n_modes]
            W2 = transformation_matrix[:, experiment.n_modes :]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            hamiltonian_parity = np.sign(
                np.real(np.linalg.det(full_transformation_matrix))
            )
            # compute results
            for occupied_orbitals in experiment.occupied_orbitals_list:
                # compute exact correlation matrix
                occupation = np.zeros(experiment.n_modes)
                occupation[list(occupied_orbitals)] = 1.0
                corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
                corr_exact = (
                    full_transformation_matrix.T.conj()
                    @ corr_diag
                    @ full_transformation_matrix
                )
                # exact values
                exact_energy = (
                    np.sum(orbital_energies[list(occupied_orbitals)]) + constant
                )
                exact_edge_correlation, _ = np.real(
                    expectation_from_correlation_matrix(edge_correlation, corr_exact)
                )
                exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity
                exact_number = np.real(
                    np.sum(np.diag(corr_exact)[: experiment.n_modes])
                )
                # add computed values to data storage objects
                energy_exact[occupied_orbitals].append(exact_energy + energy_shift)
                edge_correlation_exact[occupied_orbitals].append(exact_edge_correlation)
                parity_exact[occupied_orbitals].append(exact_parity)
                number_exact[occupied_orbitals].append(exact_number)

        def zip_dict(d):
            return {k: (np.array(v), chemical_potential_values) for k, v in d.items()}

        yield AnalysisResultData("energy_exact", zip_dict(energy_exact))
        yield AnalysisResultData(
            "edge_correlation_exact", zip_dict(edge_correlation_exact)
        )
        yield AnalysisResultData("parity_exact", zip_dict(parity_exact))
        yield AnalysisResultData("number_exact", zip_dict(number_exact))

    def _compute_fidelity_witness(
        self,
        label: str,
        corr: Dict[
            Tuple[int, float, Union[float, complex], Tuple[int, ...]],
            Tuple[np.ndarray, "_CovarianceDict"],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[Tuple[int, ...]],
    ) -> Iterable[AnalysisResultData]:
        data = defaultdict(list)  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        for chemical_potential in chemical_potential_values:
            # create Hamiltonian
            hamiltonian_quad = kitaev_hamiltonian(
                n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # diagonalize
            (
                transformation_matrix,
                _,
                _,
            ) = hamiltonian_quad.diagonalizing_bogoliubov_transform()
            W1 = transformation_matrix[:, :n_modes]
            W2 = transformation_matrix[:, n_modes:]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            for occupied_orbitals in occupied_orbitals_list:
                # compute exact correlation matrix
                occupation = np.zeros(n_modes)
                occupation[list(occupied_orbitals)] = 1.0
                corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
                corr_exact = (
                    full_transformation_matrix.T.conj()
                    @ corr_diag
                    @ full_transformation_matrix
                )
                corr_mat, cov = corr[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ]
                fidelity_wit, stddev = fidelity_witness(corr_mat, corr_exact, cov)
                data[occupied_orbitals].append(
                    (
                        fidelity_wit,
                        stddev,
                    )
                )
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"fidelity_witness_{label}", data_zipped)

        fidelity_witness_avg = np.zeros(len(chemical_potential_values))
        fidelity_witness_stddev = np.zeros(len(chemical_potential_values))
        for occupied_orbitals in occupied_orbitals_list:
            values, stddevs = data_zipped[occupied_orbitals]
            fidelity_witness_avg += np.array(values)
            fidelity_witness_stddev += np.array(stddevs) ** 2
        fidelity_witness_avg /= len(occupied_orbitals_list)
        fidelity_witness_stddev = np.sqrt(fidelity_witness_stddev) / len(
            occupied_orbitals_list
        )
        yield AnalysisResultData(
            f"fidelity_witness_avg_{label}",
            (fidelity_witness_avg, fidelity_witness_stddev),
        )

    def _compute_energy(
        self,
        label: str,
        corr: Dict[
            Tuple[int, float, Union[float, complex], Tuple[int, ...]],
            Tuple[np.ndarray, "_CovarianceDict"],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[Tuple[int, ...]],
    ) -> Iterable[AnalysisResultData]:
        energy_exact = defaultdict(list)  # Dict[Tuple[int, ...], List[float]]
        data = defaultdict(list)  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        for chemical_potential in chemical_potential_values:
            # create Hamiltonian
            hamiltonian_quad = kitaev_hamiltonian(
                n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # diagonalize
            (
                _,
                orbital_energies,
                constant,
            ) = hamiltonian_quad.diagonalizing_bogoliubov_transform()
            energy_shift = -0.5 * np.sum(orbital_energies) - constant
            for occupied_orbitals in occupied_orbitals_list:
                exact_energy = (
                    np.sum(orbital_energies[list(occupied_orbitals)]) + constant
                )
                energy_exact[occupied_orbitals].append(exact_energy + energy_shift)

                corr_mat, cov = corr[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ]
                energy, stddevs = np.real(
                    expectation_from_correlation_matrix(hamiltonian_quad, corr_mat, cov)
                )
                data[occupied_orbitals].append(
                    (
                        energy + energy_shift,
                        stddevs,
                    )
                )
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"energy_{label}", data_zipped)

        error = np.zeros(len(chemical_potential_values))
        error_stddev = np.zeros(len(chemical_potential_values))
        for occupied_orbitals in occupied_orbitals_list:
            exact = np.array(energy_exact[occupied_orbitals])
            values, stddevs = data_zipped[occupied_orbitals]
            values = np.array(values)
            error += np.abs(values - exact)
            error_stddev += np.array(stddevs) ** 2
        error /= len(occupied_orbitals_list)
        error_stddev = np.sqrt(error_stddev) / len(occupied_orbitals_list)
        yield AnalysisResultData(f"energy_error_{label}", (error, error_stddev))

    def _compute_edge_correlation(
        self,
        label: str,
        corr: Dict[
            Tuple[int, float, Union[float, complex], Tuple[int, ...]],
            Tuple[np.ndarray, "_CovarianceDict"],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[Tuple[int, ...]],
    ) -> Iterable[AnalysisResultData]:
        edge_correlation = edge_correlation_op(n_modes)
        data = defaultdict(list)  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        for chemical_potential in chemical_potential_values:
            for occupied_orbitals in occupied_orbitals_list:
                corr_mat, cov = corr[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ]
                edge_correlation_val, stddev = np.real(
                    expectation_from_correlation_matrix(edge_correlation, corr_mat, cov)
                )
                data[occupied_orbitals].append(
                    (
                        edge_correlation_val,
                        stddev,
                    )
                )
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"edge_correlation_{label}", data_zipped)

    def _compute_number(
        self,
        label: str,
        corr: Dict[
            Tuple[int, float, Union[float, complex], Tuple[int, ...]],
            Tuple[np.ndarray, "_CovarianceDict"],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[Tuple[int, ...]],
    ) -> Iterable[AnalysisResultData]:
        number = number_op(n_modes)
        data = defaultdict(list)  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        for chemical_potential in chemical_potential_values:
            for occupied_orbitals in occupied_orbitals_list:
                corr_mat, cov = corr[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ]
                number_val, stddev = np.real(
                    expectation_from_correlation_matrix(number, corr_mat, cov)
                )
                data[occupied_orbitals].append(
                    (
                        number_val,
                        stddev,
                    )
                )
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"number_{label}", data_zipped)

    def _compute_parity(
        self,
        label: str,
        quasi_dists: Dict[
            Tuple[int, float, Union[float, complex], Tuple[int, ...]],
            Dict[Tuple[Tuple[int, ...], str], "QuasiDistribution"],
        ],
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[Tuple[int, ...]],
    ) -> Iterable[AnalysisResultData]:
        data = defaultdict(list)  # Dict[Tuple[int, ...], List[Tuple[float, float]]]
        for chemical_potential in chemical_potential_values:
            for occupied_orbitals in occupied_orbitals_list:
                quasis = quasi_dists[
                    tunneling, superconducting, chemical_potential, occupied_orbitals
                ]
                parity, stddev = compute_parity(quasis)
                data[occupied_orbitals].append(
                    (
                        parity,
                        stddev,
                    )
                )
        data_zipped = {k: tuple(np.array(a) for a in zip(*v)) for k, v in data.items()}
        yield AnalysisResultData(f"parity_{label}", data_zipped)