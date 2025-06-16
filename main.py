import itertools
import logging
import time
from pathlib import Path

import numpy as np
from pytket.extensions.qiskit import qiskit_to_tk
from pytket_dqc.allocators import Annealing, HypergraphPartitioning
from pytket_dqc.distributors import Distributor
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.refiners import (DetachedGates, EagerHTypeMerge,
                                 IntertwinedDTypeMerge, NeighbouringDTypeMerge,
                                 RepeatRefiner, SequenceRefiner, VertexCover)
from pytket_dqc.utils import DQCPass, ebit_cost
from qiskit import transpile
from qiskit.circuit.library import QFT

from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.parti.fgp.fgp_roee import main_algorithm as fgp_algorithm
from disqco.parti.fgp.fgp_roee import set_initial_partition_fgp
from disqco.parti.FM.FM_methods import set_initial_partitions
from disqco.parti.FM.multilevel_FM import MLFM_recursive
from disqco.parti.genetic.genetic_algorithm_original import \
    Genetic_Partitioning


def _test_GCP(circuit, qpu_sizes, num_partitions, gate_packing):
    start = time.time()
    genetic_partitioner = Genetic_Partitioning(circuit, qpu_sizes, gate_packing=gate_packing)
    population, max_over_time = genetic_partitioner.run(
        pop_size=100,
        num_generations=100,
        mutation_rate=0.9, 
        multi_process=True, 
        search_method=True, 
        search_number=100, 
        log=False
    )
    stop = time.time()
    duration = stop - start
    best_score = max_over_time[-1]
    return best_score, duration

def test_GCP_S(circuit, qpu_sizes, num_partitions):
    return _test_GCP(circuit, qpu_sizes, num_partitions, gate_packing=False)

def test_GCP_E(circuit, qpu_sizes, num_partitions):
    return _test_GCP(circuit, qpu_sizes, num_partitions, gate_packing=True)

def test_FGP(circuit, qpu_sizes, num_partitions):
    start = time.time()
    initial_partition = set_initial_partition_fgp(qpu_info=qpu_sizes, num_partitions=num_partitions)
    partition, cost, mapping = fgp_algorithm(circuit=circuit, 
                                              qpu_info=qpu_sizes,
                                              initial_partition=initial_partition,
                                              remove_singles=False,
                                              choose_initial=True)
    stop = time.time()
    duration = stop - start
    return cost, duration

def test_MLFM_R(circuit, qpu_sizes, num_partitions):
    quantum_network = QuantumNetwork(qpu_sizes)
    num_qubits = circuit.num_qubits
    depth = circuit.depth()
    start = time.time()
    assignment = set_initial_partitions(quantum_network, num_qubits, depth)
    graph = QuantumCircuitHyperGraph(circuit, group_gates=True, anti_diag=True, map_circuit=True)
    assignment_list_MLFMR, cost_list_MLFMR, _ = MLFM_recursive(graph,
                                            assignment,  
                                            qpu_sizes,
                                            limit=num_qubits,
                                            log = False)
    stop = time.time()
    duration = stop - start
    cost = min(cost_list_MLFMR)
    # assignment = assignment_list_MLFMR[np.argmin(cost_list_MLFMR)]
    return cost, duration

def test_ZV_THY(circuit, qpu_sizes, num_partitions):
    qpu_sizes = sorted(qpu_sizes)
    cost = sum(k*qpu_size for k,qpu_size in enumerate(qpu_sizes))
    duration = 0
    return cost, duration

def test_PYTKET_PE(circuit, qpu_sizes, num_partitions):
    qpu_indices = list(range(num_partitions))
    qpu_cum_sizes = [0] + list(np.cumsum(qpu_sizes))
    qpu_qubits_ranges = [range(start, stop) for start,stop in zip(qpu_cum_sizes, qpu_cum_sizes[1:])]

    network = NISQNetwork(
        server_coupling=list(itertools.product(qpu_indices, qpu_indices)), # fully connected
        server_qubits={qpu_idx:list(qubits_range) for qpu_idx,qubits_range in enumerate(qpu_qubits_ranges)}
    )

    tk_circuit = qiskit_to_tk(circuit.decompose()) #assumes circuit is a box
    DQCPass().apply(tk_circuit) # decompose into CP, H, and Rz

    start = time.time()
    """Equivalent too the workflow PartitionEmbed 
    cited in [Andres-Matrinez et al. 2024] and [Burt et al. 2025]

    Implemented by merging the pytket-dqc distributors
    - PartitioningHeterogeneous, which includes the HypergraphPartitioning
    - PartitioningHeterogeneousEmbedding, that includes embedding
    but removing the boundary reallocation of the latter (i.e. removing the 'Heterogeneous' part).
    """
    distribution = HypergraphPartitioning().allocate(tk_circuit, network) #seed goes as kwarg
    refiner = RepeatRefiner(EagerHTypeMerge())
    refiner.refine(distribution)
    
    stop = time.time()
    duration = stop - start

    cost = distribution.cost()
    # distributed_circuit = distribution.to_pytket_circuit()
    # circuit_cost = ebit_cost(distributed_circuit)
    # nl_count = distribution.non_local_gate_count()
    # detached_count = distribution.detached_gate_count()

    # print(f"{cost=}")
    # print(f"{circuit_cost=}")
    # print(f"{nl_count=}")
    # print(f"{detached_count=}")
    
    return cost, duration

def test_PYTKET_AESD(circuit, qpu_sizes, num_partitions):
    qpu_indices = list(range(num_partitions))
    qpu_cum_sizes = [0] + list(np.cumsum(qpu_sizes))
    qpu_qubits_ranges = [range(start, stop) for start,stop in zip(qpu_cum_sizes, qpu_cum_sizes[1:])]

    network = NISQNetwork(
        server_coupling=list(itertools.product(qpu_indices, qpu_indices)), # fully connected
        server_qubits={qpu_idx:list(qubits_range) for qpu_idx,qubits_range in enumerate(qpu_qubits_ranges)}
    )

    tk_circuit = qiskit_to_tk(circuit.decompose()) #assumes circuit is a box
    DQCPass().apply(tk_circuit) # decompose into CP, H, and Rz

    start = time.time()
    """Equivalent too the workflow EmbedSteinerDetach 
    cited in [Andres-Matrinez et al. 2024] 
    but with Annealer instead of KaHyPar, as in [Burt et al. 2024]

    Implemented by merging the pytket-dqc distributors
    - PartitioningAnnealing, which includes the annealing
    - CoverEmbedding, which includes the embedding
    - CoverEmbeddingSteiner, which includes the Steiner tree merging
    - CoverEmbeddingSteinerDetached, which includes the reallocation using detached gates
    """
    # PartitioningAnnealing
    distribution = Annealing().allocate(tk_circuit, network)  #seed goes as kwarg
    # CoverEmbedding
    VertexCover().refine(distribution)
    # Steiner
    refiner_list = [
        NeighbouringDTypeMerge(),
        IntertwinedDTypeMerge(),
    ]
    refiner = RepeatRefiner(SequenceRefiner(refiner_list))
    refiner.refine(distribution)
    # Detached
    DetachedGates().refine(distribution)

    stop = time.time()
    duration = stop - start

    cost = distribution.cost()

    return cost, duration


def main():
    result_filenames = [
        "results-data-gcp-s.txt",
        "results-data-gcp-e.txt",
        "results-data-fgp-roee.txt",
        "results-data-mlfm-r.txt", 
        "results-data-zv-thy.txt",
        "results-data-pytket-pe.txt",
        "results-data-pytket-aesd.txt",
    ]
    method_names = [
        "GCP-S",
        "GCP-E",
        "FGP-rOEE",
        "MLFM_R",
        "ZV_THY",
        "PYTKET_PE",
        "PYTKET_AESD",
    ]
    test_methods = [
        test_GCP_S, 
        test_GCP_E, 
        test_FGP, 
        test_MLFM_R, 
        test_ZV_THY, 
        test_PYTKET_PE, 
        test_PYTKET_AESD,
    ]

    for test_method, method_name, result_filename in zip(test_methods, method_names, result_filenames):
        print(f"Testing {method_name}")

        num_qubits = 256
        print(f"Testing {num_qubits=}")

        for num_partitions in [2, 4, 8, 32, 128]:
            print(f"Testing {num_partitions=}")

            output_string = ""

            circuit = QFT(num_qubits, do_swaps=False)

            qpu_size = num_qubits // num_partitions #mode
            remaining_qubits = num_qubits - qpu_size * num_partitions
            qpu_sizes = [qpu_size] * num_partitions # Equal sized QPUs
            qpu_sizes[-1] += remaining_qubits # except last QPU
            depth = circuit.depth()

            output_string += f"{qpu_size=}, {num_qubits=}, {num_partitions=}\n"

            # Transpile the circuit to the basis gates
            basis_gates = ['u', 'cp']
            circuit = transpile(circuit, basis_gates=basis_gates) # TODO refactor

            output_string += f'Number of partitions {num_partitions}\n'
            best_score, time = test_method(circuit, qpu_sizes, num_partitions)
            output_string += f"Min e-bit count: {best_score}\n"
            output_string += f"Time taken for {method_name}: {time} seconds\n"

            filepath = Path(f'./results_massive_qft/{result_filename}')
            mode = 'w' if num_partitions == 2 else 'a'
            with filepath.open(mode) as f:
                print(output_string, file=f)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    main()