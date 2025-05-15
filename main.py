from qiskit.circuit.library import QFT
from qiskit import transpile

from disqco.parti.genetic.genetic_algorithm_original import Genetic_Partitioning

from disqco.parti.fgp.fgp_roee import set_initial_partition_fgp
from disqco.parti.fgp.fgp_roee import main_algorithm as fgp_algorithm

from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.parti.FM.FM_methods import set_initial_partitions
from disqco.parti.FM.multilevel_FM import MLFM_recursive

import time

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
    assignment = set_initial_partitions(quantum_network, num_qubits, depth, num_partitions)
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


def main():
    qpu_size = 8
    num_qubits_range = range(qpu_size*2, qpu_size*6+1, qpu_size)

    for num_qubits in num_qubits_range:
        circuit = QFT(num_qubits, do_swaps=False)

        num_partitions = num_qubits // qpu_size
        qpu_sizes = [qpu_size] * num_partitions # Equal sized QPUs
        depth = circuit.depth()

        print(f"{qpu_size=}, {num_qubits=}, {num_partitions=}")

        # Transpile the circuit to the basis gates
        basis_gates = ['u', 'cp']
        circuit = transpile(circuit, basis_gates=basis_gates)

        print(f'Number of qubits in circuit {circuit.num_qubits}')
        best_score, time = test_MLFM_R(circuit, qpu_sizes, num_partitions)
        print(f"Min e-bit count: {best_score}")
        print(f"Time taken for MLFM_R: {time} seconds")
        print()

if __name__ == "__main__":
    main()