from qiskit.circuit.library import QFT
from qiskit import transpile

from disqco.parti.genetic.genetic_algorithm_original import Genetic_Partitioning

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
        best_score, time = test_GCP_S(circuit, qpu_sizes, num_partitions)
        print(f"Min e-bit count: {best_score}")
        print(f"Time taken for GCP-S: {time} seconds")
        print()

if __name__ == "__main__":
    main()