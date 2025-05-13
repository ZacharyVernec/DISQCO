import os
import json
import time
import itertools
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QFT, QuantumVolume
from disqco.circuits.QAOA import QAOA_random
from disqco.circuits.cp_fraction import cp_fraction
from disqco.parti.fgp.fgp_roee import *
from qiskit.circuit.library import QuantumVolume, QFT
from qiskit import transpile
import time

detailed_filename_list = [
    "benchmark_results_FGP_CP_large_5it_2-4part_new.json",
    "benchmark_results_FGP_QV_5it_2-4part_new.json",
    "benchmark_results_FGP_QV_5it_2-4part_new.json",
    "benchmark_results_FGP_QFT_5it_2-4part_new.json",
    "benchmark_results_FGP_CP_2-12_new.json"

]

means_filename_list = [
    "benchmark_means_FGP_CP_large_5it_2-4part_new.json",
    "benchmark_means_FGP_QV_5it_2-4part_new.json",
    "benchmark_means_FGP_QV_5it_2-4part_new.json",
    "benchmark_means_FGP_QFT_5it_2-4part_new.json",
    "benchmark_means_FGP_CP_2-12_new.json"
]

fractions_list = [
    [None],
    [None],
    [None],
    [None],
    [0.3, 0.5, 0.7, 0.9]
]

num_partitions_and_i_sizes_list = [
    itertools.product([2,4], enumerate(range(128, 257, 32))),
    itertools.product([2,4], enumerate(range(16, 97, 8))),
    itertools.product([4], enumerate(range(16, 97, 8))),
    itertools.product([2,4], enumerate(range(16, 97, 8))),
    [(2+i, (i,size)) for i,size in enumerate(range(16, 97, 8))]
]

circuit_generator_list = [
    lambda num_qubits: cp_fraction(num_qubits, num_qubits, fraction=0.5),
    lambda num_qubits: QuantumVolume(num_qubits, depth=num_qubits),
    lambda num_qubits: QuantumVolume(num_qubits, depth=num_qubits),
    lambda num_qubits: QFT(num_qubits, do_swaps=False),
    lambda num_qubits, fraction: cp_fraction(num_qubits, num_qubits, fraction=fraction)
]

experiments = zip(detailed_filename_list, means_filename_list, num_partitions_and_i_sizes_list, fractions_list, circuit_generator_list)


for experiment in experiments:
    detailed_filename, means_filename, num_partitions_and_i_sizes, fractions, circuit_generator = experiment

    ###############################################################################
    # Set up JSON file for storing *all* iteration results (detailed data)
    ###############################################################################

    if os.path.exists(detailed_filename):
        with open(detailed_filename, "r") as f:
            detailed_results = json.load(f)
    else:
        detailed_results = []

    ###############################################################################
    # Set up JSON file for *aggregated* results (mean cost/time)
    ###############################################################################

    if os.path.exists(means_filename):
        with open(means_filename, "r") as f:
            mean_results = json.load(f)
    else:
        mean_results = []

    for fraction in fractions:
        for (num_partitions, (i, num_qubits)) in num_partitions_and_i_sizes:
            # For each increase of 8 qubits, increase the number of partitions by 1


            # Create an All-to-All network
            qpu_info = [int(num_qubits / num_partitions) + 1 for _ in range(num_partitions)]
            
            # Sweep the fraction parameter from 0.1 to 0.9
                # Collect data for computing means across 10 iterations
            iteration_data = []
            for iteration in range(5):
                
                # -------------------------
                # 1. Define/redefine circuit
                # -------------------------

                # base_graph = build_circuit(num_qubits,fraction=fraction,group_gates=True)
                # Define the number of qubits and the circuit
                if fraction is None:
                    circuit = circuit_generator(num_qubits)
                else:
                    circuit = circuit_generator(num_qubits, fraction)

                # Transpile the circuit into some basis gates. The gate set here was used to match those used in the GCP paper, but it shouldn't matter which gates are used.
                basis_gates = ['cp','u']
                transpiled_circuit = transpile(circuit, basis_gates=basis_gates)
                # Define the number of partitions
                # Define the QPU sizes in terms of data qubit capacity, here they are defined to be equal and match the number of qubits in the circuit.
                # Note that if the number of qubits in the circuit is odd, fully local partitions can be impossible. 
                # E.g. if you have a 9 qubit circuit and 3x3 qubit QPUs, then you can't accomodate 4 pairs of qubits interacting at the same time, so you need to increase the size of the QPUs.


                initial_partition = set_initial_partition_fgp(qpu_info=qpu_info,num_partitions=num_partitions)
                start = time.time()
                partition, cost, mapping = main_algorithm(circuit=transpiled_circuit, qpu_info=qpu_info,initial_partition=initial_partition,remove_singles=False,choose_initial=True)
                stop = time.time()

                total_time_fgp = stop - start
                min_cost_fgp = cost

                
                # -------------------------
                # 6. Store iteration-level results
                # -------------------------
                if fraction is None:
                    result_entry = {
                        "num_qubits": num_qubits,
                        "num_partitions": num_partitions,
                        "iteration": iteration,
                        "fgp_cost":  min_cost_fgp,
                        "time_fgp": total_time_fgp,
                    }
                else:
                    result_entry = {
                        "num_qubits": num_qubits,
                        "num_partitions": num_partitions,
                        "fraction" : fraction,
                        "iteration": iteration,
                        "fgp_cost":  min_cost_fgp,
                        "time_fgp": total_time_fgp,
                    }


                
                detailed_results.append(result_entry)
                iteration_data.append(result_entry)
                
                # Update detailed JSON right away
                with open(detailed_filename, "w") as f:
                    json.dump(detailed_results, f, indent=2)
            
            # ---------------------------------------------------------------------
            # After 10 iterations, compute the means and log them
            # ---------------------------------------------------------------------

            r_cost_list = [x["fgp_cost"] for x in iteration_data]
            

            r_time_list = [x["time_fgp"] for x in iteration_data]
            

            mean_r_cost = float(np.mean(r_cost_list))
            

            mean_r_time = float(np.mean(r_time_list))
            
            # Print to console for quick logging
            print("=============================================")
            print(f"Finished 10 iterations for:")
            if fraction is None:
                print(f"  # Qubits: {num_qubits}, # Partitions: {num_partitions}")
            else:
                print(f"  # Qubits: {num_qubits}, # Partitions: {num_partitions} , Fraction: {fraction}")
            print("Mean Costs:")

            print(f"  Recursive:{mean_r_cost:.3f}")
            print("Mean Times (s):")

            print(f"  Recursive:{mean_r_time:.3f}")
            print("=============================================")
            
            # Store the aggregated means in a separate JSON
            if fraction is None:
                mean_entry = {
                    "num_qubits": num_qubits,
                    "num_partitions": num_partitions,
                    "mean_r_cost": mean_r_cost,
                    "mean_r_time": mean_r_time,
                }
            else:
                mean_entry = {
                    "num_qubits": num_qubits,
                    "num_partitions": num_partitions,
                    "fraction" : fraction,
                    "mean_r_cost": mean_r_cost,
                    "mean_r_time": mean_r_time,
                }
            
            mean_results.append(mean_entry)
            
            # Update the means JSON file
            with open(means_filename, "w") as f:
                json.dump(mean_results, f, indent=2)

        print("Benchmarking completed. Detailed results saved to", detailed_filename)
        print("Aggregated means saved to", means_filename)
        print()
        print()