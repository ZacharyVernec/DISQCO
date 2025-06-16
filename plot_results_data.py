from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results():
    # Define file paths
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

    folderpath = Path(f'./')

    # Initialize data storage
    qubit_counts = {method:[] for method in method_names}
    e_bit_counts = {method:[] for method in method_names}
    times = {method:[] for method in method_names}


    for filename, method_name in zip(result_filenames, method_names):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if "num_qubits" in line:
                    qubit_count = int(line.split(",")[1].split("=")[1])
                    qubit_counts[method_name].append(qubit_count)
                elif "Min e-bit count" in line:
                    e_bit_count = int(line.split(":")[1].strip())
                    e_bit_counts[method_name].append(e_bit_count)
                elif "Time taken" in line:
                    time = float(line.split(":")[1].split()[0].strip())
                    times[method_name].append(time)

    for i in range(len(method_names)-1):
        assert qubit_counts[method_names[i]] == qubit_counts[method_names[i+1]], f"Qubit counts do not match across methods {method_names[i]} and {method_names[i+1]}"
    qubit_counts = qubit_counts[method_names[0]]
    x = np.arange(len(qubit_counts))

    # Plotting
    width = 1 / (len(method_names)+1)  # the width of the bars
    figsize=(width*len(qubit_counts)*20, 5)

    # Plot e-bit count vs num_qubits
    plt.figure(figsize=figsize)
    for i, (method, e_bit_count) in enumerate(e_bit_counts.items()):
        offset = width * i
        rects = plt.bar(x + offset, e_bit_count, width, label=method)
        plt.bar_label(rects, padding=3)
    plt.xlabel("Number of Qubits")
    plt.ylabel("Min E-bit Count")
    plt.title("E-bit Count vs Number of Qubits")
    plt.xticks(x, qubit_counts)
    plt.legend(loc='upper left', ncols=len(method_names))
    plt.legend()
    plt.savefig(folderpath / "bar_chart_e_bit_count_vs_num_qubits.png")
    
    # Plot time vs num_qubits
    plt.figure(figsize=figsize)
    def formatter(f: float):
        if f == 0:
            return "0"
        elif f < 1:
            return "~0"
        else:
            return str(int(f))
    for i, (method, time) in enumerate(times.items()):
        offset = width * i
        rects = plt.bar(x + offset, time, width, label=method)
        plt.bar_label(rects, fmt=formatter, padding=3)
    plt.xlabel("Number of Qubits")
    plt.ylabel("Time (seconds)")
    plt.title("Time vs Number of Qubits")
    plt.xticks(x, qubit_counts)
    plt.legend(loc='upper left', ncols=len(method_names))
    plt.legend()
    plt.savefig(folderpath / "bar_chart_time_vs_num_qubits.png")

if __name__ == "__main__":
    plot_results()