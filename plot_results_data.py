import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    # Define file paths
    result_filenames = [
        "gcp-s-results-data.txt",
        "gcp-e-results-data.txt",
        "fgp-roee-results-data.txt",
        "mlfm-r-results-data.txt",
    ]
    method_names = ['-'.join(filename.split("-")[:2]).upper() for filename in result_filenames]

    # Initialize data storage
    qubit_counts = {method:[] for method in method_names}
    e_bit_counts = {method:[] for method in method_names}
    times = {method:[] for method in method_names}

    for filename, method_name in zip(result_filenames, method_names):
        with open(filename, "r", encoding="utf-16") as f:
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

    # Plot e-bit count vs num_qubits
    width = 1 / len(qubit_counts)  # the width of the bars
    multiplier = 0
    plt.figure(figsize=(10, 5))
    for method, e_bit_count in e_bit_counts.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, e_bit_count, width, label=method)
        plt.bar_label(rects, padding=3)
        multiplier +=1
    plt.xlabel("Number of Qubits")
    plt.ylabel("Min E-bit Count")
    plt.title("E-bit Count vs Number of Qubits")
    plt.xticks(x + width, qubit_counts)
    plt.legend(loc='upper left', ncols=len(method_names))
    plt.legend()
    plt.savefig("bar_chart_e_bit_count_vs_num_qubits.png")
    
    # Plot time vs num_qubits
    width = 1 / len(qubit_counts)  # the width of the bars
    multiplier = 0
    plt.figure(figsize=(10, 5))
    for method, time in times.items():
        offset = width * multiplier
        # time_strs = [f"{time_:.8f}" for time_ in time] #TODO rename
        rects = plt.bar(x + offset, time, width, label=method)
        plt.bar_label(rects, padding=3)
        multiplier +=1
    plt.xlabel("Number of Qubits")
    plt.ylabel("Time (seconds)")
    plt.title("Time vs Number of Qubits")
    plt.xticks(x + width, qubit_counts)
    plt.legend(loc='upper left', ncols=len(method_names))
    plt.legend()
    plt.savefig("bar_chart_time_vs_num_qubits.png")

if __name__ == "__main__":
    plot_results()