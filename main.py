import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd


def parse_output_file(filename):
    procs_match = re.search(r'output_ExpMPI_(\d+)_procs_run', filename)
    if not procs_match:
        return None

    procs = int(procs_match.group(1))

    with open(filename, 'r') as f:
        content = f.read()

    time_match = re.search(r'Execution time:\s+([\d.]+)', content)
    if not time_match:
        return None

    return {'processes': procs, 'time': float(time_match.group(1))}


def collect_data(directory='output'):
    data = defaultdict(list)
    pattern = f"{directory}/output_ExpMPI_*_procs_run*_*.out"

    for filename in glob.glob(pattern):
        result = parse_output_file(filename)
        if result:
            data[result['processes']].append(result['time'])

    return data


def calculate_stats(data):
    procs = sorted(data.keys())
    avg_times = [np.mean(data[p]) for p in procs]
    std_times = [np.std(data[p]) for p in procs]
    min_times = [np.min(data[p]) for p in procs]
    max_times = [np.max(data[p]) for p in procs]
    return procs, avg_times, std_times, min_times, max_times


def plot_results(procs, avg_times, min_times, max_times):
    procs = np.array(procs)
    avg_times = np.array(avg_times)
    min_times = np.array(min_times)
    max_times = np.array(max_times)

    base_proc = min(procs)
    base_idx = np.where(procs == base_proc)[0][0]
    base_time = avg_times[base_idx]

    speedup = base_time / avg_times
    efficiency = speedup / procs

    os.makedirs('results', exist_ok=True)

    plt.figure(figsize=(15, 5))


    plt.subplot(1, 3, 1)
    error_lower = avg_times - min_times
    error_upper = max_times - avg_times
    plt.errorbar(procs, avg_times, yerr=[error_lower, error_upper], fmt='-o', capsize=5)
    plt.xlabel('Number of processes')
    plt.ylabel('Execution time (s)')
    plt.title(f'Execution Time vs Number of Processes\n(Baseline: {base_proc} process(es))')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(procs, speedup, 'b-o', label='Actual speedup')
    plt.plot(procs, procs, 'r--', label='Linear speedup')
    plt.xlabel('Number of processes')
    plt.ylabel('Speedup (T1 / Tp)')
    plt.title('Speedup')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(procs, efficiency, 'g-o')
    plt.xlabel('Number of processes')
    plt.ylabel('Efficiency (S/p)')
    plt.title('Efficiency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/parallel_performance.png')
    plt.show()

    return speedup, efficiency


def save_to_excel(procs, avg_times, std_times, min_times, max_times, speedup, efficiency):
    df = pd.DataFrame({
        'Processes': procs,
        'Avg Time (s)': avg_times,
        'Std Time (s)': std_times,
        'Min Time (s)': min_times,
        'Max Time (s)': max_times,
        'Speedup': speedup,
        'Efficiency': efficiency
    })
    df.to_excel('results/metrics.xlsx', index=False)
    print("Excel файл сохранен в results/metrics.xlsx")


def main():
    data = collect_data()

    if not data:
        print("No valid data found in output files.")
        return

    procs, avg_times, std_times, min_times, max_times = calculate_stats(data)

    print("Collected data:")
    for i, p in enumerate(procs):
        print(f"Processes: {p}, Runs: {len(data[p])}, "
              f"Avg: {avg_times[i]:.4f} ± {std_times[i]:.4f}, "
              f"Min: {min_times[i]:.4f}, Max: {max_times[i]:.4f}")

    speedup, efficiency = plot_results(procs, avg_times, std_times, min_times, max_times)
    save_to_excel(procs, avg_times, std_times, min_times, max_times, speedup, efficiency)


if __name__ == '__main__':
    main()
