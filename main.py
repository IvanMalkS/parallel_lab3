import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd


def parse_output_file(filename):
    tech_type = None
    procs = None
    
    mpi_match = re.search(r'output_(MPI|ExpMPI)_(\d+)_(procs|threads)_run', filename)
    omp_match = re.search(r'output_OMP_(\d+)_threads_run', filename)
    
    if mpi_match:
        tech_type = 'MPI'
        procs = int(mpi_match.group(2))
    elif omp_match:
        tech_type = 'OpenMP'
        procs = int(omp_match.group(1))
    else:
        return None

    with open(filename, 'r') as f:
        content = f.read()

    time_match = re.search(r'Execution time:\s+([\d.]+)', content)
    if not time_match:
        return None

    return {
        'technology': tech_type,
        'processes': procs,
        'time': float(time_match.group(1))
    }


def collect_data(directory='logs'):
    data = defaultdict(lambda: defaultdict(list))
    pattern = f"{directory}/output_*.out"

    for filename in glob.glob(pattern):
        result = parse_output_file(filename)
        if result:
            data[result['technology']][result['processes']].append(result['time'])

    return data


def calculate_stats(data):
    stats = {}
    for tech in data:
        procs = sorted(data[tech].keys())
        avg_times = [np.mean(data[tech][p]) for p in procs]
        std_times = [np.std(data[tech][p]) for p in procs]
        min_times = [np.min(data[tech][p]) for p in procs]
        max_times = [np.max(data[tech][p]) for p in procs]
        
        stats[tech] = {
            'procs': procs,
            'avg_times': avg_times,
            'std_times': std_times,
            'min_times': min_times,
            'max_times': max_times
        }
    
    return stats


def plot_results(stats):
    os.makedirs('results', exist_ok=True)
    
    for tech in stats:
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Execution time
        plt.subplot(1, 3, 1)
        procs = np.array(stats[tech]['procs'])
        avg_times = np.array(stats[tech]['avg_times'])
        min_times = np.array(stats[tech]['min_times'])
        max_times = np.array(stats[tech]['max_times'])
        
        error_lower = avg_times - min_times
        error_upper = max_times - avg_times
        
        plt.errorbar(procs, avg_times, yerr=[error_lower, error_upper], 
                    fmt='-o', capsize=5, color='blue')
        plt.xlabel('Number of processes/threads')
        plt.ylabel('Execution time (s)')
        plt.title(f'{tech} Execution Time')
        plt.grid(True)

        # Subplot 2: Speedup
        plt.subplot(1, 3, 2)
        base_time = avg_times[0]
        speedup = base_time / avg_times
        
        plt.plot(procs, speedup, '-o', color='green', label='Actual speedup')
        plt.plot(procs, procs, 'r--', label='Linear speedup')
        plt.xlabel('Number of processes/threads')
        plt.ylabel('Speedup (T1 / Tp)')
        plt.title(f'{tech} Speedup')
        plt.legend()
        plt.grid(True)
        
        # Subplot 3: Efficiency
        plt.subplot(1, 3, 3)
        efficiency = speedup / procs
        
        plt.plot(procs, efficiency, '-o', color='purple')
        plt.xlabel('Number of processes/threads')
        plt.ylabel('Efficiency (S/p)')
        plt.title(f'{tech} Efficiency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/{tech}_performance.png')
        plt.show()


def save_to_excel(stats):
    all_data = []
    
    for tech in stats:
        for i, p in enumerate(stats[tech]['procs']):
            base_time = stats[tech]['avg_times'][0]
            speedup = base_time / stats[tech]['avg_times'][i]
            efficiency = speedup / p
            
            all_data.append({
                'Technology': tech,
                'Processes/Threads': p,
                'Avg Time (s)': stats[tech]['avg_times'][i],
                'Std Time (s)': stats[tech]['std_times'][i],
                'Min Time (s)': stats[tech]['min_times'][i],
                'Max Time (s)': stats[tech]['max_times'][i],
                'Speedup': speedup,
                'Efficiency': efficiency
            })
    
    df = pd.DataFrame(all_data)
    df.to_excel('results/metrics_comparison.xlsx', index=False)
    print("Excel файл сохранен в results/metrics_comparison.xlsx")


def main():
    data = collect_data()

    if not data:
        print("No valid data found in output files.")
        return

    stats = calculate_stats(data)

    print("Collected data:")
    for tech in stats:
        print(f"\nTechnology: {tech}")
        for i, p in enumerate(stats[tech]['procs']):
            print(f"Processes/Threads: {p}, Runs: {len(data[tech][p])}, "
                  f"Avg: {stats[tech]['avg_times'][i]:.4f} ± {stats[tech]['std_times'][i]:.4f}, "
                  f"Min: {stats[tech]['min_times'][i]:.4f}, Max: {stats[tech]['max_times'][i]:.4f}")

    plot_results(stats)
    save_to_excel(stats)


if __name__ == '__main__':
    main()