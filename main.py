import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_output_file(filename):
    """Парсит файл вывода и извлекает количество процессов и время выполнения"""
    # Извлекаем количество процессов из имени файла
    procs_match = re.search(r'output_ExpMPI_(\d+)_procs_run', filename)
    if not procs_match:
        return None

    procs = int(procs_match.group(1))

    with open(filename, 'r') as f:
        content = f.read()

        # Ищем время выполнения
        time_match = re.search(r'Execution time:\s+([\d.]+)', content)
        if not time_match:
            return None

        return {
            'processes': procs,
            'time': float(time_match.group(1))
        }


def collect_data(directory='output'):
    """Собирает данные из всех файлов output в указанной директории"""
    data = defaultdict(list)
    pattern = f"{directory}/output_ExpMPI_*_procs_run*_*.out"

    for filename in glob.glob(pattern):
        result = parse_output_file(filename)
        if result:
            data[result['processes']].append(result['time'])

    return data


def calculate_stats(data):
    """Вычисляет среднее время и стандартное отклонение для каждого количества процессов"""
    procs = sorted(data.keys())
    avg_times = [np.mean(data[p]) for p in procs]
    std_times = [np.std(data[p]) for p in procs]
    return procs, avg_times, std_times


def plot_results(procs, avg_times, std_times):
    """Строит графики времени выполнения, ускорения и эффективности"""
    # Конвертируем в массивы numpy для удобства вычислений
    procs = np.array(procs)
    avg_times = np.array(avg_times)
    std_times = np.array(std_times)

    # Находим минимальное количество процессов (будем считать его базовым)
    base_proc = min(procs)
    base_idx = np.where(procs == base_proc)[0][0]
    base_time = avg_times[base_idx]

    # Вычисляем ускорение и эффективность
    speedup = base_time / avg_times
    efficiency = speedup / procs

    # Создаем фигуру с тремя подграфиками
    plt.figure(figsize=(15, 5))

    # График времени выполнения
    plt.subplot(1, 3, 1)
    plt.errorbar(procs, avg_times, yerr=std_times, fmt='-o', capsize=5)
    plt.xlabel('Number of processes')
    plt.ylabel('Execution time (s)')
    plt.title(f'Execution Time vs Number of Processes\n(Baseline: {base_proc} process(es))')
    plt.grid(True)

    # График ускорения
    plt.subplot(1, 3, 2)
    plt.plot(procs, speedup, 'b-o', label='Actual speedup')
    plt.plot(procs, procs, 'r--', label='Linear speedup')
    plt.xlabel('Number of processes')
    plt.ylabel('Speedup (T1 / Tp)')
    plt.title(f'Speedup vs Number of Processes\n(Relative to {base_proc} process(es))')
    plt.legend()
    plt.grid(True)

    # График эффективности
    plt.subplot(1, 3, 3)
    plt.plot(procs, efficiency, 'g-o')
    plt.xlabel('Number of processes')
    plt.ylabel('Efficiency (S/p)')
    plt.title(f'Efficiency vs Number of Processes\n(Relative to {base_proc} process(es))')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('parallel_performance.png')
    plt.show()


def main():
    # Собираем данные из файлов вывода
    data = collect_data()

    if not data:
        print("No valid data found in output files.")
        return

    # Вычисляем статистику
    procs, avg_times, std_times = calculate_stats(data)

    # Выводим собранные данные для проверки
    print("Collected data:")
    for p in sorted(data.keys()):
        print(f"Processes: {p}, Runs: {len(data[p])}, Avg: {np.mean(data[p]):.4f} ± {np.std(data[p]):.4f}")

    # Строим графики
    plot_results(procs, avg_times, std_times)


if __name__ == '__main__':
    main()