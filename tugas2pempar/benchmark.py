import subprocess
import time
import csv
import statistics

# ==========================================
# KONFIGURASI
# ==========================================
datasets = [
    "spam_10000.csv",
    "spam_25000.csv",
    "spam_50000.csv",
    "spam_75000.csv",
    "spam_100000.csv"
]

process_counts = [2, 4, 6, 8, 12, 16, 24]

trials = 5

output_csv = "benchmark_results.csv"

# ==========================================
# AMBIL EXECUTION TIME
# ==========================================
def extract_time(output):
    for line in output.splitlines():
        if "Waktu eksekusi:" in line:
            try:
                value = float(line.split(":")[1].split()[0])
                return value
            except:
                return None
    return None

# ==========================================
# JALANKAN SERIAL
# ==========================================
def run_serial(dataset):
    cmd = ["python", "tfidf_serial.py", dataset, "--no-input"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    return extract_time(result.stdout)

# ==========================================
# JALANKAN PARALEL
# ==========================================
def run_parallel(dataset, process_count):
    cmd = [
        "mpiexec",
        "-n",
        str(process_count),
        "python",
        "tfidf_parallel.py",
        dataset
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    return extract_time(result.stdout)

# ==========================================
# HAPUS OUTLIER
# ==========================================
def remove_outlier(data):
    if len(data) <= 2:
        return data

    mean = statistics.mean(data)
    stdev = statistics.stdev(data)

    filtered = []

    for x in data:
        if abs(x - mean) <= 2 * stdev:
            filtered.append(x)

    return filtered if filtered else data

# ==========================================
# MAIN BENCHMARK
# ==========================================
results = []

for dataset in datasets:

    print("\n" + "="*60)
    print("DATASET:", dataset)
    print("="*60)

    # ======================================
    # SERIAL
    # ======================================
    serial_times = []

    for i in range(trials):
        print(f"Serial Trial {i+1}...")
        t = run_serial(dataset)

        if t is not None:
            serial_times.append(t)

    serial_filtered = remove_outlier(serial_times)

    serial_avg = statistics.mean(serial_filtered)

    results.append([
        dataset,
        "serial",
        1,
        serial_times,
        serial_avg
    ])

    print("Serial Average:", serial_avg)

    # ======================================
    # PARALEL
    # ======================================
    for p in process_counts:

        parallel_times = []

        for i in range(trials):
            print(f"Parallel {p} Process Trial {i+1}...")

            t = run_parallel(dataset, p)

            if t is not None:
                parallel_times.append(t)

        parallel_filtered = remove_outlier(parallel_times)

        parallel_avg = statistics.mean(parallel_filtered)

        results.append([
            dataset,
            "parallel",
            p,
            parallel_times,
            parallel_avg
        ])

        print(f"Parallel {p} Average:", parallel_avg)

# ==========================================
# SIMPAN CSV
# ==========================================
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "dataset",
        "mode",
        "process",
        "all_times",
        "average_time"
    ])

    writer.writerows(results)

print("\nBenchmark selesai!")
print("Data tersimpan di:", output_csv)