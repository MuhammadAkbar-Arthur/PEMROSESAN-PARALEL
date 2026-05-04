import subprocess
import time
import csv

# =========================
# KONFIGURASI
# =========================
process_list = [1, 2, 4, 8]
results = []

# =========================
# RUN SERIAL
# =========================
print("Running SERIAL...")
start = time.time()
subprocess.run(["python", "tfidf_serial.py", "--no-input"], stdout=subprocess.DEVNULL)
end = time.time()

serial_time = end - start
print("Serial time:", serial_time)

# simpan baseline
results.append(("serial", 1, serial_time))

# =========================
# RUN PARALEL
# =========================
for p in process_list:
    print(f"Running PARALLEL with {p} processes...")

    start = time.time()
    subprocess.run(["mpiexec", "-n", str(p), "python", "tfidf_parallel.py"], stdout=subprocess.DEVNULL)
    end = time.time()

    parallel_time = end - start

    print(f"Process {p}: {parallel_time}")

    results.append(("parallel", p, parallel_time))

# =========================
# SIMPAN CSV
# =========================
with open("benchmark.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mode", "process", "time"])

    for row in results:
        writer.writerow(row)

print("\nData tersimpan di benchmark.csv")