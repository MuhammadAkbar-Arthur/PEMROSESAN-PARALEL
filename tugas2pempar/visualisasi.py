import csv
import matplotlib.pyplot as plt

processes = []
times = []
speedup = []
efficiency = []

serial_time = None

# =========================
# LOAD DATA
# =========================
with open("benchmark.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        mode = row["mode"]
        p = int(row["process"])
        t = float(row["time"])

        if mode == "serial":
            serial_time = t
        else:
            processes.append(p)
            times.append(t)

# =========================
# HITUNG SPEEDUP & EFFICIENCY
# =========================
for i, p in enumerate(processes):
    s = serial_time / times[i]
    e = s / p

    speedup.append(s)
    efficiency.append(e)

# =========================
# PLOT 1: EXECUTION TIME
# =========================
plt.figure()
plt.plot(processes, times, marker='o')
plt.xlabel("Jumlah Process")
plt.ylabel("Waktu Eksekusi (detik)")
plt.title("Execution Time vs Process")
plt.grid()
plt.show()

# =========================
# PLOT 2: SPEEDUP
# =========================
plt.figure()
plt.plot(processes, speedup, marker='o')
plt.xlabel("Jumlah Process")
plt.ylabel("Speedup")
plt.title("Speedup vs Process")
plt.grid()
plt.show()

# =========================
# PLOT 3: EFFICIENCY
# =========================
plt.figure()
plt.plot(processes, efficiency, marker='o')
plt.xlabel("Jumlah Process")
plt.ylabel("Efficiency")
plt.title("Efficiency vs Process")
plt.grid()
plt.show()