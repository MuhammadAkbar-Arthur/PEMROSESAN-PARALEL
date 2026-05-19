import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# STYLE
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# LOAD CSV
# ==========================================
df = pd.read_csv("benchmark_results.csv")

# ==========================================
# EXTRACT DATASET SIZE
# ==========================================
def extract_size(name):
    return int(name.replace("spam_", "").replace(".csv", ""))

df["dataset_size"] = df["dataset"].apply(extract_size)

dataset_sizes = sorted(df["dataset_size"].unique())

# ==========================================
# OUTPUT FOLDER
# ==========================================
output_folder = "graphs"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ==========================================
# FUNCTION LABEL TITIK
# ==========================================
def add_labels(x, y):

    for i in range(len(x)):
        plt.text(
            x[i],
            y[i] + 0.01,
            f"{y[i]:.2f}",
            fontsize=9,
            ha='center'
        )

# ==========================================
# GRAFIK PER DATASET
# ==========================================
for size in dataset_sizes:

    plt.figure(figsize=(10,6))

    temp = df[df["dataset_size"] == size]

    # ======================================
    # SERIAL
    # ======================================
    serial = temp[temp["mode"] == "serial"]

    serial_time = serial["average_time"].values[0]

    # ======================================
    # PARALLEL
    # ======================================
    parallel = temp[temp["mode"] == "parallel"]
    parallel = parallel.sort_values(by="process")

    # ======================================
    # GABUNGKAN
    # ======================================
    x_values = [1]
    y_values = [serial_time]

    for _, row in parallel.iterrows():

        x_values.append(row["process"])
        y_values.append(row["average_time"])

    # ======================================
    # PLOT
    # ======================================
    plt.plot(
        x_values,
        y_values,
        marker='o',
        linewidth=3,
        markersize=10,
        label='Serial → Parallel MPI'
    )

    # ======================================
    # LABEL TITIK
    # ======================================
    add_labels(x_values, y_values)

    # ======================================
    # CARI TITIK OPTIMAL
    # ======================================
    min_time = min(y_values)
    min_index = y_values.index(min_time)

    optimal_x = x_values[min_index]

    plt.scatter(
        optimal_x,
        min_time,
        s=200,
        marker='*',
        label=f'Optimal ({optimal_x} Process)'
    )

    # ======================================
    # TITLE
    # ======================================
    plt.title(
        f"Execution Time vs Process ({size} Dokumen)",
        fontsize=16,
        fontweight='bold'
    )

    plt.xlabel("Jumlah Process", fontsize=12)
    plt.ylabel("Waktu Eksekusi (detik)", fontsize=12)

    plt.xticks([1,2,4,8])

    # ======================================
    # FOOTNOTE
    # ======================================
    plt.figtext(
        0.5,
        -0.03,
        "Semakin kecil execution time maka performa semakin baik",
        ha="center",
        fontsize=10
    )

    plt.legend()

    filename = f"{output_folder}/dataset_{size}.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"Saved: {filename}")

    plt.close()

# ==========================================
# SUMMARY GRAPH
# ==========================================
plt.figure(figsize=(12,7))

# ==========================================
# SERIAL
# ==========================================
serial_df = df[df["mode"] == "serial"]

plt.plot(
    serial_df["dataset_size"],
    serial_df["average_time"],
    marker='o',
    linewidth=3,
    markersize=8,
    label='Serial'
)

# ==========================================
# PARALLEL
# ==========================================
parallel_processes = sorted(
    df[df["mode"] == "parallel"]["process"].unique()
)

for p in parallel_processes:

    temp = df[
        (df["mode"] == "parallel") &
        (df["process"] == p)
    ]

    plt.plot(
        temp["dataset_size"],
        temp["average_time"],
        marker='o',
        linewidth=3,
        markersize=8,
        label=f'Parallel {p} Process'
    )

# ==========================================
# LABEL
# ==========================================
plt.title(
    "Perbandingan Serial vs Paralel MPI",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Jumlah Dokumen", fontsize=12)
plt.ylabel("Waktu Eksekusi (detik)", fontsize=12)

plt.xticks(dataset_sizes)

plt.figtext(
    0.5,
    -0.03,
    "Grafik menunjukkan pengaruh jumlah process terhadap waktu eksekusi TF-IDF",
    ha="center",
    fontsize=10
)

plt.legend()

summary_file = f"{output_folder}/summary_all.png"

plt.savefig(
    summary_file,
    dpi=300,
    bbox_inches='tight'
)

print(f"Saved: {summary_file}")

plt.show()