import pandas as pd
import random

# Load dataset asli
df = pd.read_csv("spam_10000.csv", encoding='latin-1')

# Ambil hanya kolom penting
df = df[['v1', 'v2']]

# Target jumlah data
targets = [10000, 25000, 50000, 75000, 100000, ]

for target in targets:
    # Random duplicate rows sampai mencapai target
    rows = [df.sample(n=1).iloc[0] for _ in range(target)]

    new_df = pd.DataFrame(rows)

    filename = f"spam_{target}.csv"

    new_df.to_csv(filename, index=False)

    print(f"{filename} berhasil dibuat dengan {target} dokumen")