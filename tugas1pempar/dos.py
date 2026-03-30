# mpi_reduce.py

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# hanya rank 0 yang punya data
if rank == 0:
    data = [10, 20, 30, 40, 50, 60, 70, 80]

    # membagi data menjadi beberapa bagian
    chunks = []
    for i in range(size):
        bagian = data[i::size]
        chunks.append(bagian)
else:
    chunks = None

# membagikan data ke semua proses
data_split = comm.scatter(chunks, root=0)

# setiap proses menghitung jumlah lokal
local_sum = sum(data_split)
print(f"Rank {rank} menghitung jumlah lokal: {local_sum}")
# menggabungkan hasil ke rank 0
total = comm.reduce(local_sum, op=MPI.SUM, root=0)

# tampilkan hasil
if rank == 0:
    print("Total jumlah (MPI Reduce):", total)