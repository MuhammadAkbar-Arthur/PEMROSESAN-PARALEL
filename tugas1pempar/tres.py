# mpi_gather.py

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# data di rank 0
if rank == 0:
    data = [5, 10, 15, 20, 25, 30, 35, 40]
else:
    data = None

# broadcast data ke semua proses
data = comm.bcast(data, root=0)

# bagi data
chunk = data[rank::size]

# hitung lokal
local_sum = sum(chunk)
print(f"Rank {rank} menghitung jumlah lokal: {local_sum}")
# kumpulkan hasil ke rank 0
gathered = comm.gather(local_sum, root=0)

# sinkronisasi semua proses
comm.Barrier()

# proses akhir di rank 0
if rank == 0:
    total = sum(gathered)
    print("Total jumlah (MPI Gather):", total)