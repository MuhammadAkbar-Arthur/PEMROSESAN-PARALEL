from mpi4py import MPI
import math
import time
import os
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# =========================
# LOAD DATA (ROOT ONLY)
# =========================
def load_data(filename):

    documents = []

    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, filename)

    print("Current working dir:", os.getcwd())
    print("Membaca file:", filepath)

    with open(filepath, 'r', encoding='latin-1') as f:

        next(f)

        for line in f:

            parts = line.strip().split(',')

            if len(parts) >= 2:
                text = parts[1]
                documents.append(text)

    print("DEBUG: jumlah dokumen terbaca =", len(documents))

    return documents


# =========================
# PREPROCESSING
# =========================
def preprocess(text):

    return text.lower().split()


# =========================
# TERM FREQUENCY
# =========================
def compute_tf(doc_words):

    tf = {}

    total_words = len(doc_words)

    for word in doc_words:
        tf[word] = tf.get(word, 0) + 1

    for word in tf:
        tf[word] = tf[word] / total_words

    return tf


# =========================
# DOCUMENT FREQUENCY
# =========================
def compute_df(docs_words):

    df = {}

    for doc in docs_words:

        unique_words = set(doc)

        for word in unique_words:
            df[word] = df.get(word, 0) + 1

    return df


# =========================
# IDF
# =========================
def compute_idf(df, total_docs):

    idf = {}

    for word, freq in df.items():
        idf[word] = math.log(total_docs / freq)

    return idf


# =========================
# TF-IDF
# =========================
def compute_tfidf(tf_list, idf):

    tfidf_list = []

    for tf in tf_list:

        doc_tfidf = {}

        for word, val in tf.items():
            doc_tfidf[word] = val * idf.get(word, 0)

        tfidf_list.append(doc_tfidf)

    return tfidf_list


# =========================
# TOP WORDS GLOBAL
# =========================
def get_top_words(tfidf_list, top_n=10):

    total_score = {}

    for doc in tfidf_list:

        for word, score in doc.items():
            total_score[word] = total_score.get(word, 0) + score

    sorted_words = sorted(
        total_score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_words[:top_n]


# =========================
# MAIN
# =========================
def main(filename):

    start_time = time.time()

    # =========================
    # ROOT LOAD DATA
    # =========================
    if rank == 0:

        documents = load_data(filename)

        total_docs = len(documents)

        # DISTRIBUSI ROUND ROBIN
        chunks = [[] for _ in range(size)]

        for i, doc in enumerate(documents):
            chunks[i % size].append(doc)

    else:

        chunks = None
        total_docs = None

    # =========================
    # BROADCAST TOTAL DOCS
    # =========================
    total_docs = comm.bcast(total_docs, root=0)

    # =========================
    # SCATTER DATA
    # =========================
    local_docs = comm.scatter(chunks, root=0)

    print(f"Process {rank} menerima {len(local_docs)} dokumen")

    # =========================
    # PREPROCESSING
    # =========================
    local_docs_words = [
        preprocess(doc)
        for doc in local_docs
    ]

    # =========================
    # TF LOKAL
    # =========================
    local_tf_list = [
        compute_tf(doc)
        for doc in local_docs_words
    ]

    # =========================
    # DF LOKAL
    # =========================
    local_df = compute_df(local_docs_words)

    # =========================
    # GATHER DF
    # =========================
    all_df = comm.gather(local_df, root=0)

    if rank == 0:

        global_df = {}

        for df_part in all_df:

            for word, count in df_part.items():

                global_df[word] = (
                    global_df.get(word, 0) + count
                )

        idf = compute_idf(global_df, total_docs)

        print("\nJumlah kata unik:", len(global_df))

    else:

        idf = None

    # =========================
    # BROADCAST IDF
    # =========================
    idf = comm.bcast(idf, root=0)

    # =========================
    # TF-IDF LOKAL
    # =========================
    local_tfidf = compute_tfidf(local_tf_list, idf)

    # =========================
    # GATHER TF-IDF
    # =========================
    all_tfidf = comm.gather(local_tfidf, root=0)

    # =========================
    # OUTPUT ROOT
    # =========================
    if rank == 0:

        end_time = time.time()

        print("\n=== HASIL PARALEL TF-IDF ===")

        print("Jumlah dokumen:", total_docs)
        print("Jumlah process:", size)

        print(
            "Waktu eksekusi:",
            end_time - start_time,
            "detik"
        )

        print(
            "Detail tiap process:",
            [len(chunk) for chunk in chunks]
        )

        # flatten
        tfidf_list = []

        for part in all_tfidf:
            tfidf_list.extend(part)

        # TOP WORDS
        print("\n=== TOP 10 KATA PALING PENTING ===")

        top_words = get_top_words(tfidf_list)

        for word, score in top_words:
            print(f"{word} : {score}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":

    # ambil filename dari terminal
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        filename = "spam.csv"

    main(filename)