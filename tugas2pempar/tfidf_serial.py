import math
import time
import os
import sys

# =========================
# LOAD DATA
# =========================
def load_data(filename):
    documents = []

    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, filename)

    print("Current working dir:", os.getcwd())
    print("Membaca file:", filepath)

    with open(filepath, 'r', encoding='latin-1') as f:
        next(f)  # skip header
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
    text = text.lower()
    words = text.split()
    return words


# =========================
# TERM FREQUENCY
# =========================
def compute_tf(doc_words):
    tf = {}
    total_words = len(doc_words)

    for word in doc_words:
        tf[word] = tf.get(word, 0) + 1

    # normalisasi
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

    sorted_words = sorted(total_score.items(), key=lambda x: x[1], reverse=True)

    return sorted_words[:top_n]


# =========================
# ANALISIS DETAIL KATA
# =========================
def analyze_word(word, docs_words, tf_list, idf, tfidf_list):
    print("\n" + "="*50)
    print("ANALISIS DETAIL KATA:", word)
    print("="*50)

    total_docs = len(docs_words)

    # cari dokumen yang mengandung kata
    doc_indices = []
    for i, doc in enumerate(docs_words):
        if word in doc:
            doc_indices.append(i)

    df = len(doc_indices)

    if df == 0:
        print("❌ Kata tidak ditemukan dalam dataset.")
        return

    # IDF
    idf_value = idf[word]

    print("\n[STEP 1] DOCUMENT FREQUENCY (DF)")
    print(f"Kata muncul di {df} dari {total_docs} dokumen")

    print("\n[STEP 2] INVERSE DOCUMENT FREQUENCY (IDF)")
    print("Rumus: IDF = log(N / DF)")
    print(f"IDF = log({total_docs} / {df}) = {idf_value}")

    print("\n[STEP 3] TERM FREQUENCY (TF) + TF-IDF PER DOKUMEN")
    print("-"*80)
    print(f"{'Doc':<6} {'Jumlah Kata':<12} {'Count':<8} {'TF':<10} {'TF-IDF':<10}")
    print("-"*80)

    results = []

    for i in doc_indices:
        doc = docs_words[i]
        tf = tf_list[i]

        total_words = len(doc)
        count = doc.count(word)
        tf_value = tf[word]
        tfidf_value = tfidf_list[i][word]

        results.append((i, tfidf_value))

        print(f"{i:<6} {total_words:<12} {count:<8} {tf_value:<10.4f} {tfidf_value:<10.4f}")

    print("-"*80)

    # TOP dokumen
    results = sorted(results, key=lambda x: x[1], reverse=True)[:5]

    print("\n[STEP 4] TOP 5 DOKUMEN PALING RELEVAN")
    for doc_id, score in results:
        print(f"Doc {doc_id} → TF-IDF = {score}")

    print("\n[INTERPRETASI]")
    print("- TF tinggi → kata sering muncul di dokumen itu")
    print("- IDF tinggi → kata jarang di dataset (lebih penting)")
    print("- TF-IDF tinggi → kata sangat relevan di dokumen tersebut")


# =========================
# MAIN
# =========================
def main(interactive=True):
    start_time = time.time()

    documents = load_data("spam.csv")

    # preprocessing
    docs_words = [preprocess(doc) for doc in documents]

    # TF
    tf_list = [compute_tf(doc) for doc in docs_words]

    # DF
    df = compute_df(docs_words)

    # IDF
    idf = compute_idf(df, len(documents))

    # TF-IDF
    tfidf_list = compute_tfidf(tf_list, idf)

    end_time = time.time()

    # =========================
    # OUTPUT GLOBAL
    # =========================
    print("\n=== HASIL MANUAL TF-IDF ===")
    print("Jumlah dokumen:", len(documents))
    print("Jumlah kata unik:", len(df))
    print("Waktu eksekusi:", end_time - start_time, "detik")

    # TOP WORDS
    print("\n=== TOP 10 KATA PALING PENTING ===")
    top_words = get_top_words(tfidf_list)

    for word, score in top_words:
        print(f"{word} : {score}")

    # =========================
    # INTERAKTIF
    # =========================
    if interactive:
        while True:
            word = input("\nMasukkan kata (ketik 'exit' untuk keluar): ").lower()

            if word == "exit":
                break

            analyze_word(word, docs_words, tf_list, idf, tfidf_list)


if __name__ == "__main__":
    interactive = True

    if "--no-input" in sys.argv:
        interactive = False

    main(interactive)