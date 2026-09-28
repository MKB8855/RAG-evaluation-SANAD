"""SANAD FAISS retrieval and evaluation utilities."""

import time
import faiss
import pandas as pd
from tqdm.auto import tqdm

from embedding import encode_query
from preprocessing import normalize_label


def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_faiss_index(index, path: str) -> None:
    faiss.write_index(index, path)


def load_faiss_index(path: str):
    return faiss.read_index(path)


def find_gold_rank(indices, df_clean: pd.DataFrame, gold_law: str, gold_article: str):
    gold_law = normalize_label(gold_law)
    gold_article = normalize_label(gold_article)

    for rank, idx in enumerate(indices[0], start=1):
        row = df_clean.iloc[idx]
        if (
            normalize_label(row["law_name"]) == gold_law
            and normalize_label(row["article_number"]) == gold_article
        ):
            return rank
    return None


def warm_up(model, model_name: str) -> None:
    _ = encode_query(model, "ما حقوق العامل؟", model_name)


def evaluate_retrieval(model, model_name: str, index, df_clean: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    warm_up(model, model_name)
    results = []

    for _, test_row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {model_name}"):
        query = str(test_row["query"])
        gold_law = normalize_label(test_row["gold_law"])
        gold_article = normalize_label(test_row["gold_article"])

        start = time.perf_counter()
        query_embedding = encode_query(model, query, model_name)
        scores, indices = index.search(query_embedding, index.ntotal)
        elapsed = time.perf_counter() - start

        gold_rank = find_gold_rank(indices, df_clean, gold_law, gold_article)

        result = {
            "query_id": test_row["query_id"],
            "query_type": test_row["query_type"],
            "query": query,
            "gold_law": gold_law,
            "gold_article": gold_article,
            "gold_rank": gold_rank,
            "Recall@3": int(gold_rank is not None and gold_rank <= 3),
            "Recall@5": int(gold_rank is not None and gold_rank <= 5),
            "Reciprocal_Rank": (1 / gold_rank) if gold_rank else 0,
            "Time_seconds": elapsed,
        }

        for i in range(5):
            idx = indices[0][i]
            row = df_clean.iloc[idx]
            result[f"Top{i+1}_Law"] = str(row["law_name"])
            result[f"Top{i+1}_Article"] = str(row["article_number"])
            result[f"Top{i+1}_Score"] = float(scores[0][i])

        results.append(result)

    return pd.DataFrame(results)


def summarize_results(results_df: pd.DataFrame):
    overall = {
        "Recall@3": results_df["Recall@3"].mean(),
        "Recall@5": results_df["Recall@5"].mean(),
        "MRR": results_df["Reciprocal_Rank"].mean(),
        "Average Retrieval Time": results_df["Time_seconds"].mean(),
    }

    by_type = results_df.groupby("query_type").agg(
        Recall_at_3=("Recall@3", "mean"),
        Recall_at_5=("Recall@5", "mean"),
        MRR=("Reciprocal_Rank", "mean"),
        Avg_Time=("Time_seconds", "mean")
    )

    return overall, by_type
