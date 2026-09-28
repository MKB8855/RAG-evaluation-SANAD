"""SANAD preprocessing utilities."""

import pandas as pd


def load_clean_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).reset_index(drop=True)


def build_document_text(row) -> str:
    return (
        "اسم النظام: " + str(row["law_name"]) +
        "\nرقم المادة: " + str(row["article_number"]) +
        "\nالنص القانوني: " + str(row["text"])
    )


def prepare_documents(df: pd.DataFrame) -> list[str]:
    return [build_document_text(row) for _, row in df.iterrows()]


def normalize_label(value) -> str:
    return str(value).strip()
