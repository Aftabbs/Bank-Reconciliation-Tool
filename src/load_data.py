import re
from typing import List, Dict, Tuple

import pandas as pd

TYPE_REGISTER_TO_CANONICAL = {"DR": "DEBIT", "CR": "CREDIT"}
TYPE_BANK_TO_CANONICAL = {"DEBIT": "DEBIT", "CREDIT": "CREDIT"}


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")


def _normalize_type(series: pd.Series, is_bank: bool) -> pd.Series:
    if is_bank:
        return series.str.upper().map(lambda x: TYPE_BANK_TO_CANONICAL.get(x, x))
    return series.str.upper().map(lambda x: TYPE_REGISTER_TO_CANONICAL.get(x, x))


def _tokenize_description(desc: str) -> List[str]:
    if pd.isna(desc) or not isinstance(desc, str):
        return []
    tokens = re.findall(r"[a-z0-9]+", desc.lower())
    return [t for t in tokens if len(t) > 1]


def load_bank_statements(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = _parse_date(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["type_normalized"] = _normalize_type(df["type"], is_bank=True)
    df["numeric_id"] = df["transaction_id"].str.extract(r"(\d+)", expand=False).astype(str)
    return df.dropna(subset=["date", "amount"]).reset_index(drop=True)


def load_check_register(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = _parse_date(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["type_normalized"] = _normalize_type(df["type"], is_bank=False)
    df["numeric_id"] = df["transaction_id"].str.extract(r"(\d+)", expand=False).astype(str)
    return df.dropna(subset=["date", "amount"]).reset_index(drop=True)


def amounts_match(a: float, b: float, tolerance: float = 0.01) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def get_ground_truth(
    bank_df: pd.DataFrame, register_df: pd.DataFrame
) -> Dict[str, str]:
    bank_ids = set(bank_df["numeric_id"].astype(str))
    reg_ids = set(register_df["numeric_id"].astype(str))
    return {f"B{nid}": f"R{nid}" for nid in bank_ids if nid in reg_ids}


def add_descriptor_terms(
    df: pd.DataFrame, description_col: str = "description", include_type: bool = True
) -> pd.DataFrame:
    terms = df[description_col].apply(_tokenize_description)
    if include_type and "type_normalized" in df.columns:
        type_term = df["type_normalized"].fillna("").astype(str).str.lower()

        def add_type(tokens, ttype):
            if ttype:
                return list(tokens) + [ttype]
            return list(tokens)

        terms = [add_type(terms.iloc[i], type_term.iloc[i]) for i in range(len(df))]
    df = df.copy()
    df["terms"] = terms
    return df


def load_and_normalize(
    bank_path: str,
    register_path: str,
    amount_tolerance: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    bank_df = load_bank_statements(bank_path)
    register_df = load_check_register(register_path)
    bank_df = add_descriptor_terms(bank_df)
    register_df = add_descriptor_terms(register_df)
    ground_truth = get_ground_truth(bank_df, register_df)
    return bank_df, register_df, ground_truth
