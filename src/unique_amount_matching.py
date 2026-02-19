from typing import List, Tuple, Dict, Any
import pandas as pd

from .load_data import amounts_match


def _round_amount(a: float) -> float:
    return round(float(a), 2)


def get_unique_amounts(
    bank_df: pd.DataFrame,
    register_df: pd.DataFrame,
    tolerance: float = 0.01,
) -> Tuple[set, set]:
    bank_amounts = bank_df["amount"].apply(_round_amount)
    reg_amounts = register_df["amount"].apply(_round_amount)
    bank_per_amount = bank_amounts.value_counts()
    reg_per_amount = reg_amounts.value_counts()
    unique_in_bank = set(bank_per_amount[bank_per_amount == 1].index)
    unique_in_reg = set(reg_per_amount[reg_per_amount == 1].index)
    return unique_in_bank, unique_in_reg


def match_unique_amounts(
    bank_df: pd.DataFrame,
    register_df: pd.DataFrame,
    amount_tolerance: float = 0.01,
    date_lag_threshold_days: int = 5,
    date_lag_confidence_cap: float = 0.9,
) -> List[Dict[str, Any]]:
    bank_df = bank_df.copy()
    register_df = register_df.copy()
    bank_df["_row"] = range(len(bank_df))
    register_df["_row"] = range(len(register_df))

    unique_bank, unique_reg = get_unique_amounts(
        bank_df, register_df, tolerance=amount_tolerance
    )
    bank_rounded = bank_df["amount"].apply(_round_amount)
    reg_rounded = register_df["amount"].apply(_round_amount)
    candidate_amounts = unique_bank & unique_reg

    results = []
    for amt in candidate_amounts:
        bank_row = bank_df[bank_rounded == amt].iloc[0]
        reg_row = register_df[reg_rounded == amt].iloc[0]
        if bank_row["type_normalized"] != reg_row["type_normalized"]:
            continue
        bdate = pd.Timestamp(bank_row["date"])
        rdate = pd.Timestamp(reg_row["date"])
        lag_days = (rdate - bdate).days
        lag_abs = abs(lag_days)
        if lag_abs <= date_lag_threshold_days:
            confidence = 1.0
        else:
            confidence = max(0.0, date_lag_confidence_cap)
        flags = []
        if lag_abs > date_lag_threshold_days:
            flags.append("date_lag_exceeds_threshold")
        if bank_row["type_normalized"] != reg_row["type_normalized"]:
            flags.append("type_mismatch")

        results.append({
            "bank_id": bank_row["transaction_id"],
            "register_id": reg_row["transaction_id"],
            "bank_tx": bank_row.drop(labels=["_row"], errors="ignore").to_dict(),
            "register_tx": reg_row.drop(labels=["_row"], errors="ignore").to_dict(),
            "confidence": confidence,
            "flags": flags,
            "source": "unique_amount",
            "date_lag_days": lag_days,
        })
    return results
