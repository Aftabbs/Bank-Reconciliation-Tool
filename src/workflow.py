import json
import os
from typing import List, Dict, Any, Set, Tuple
import pandas as pd

try:
    from . import load_data
    from . import unique_amount_matching
    from . import ml_matching
    from . import evaluation
except ImportError:
    import load_data
    import unique_amount_matching
    import ml_matching
    import evaluation


def _serialize_match(m: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in m.items() if k in ("bank_id", "register_id", "confidence", "flags", "source")}
    if "bank_tx" in m and isinstance(m["bank_tx"], dict):
        out["bank_tx"] = _serialize_row(m["bank_tx"])
    if "register_tx" in m and isinstance(m["register_tx"], dict):
        out["register_tx"] = _serialize_row(m["register_tx"])
    return out


def _serialize_row(row: Dict) -> Dict:
    d = {}
    for k, v in row.items():
        if pd.isna(v):
            d[k] = None
        elif hasattr(v, "isoformat"):
            d[k] = v.isoformat()
        elif hasattr(v, "item"):
            d[k] = v.item()
        elif isinstance(v, (list, dict)):
            d[k] = v
        else:
            d[k] = v
    return d


def run_full_matching(
    bank_df: pd.DataFrame,
    register_df: pd.DataFrame,
    validated_pairs: List[Dict],
    amount_tolerance: float = 0.01,
    date_lag_threshold_days: int = 5,
    date_lag_confidence_cap: float = 0.9,
    svd_n_components: int = 30,
    ml_confidence_threshold: float = 0.3,
) -> Tuple[List[Dict], Set[str], Set[str]]:
    excluded_bank = {p.get("bank_id") for p in validated_pairs if p.get("bank_id")}
    excluded_reg = {p.get("register_id") for p in validated_pairs if p.get("register_id")}

    uam = unique_amount_matching.match_unique_amounts(
        bank_df,
        register_df,
        amount_tolerance=amount_tolerance,
        date_lag_threshold_days=date_lag_threshold_days,
        date_lag_confidence_cap=date_lag_confidence_cap,
    )
    uam_filtered = [
        m for m in uam
        if m["bank_id"] not in excluded_bank and m["register_id"] not in excluded_reg
    ]
    for m in uam_filtered:
        excluded_bank.add(m["bank_id"])
        excluded_reg.add(m["register_id"])

    ml_matches = ml_matching.match_remaining_with_ml(
        bank_df,
        register_df,
        validated_pairs=validated_pairs,
        excluded_bank_ids=excluded_bank,
        excluded_register_ids=excluded_reg,
        n_components=svd_n_components,
        confidence_threshold=ml_confidence_threshold,
    )
    for m in ml_matches:
        excluded_bank.add(m["bank_id"])
        excluded_reg.add(m["register_id"])

    all_proposed = uam_filtered + ml_matches
    return all_proposed, excluded_bank, excluded_reg


def load_validated_pairs(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_validated_pairs(path: str, pairs: List[Dict]) -> None:
    serialized = [_serialize_match(p) for p in pairs]
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)


def run_and_evaluate(
    bank_path: str,
    register_path: str,
    validated_pairs_path: str,
    config: Any = None,
) -> Tuple[List[Dict], Dict[str, float], pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    if config is None:
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import config as cfg
        config = cfg
    bank_df, register_df, ground_truth = load_data.load_and_normalize(
        bank_path,
        register_path,
        amount_tolerance=getattr(config, "AMOUNT_TOLERANCE", 0.01),
    )
    validated = load_validated_pairs(validated_pairs_path)
    proposed, _, _ = run_full_matching(
        bank_df,
        register_df,
        validated,
        amount_tolerance=getattr(config, "AMOUNT_TOLERANCE", 0.01),
        date_lag_threshold_days=getattr(config, "DATE_LAG_FLAG_THRESHOLD_DAYS", 5),
        date_lag_confidence_cap=getattr(config, "DATE_LAG_CONFIDENCE_PENALTY", 0.9),
        svd_n_components=getattr(config, "SVD_N_COMPONENTS", 30),
        ml_confidence_threshold=getattr(config, "ML_CONFIDENCE_THRESHOLD", 0.3),
    )
    total = len(bank_df)
    metrics = evaluation.precision_recall_f1(
        proposed,
        ground_truth,
        total_should_match=total,
    )
    return proposed, metrics, bank_df, register_df, ground_truth
