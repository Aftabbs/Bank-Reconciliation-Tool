from typing import Dict, List, Any


def precision_recall_f1(
    system_matches: List[Dict[str, Any]],
    ground_truth: Dict[str, str],
    total_should_match: int,
) -> Dict[str, float]:
    correct = 0
    for m in system_matches:
        bid = m.get("bank_id")
        rid = m.get("register_id")
        if bid is None or rid is None:
            continue
        if ground_truth.get(bid) == rid:
            correct += 1
    n_matches = len(system_matches)
    prec = correct / n_matches if n_matches else 0.0
    rec = correct / total_should_match if total_should_match else 0.0
    f1 = (
        2 * prec * rec / (prec + rec)
        if (prec + rec) > 0
        else 0.0
    )
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "correct_matches": correct,
        "total_system_matches": n_matches,
        "total_should_match": total_should_match,
    }
