import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.workflow import (
    run_and_evaluate,
    load_validated_pairs,
    save_validated_pairs,
)
from src import load_data as ld


def main():
    parser = argparse.ArgumentParser(description="Financial reconciliation: match, review, improve")
    parser.add_argument("--review", action="store_true", help="Interactive accept/reject then re-run")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate current validated pairs only")
    parser.add_argument("--bank", default=config.BANK_STATEMENTS_PATH, help="Path to bank_statements.csv")
    parser.add_argument("--register", default=config.CHECK_REGISTER_PATH, help="Path to check_register.csv")
    parser.add_argument("--validated", default=config.VALIDATED_PAIRS_PATH, help="Path to validated_pairs.json")
    args = parser.parse_args()

    bank_path = os.path.join(PROJECT_ROOT, args.bank) if not os.path.isabs(args.bank) else args.bank
    register_path = os.path.join(PROJECT_ROOT, args.register) if not os.path.isabs(args.register) else args.register
    validated_path = os.path.join(PROJECT_ROOT, args.validated) if not os.path.isabs(args.validated) else args.validated

    if not os.path.isfile(bank_path) or not os.path.isfile(register_path):
        print("Error: bank_statements.csv and check_register.csv must exist.", file=sys.stderr)
        sys.exit(1)

    if args.eval_only:
        bank_df, register_df, ground_truth = ld.load_and_normalize(bank_path, register_path)
        validated = load_validated_pairs(validated_path)
        from src.evaluation import precision_recall_f1
        metrics = precision_recall_f1(validated, ground_truth, len(bank_df))
        print("Validated pairs:", len(validated))
        print("Precision:", round(metrics["precision"], 4))
        print("Recall:", round(metrics["recall"], 4))
        print("F1:", round(metrics["f1"], 4))
        return

    proposed, metrics, bank_df, register_df, ground_truth = run_and_evaluate(
        bank_path, register_path, validated_path, config=config
    )

    print("=== Reconciliation results ===")
    print("Proposed matches:", len(proposed))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("Correct:", metrics["correct_matches"], "/", metrics["total_should_match"])

    if args.review and proposed:
        validated = load_validated_pairs(validated_path)
        print("\n--- Review proposed matches (accept/reject) ---")
        for i, m in enumerate(proposed[:20], 1):
            print(f"\n[{i}] {m['bank_id']} <-> {m['register_id']} (conf={m['confidence']:.3f}, src={m['source']})")
            print("    Bank:", m.get("bank_tx", {}).get("description", ""), "| Reg:", m.get("register_tx", {}).get("description", ""))
            while True:
                choice = input("Accept (a) / Reject (r) / Quit (q): ").strip().lower()
                if choice in ("a", "r", "q"):
                    break
            if choice == "q":
                break
            if choice == "a":
                validated.append({"bank_id": m["bank_id"], "register_id": m["register_id"]})
        save_validated_pairs(validated_path, validated)
        print("\nValidated pairs saved. Re-run without --review to re-match with updated training.")
    elif not args.review:
        print("\nFirst 5 proposed matches:")
        for m in proposed[:5]:
            print(" ", m["bank_id"], "<->", m["register_id"], "conf=", round(m["confidence"], 3), m["source"])


if __name__ == "__main__":
    main()
