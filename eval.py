"""Evaluation script for the MRT route-tracing model.

Computes metrics by comparing a predictions CSV against a labeled CSV.

Labeled CSV must contain at least: id, transfer_count  (e.g. dataset/public/train.csv).
Predictions CSV must contain:      id, transfer_count[, route]

Usage:
    python eval.py --pred submission.csv --truth dataset/public/train.csv

Metrics reported:
    - Transfer-count accuracy : % of rows where predicted transfer_count == ground truth
    - Transfer-count MAE      : mean absolute error of transfer_count
    - Valid-output rate       : % of predictions that are valid (transfer_count >= 0)
    - Coverage                : % of truth ids that have a prediction
"""

import argparse
import pandas as pd


def evaluate(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    """Return a dict of evaluation metrics for predictions vs ground truth."""
    truth = truth_df[["id", "transfer_count"]].rename(columns={"transfer_count": "tc_true"})
    pred = pred_df[["id", "transfer_count"]].rename(columns={"transfer_count": "tc_pred"})

    merged = truth.merge(pred, on="id", how="left")
    n = len(merged)

    coverage = merged["tc_pred"].notna().mean()

    # Only score rows that have a prediction.
    scored = merged.dropna(subset=["tc_pred"]).copy()
    scored["tc_pred"] = scored["tc_pred"].astype(int)

    # A prediction is "valid" if the model produced a non-fallback answer (>= 0).
    valid_rate = (scored["tc_pred"] >= 0).mean() if len(scored) else 0.0

    correct = (scored["tc_pred"] == scored["tc_true"]).mean() if len(scored) else 0.0
    mae = (scored["tc_pred"] - scored["tc_true"]).abs().mean() if len(scored) else float("nan")

    return {
        "rows_total": n,
        "rows_scored": len(scored),
        "coverage": coverage,
        "valid_output_rate": valid_rate,
        "transfer_count_accuracy": correct,
        "transfer_count_mae": mae,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate MRT route-tracing predictions.")
    ap.add_argument("--pred", required=True, help="Predictions CSV (id, transfer_count[, route])")
    ap.add_argument("--truth", default="dataset/public/train.csv",
                    help="Labeled CSV with ground-truth transfer_count")
    args = ap.parse_args()

    pred_df = pd.read_csv(args.pred)
    truth_df = pd.read_csv(args.truth)

    m = evaluate(pred_df, truth_df)

    print("=" * 44)
    print("  MRT Route-Tracing — Evaluation")
    print("=" * 44)
    print(f"  Rows (truth)            : {m['rows_total']}")
    print(f"  Rows scored             : {m['rows_scored']}")
    print(f"  Coverage                : {m['coverage']:.2%}")
    print(f"  Valid-output rate       : {m['valid_output_rate']:.2%}")
    print(f"  Transfer-count accuracy : {m['transfer_count_accuracy']:.2%}")
    print(f"  Transfer-count MAE      : {m['transfer_count_mae']:.3f}")
    print("=" * 44)


if __name__ == "__main__":
    main()
