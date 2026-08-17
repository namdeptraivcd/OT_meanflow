import argparse
import csv
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(row, key, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return default


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="./outputs")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["meanflow", "ot_gmf_hard", "ot_gmf_sinkhorn_hard"],
    )
    p.add_argument("--fid-threshold", type=float, default=None)
    p.add_argument("--entropy-threshold", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for method in args.methods:
        root = Path(args.out_dir) / method
        eval_rows = read_csv(root / "eval.csv")
        log_rows = read_csv(root / "logs.csv")
        final_eval = eval_rows[-1] if eval_rows else {}
        final_log = log_rows[-1] if log_rows else {}

        reached_fid = None
        if args.fid_threshold is not None:
            for row in eval_rows:
                if to_float(row, "feature_fid") <= args.fid_threshold:
                    reached_fid = row
                    break

        reached_entropy = None
        if args.entropy_threshold is not None:
            for row in eval_rows:
                if to_float(row, "class_entropy") >= args.entropy_threshold:
                    reached_entropy = row
                    break

        rows.append(
            {
                "method": method,
                "final_step": final_eval.get("step", final_log.get("step", "")),
                "final_elapsed_min": final_eval.get("elapsed_min", final_log.get("elapsed_min", "")),
                "final_feature_fid": final_eval.get("feature_fid", ""),
                "final_gen_confidence": final_eval.get("gen_confidence", ""),
                "final_class_entropy": final_eval.get("class_entropy", ""),
                "avg_sec_per_step": final_log.get("sec_per_step", ""),
                "avg_peak_mem_mb": final_log.get("peak_mem_mb", ""),
                "fid_reached_step": reached_fid.get("step", "") if reached_fid else "",
                "fid_reached_min": reached_fid.get("elapsed_min", "") if reached_fid else "",
                "entropy_reached_step": reached_entropy.get("step", "") if reached_entropy else "",
                "entropy_reached_min": reached_entropy.get("elapsed_min", "") if reached_entropy else "",
            }
        )

    headers = list(rows[0].keys()) if rows else []
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(str(row[h]).ljust(widths[h]) for h in headers))


if __name__ == "__main__":
    main()

