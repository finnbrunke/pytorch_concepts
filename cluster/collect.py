"""Summarize the report.json files of a sweep, averaged over seeds.

    python cluster/collect.py "$CMR_LOGS"            # table on stdout
    python cluster/collect.py "$CMR_LOGS" --csv out.csv

Runs are grouped by their name without the trailing ``_s<seed>``.
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = {
    "task_auroc": ("tasks", "macro", "score"),
    "concept_completeness": ("completeness", "macro", "completeness"),
    "rule_language_completeness": ("rule_language_completeness", "macro", "completeness"),
    "intervention_gain": ("interventions", "curves", "concepts_random", "gain"),
    "time_intervention_gain": ("interventions", "curves", "time_random", "gain"),
}
MODES = ("soft", "argmax", "sample")


def get(node, path):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, (int, float)) else None


def run_name(report_path: Path, root: Path) -> str:
    # <root>/<run>_<dataset>_cmr/version_N/report.json or <root>/<run>_<dataset>_cmr/report.json
    part = report_path.relative_to(root).parts[0]
    return re.sub(r"_(predator_prey|mit_bih)_cmr$", "", part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(args.root.rglob("report.json")):
        report = json.loads(path.read_text())
        group = re.sub(r"_s\d+$", "", run_name(path, args.root))
        for mode in MODES:
            for metric, keys in METRICS.items():
                value = get(report.get(mode, {}), keys)
                if value is not None and value == value:
                    groups[group][f"{mode}/{metric}"].append(value)

    rows = []
    for group, values in sorted(groups.items()):
        row = {"config": group}
        for key, numbers in values.items():
            row[key] = statistics.mean(numbers)
            row[f"{key}_std"] = statistics.pstdev(numbers)
            row["seeds"] = max(row.get("seeds", 0), len(numbers))
        rows.append(row)
    if not rows:
        raise SystemExit(f"no report.json under {args.root}")

    shown = [f"{mode}/{metric}" for mode in ("soft", "argmax") for metric in
             ("task_auroc", "rule_language_completeness", "intervention_gain")]
    print(f"{'config':<52}{'n':>3}" + "".join(f"{k.split('/')[0][:4] + ' ' + k.split('/')[1][:12]:>22}" for k in shown))
    for row in rows:
        cells = "".join(
            f"{row[k]:>14.3f}±{row[k + '_std']:<7.3f}" if k in row else f"{'-':>22}" for k in shown
        )
        print(f"{row['config']:<52}{row.get('seeds', 0):>3}{cells}")

    if args.csv:
        fields = ["config", "seeds"] + sorted({k for row in rows for k in row} - {"config", "seeds"})
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwritten: {args.csv}")


if __name__ == "__main__":
    main()
