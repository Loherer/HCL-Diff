from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hcl_diff.evaluation.statistics import cluster_bootstrap_difference, holm_adjust, paired_wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patient-clustered method comparisons.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--method-column", required=True)
    parser.add_argument("--patient-column", required=True)
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--comparisons", required=True, nargs="+", help="Pairs formatted as first:second")
    parser.add_argument("--bootstrap-iterations", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    frame = pd.read_csv(arguments.metrics)
    rows: list[dict[str, object]] = []
    for expression in arguments.comparisons:
        first, second = expression.split(":", maxsplit=1)
        bootstrap = cluster_bootstrap_difference(
            frame=frame,
            value_column=arguments.value_column,
            method_column=arguments.method_column,
            first_method=first,
            second_method=second,
            patient_column=arguments.patient_column,
            iterations=int(arguments.bootstrap_iterations),
            seed=int(arguments.seed),
        )
        paired = frame.pivot_table(
            index=arguments.patient_column,
            columns=arguments.method_column,
            values=arguments.value_column,
            aggfunc="mean",
        ).dropna(subset=[first, second])
        rows.append(
            {
                "comparison": f"{first}-{second}",
                **bootstrap,
                "wilcoxon_p": paired_wilcoxon(paired[first], paired[second]),
            }
        )
    adjusted = holm_adjust([float(row["wilcoxon_p"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p"] = value
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)


if __name__ == "__main__":
    main()
