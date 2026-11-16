"""Print a table of start and end test accuracy from results/*.json."""

import json
import sys
from pathlib import Path


def main(results_dir: str = "results") -> None:
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        run = json.loads(path.read_text())
        cfg, hist = run["config"], run["history"]
        rows.append((cfg["objective"], cfg["seed"], hist[0]["test_acc"], hist[-1]["test_acc"]))

    print("| Objective | Seed | Test acc after warm start | Test acc after RL | Gain |")
    print("| --- | --- | --- | --- | --- |")
    for obj, seed, start, end in rows:
        print(f"| {obj} | {seed} | {start:.3f} | {end:.3f} | {end - start:+.3f} |")
    for obj in sorted({r[0] for r in rows}):
        gains = [end - start for o, _, start, end in rows if o == obj]
        print(f"| {obj} | mean | | | {sum(gains) / len(gains):+.3f} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
