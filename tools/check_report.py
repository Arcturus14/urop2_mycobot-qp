"""Verify downloaded evidence and links in the generated identification report."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_directory")
    parser.add_argument("downloads", nargs="+")
    args = parser.parse_args()
    verified = 0
    for directory in map(Path, args.downloads):
        manifest = json.loads((directory / "transfer.json").read_text(encoding="utf-8"))
        for name, expected in manifest["sha256"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
            verified += 1
    output = Path(args.report_directory)
    markdown = (output / "REPORT.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(<([^>]+)>\)", markdown)
    for link in links:
        if not link.startswith("http"):
            assert Path(link).exists(), link
    metrics = pd.read_csv(output / "step_metrics.csv")
    assert len(metrics) == 96
    assert not metrics.duplicated(["joint", "repeat", "step"]).any()
    models = json.loads((output / "model_comparison.json").read_text(encoding="utf-8"))
    assert len(models) == 6
    assert all(item["train_events"] == 7 and item["validation_events"] == 8 for item in models.values())
    for path in (output / "figures").glob("*.png"):
        with Image.open(path) as picture:
            picture.verify()
    print(f"PASS: {verified} evidence hashes; {len(links)} report links; 96 unique records; 6 models; 42 train / 48 validation events")


if __name__ == "__main__":
    main()
