import re
from pathlib import Path

_AVG = re.compile(r"### AVERAGE ACCURACY\s*\n\s*([0-9.]+)")
_EYE = re.compile(r"EYE TRACKING SCORE:\s*([0-9.]+)")
_SPR = re.compile(r"SELF-PACED READING SCORE:\s*([0-9.]+)")


def parse_eval_results(model_results_dir: Path) -> dict[str, float]:
    """Walk a model's results dir and extract scalar metrics from *report.txt files.

    The eval pipeline writes reports under <model_results_dir>/<rev>/zero_shot/<backend>/<task>/.../*report.txt
    Returns a flat dict keyed by 'eval/<rest-of-path>'.
    """
    metrics: dict[str, float] = {}
    if not model_results_dir.is_dir():
        return metrics

    for report in model_results_dir.rglob("*report.txt"):
        text = report.read_text()
        parts = report.parts
        if "zero_shot" not in parts:
            continue
        idx = parts.index("zero_shot")
        name = "eval/" + "/".join(parts[idx:-1])

        if m := _AVG.search(text):
            metrics[name] = float(m.group(1))
        if m := _EYE.search(text):
            metrics[f"{name}/eye_tracking"] = float(m.group(1))
        if m := _SPR.search(text):
            metrics[f"{name}/self_paced_reading"] = float(m.group(1))

    return metrics


def model_results_dir(results_root: Path, checkpoint_name: str) -> Path:
    """Mirror the eval pipeline's `Path(model_path).stem` directory naming."""
    return results_root / Path(checkpoint_name).stem
