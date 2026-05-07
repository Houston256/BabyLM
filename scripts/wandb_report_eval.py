import argparse
import re
from pathlib import Path


def parse_average(report_path: Path) -> float | None:
    text = report_path.read_text()
    m = re.search(r"### AVERAGE ACCURACY\s*\n\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def collect_metrics(results_dir: Path, model_name: str, backend: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    base = results_dir / model_name
    if not base.is_dir():
        return metrics
    for report in base.rglob("best_temperature_report.txt"):
        parts = report.relative_to(base).parts
        if len(parts) < 5 or parts[1] != "zero_shot" or parts[2] != backend:
            continue
        task, dataset = parts[3], parts[4]
        avg = parse_average(report)
        if avg is not None:
            metrics[f"eval/{task}/{dataset}"] = avg
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--model-name", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--project", default="babylm")
    args = p.parse_args()

    metrics = collect_metrics(args.results_dir, args.model_name, args.backend)
    if not metrics:
        print("no eval reports found; nothing to log")
        return

    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping report")
        return

    run = wandb.init(project=args.project, id=args.run_id, resume="allow")
    run.log(metrics)
    run.summary.update(metrics)
    run.finish()
    print(f"logged {len(metrics)} eval metrics to wandb run {args.run_id}")


if __name__ == "__main__":
    main()
