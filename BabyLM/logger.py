from typing import Protocol


class Logger(Protocol):
    def update_config(self, cfg: dict) -> None: ...
    def log(self, metrics: dict, step: int) -> None: ...
    def update_summary(self, metrics: dict) -> None: ...
    def finish(self) -> None: ...


class NoopLogger:
    def update_config(self, cfg: dict) -> None:
        pass

    def log(self, metrics: dict, step: int) -> None:
        pass

    def update_summary(self, metrics: dict) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbLogger:
    def __init__(self, project: str, name: str | None = None):
        import wandb

        self._wandb = wandb
        self.run = wandb.init(project=project, name=name)
        # Anchor all metrics to tokens_seen so runs with different batch sizes
        # / grad-accum settings line up on the same x-axis in the dashboard.
        wandb.define_metric("tokens_seen")
        wandb.define_metric("*", step_metric="tokens_seen")

    def update_config(self, cfg: dict) -> None:
        self.run.config.update(cfg, allow_val_change=True)

    def log(self, metrics: dict, step: int) -> None:
        self.run.log(metrics, step=step)

    def update_summary(self, metrics: dict) -> None:
        self.run.summary.update(metrics)

    def finish(self) -> None:
        self.run.finish()


def build_logger(use_wandb: bool, project: str, name: str | None = None) -> Logger:
    return WandbLogger(project, name) if use_wandb else NoopLogger()
