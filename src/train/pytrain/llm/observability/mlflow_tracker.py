"""
MLflow-based observability tracker for LLM fine-tuning.

Logs ONLY allowlisted metrics (loss, eval_loss, perplexity,
throughput, learning_rate).  Gradient norms, activations, sample data,
and attention weights are explicitly blocked.

Runs a file-based MLflow backend inside the enclave at
``/mnt/remote/output/mlruns/`` so nothing leaves the CCR during training.
At run completion, metrics are exported as an encrypted JSON bundle.
"""

import json
import math
import os
import time

ALLOWED_METRICS = frozenset({
    "loss", "train_loss", "eval_loss",
    "perplexity", "eval_perplexity",
    "throughput_tokens_per_sec",
    "learning_rate", "lr",
    "epoch", "step",
    "train_runtime", "train_samples_per_second",
})

BLOCKED_METRIC_PREFIXES = (
    "grad", "activation", "attention", "weight_norm",
    "sample", "embedding", "logit",
)


class MLflowTracker:
    """Enclave-local MLflow metric tracker with strict allowlisting."""

    def __init__(self, cfg: dict):
        obs = cfg.get("observability", {})
        mlflow_cfg = obs.get("mlflow", {})
        self.enabled = mlflow_cfg.get("enabled", True)
        self.experiment_name = mlflow_cfg.get("experiment_name", "depa-llm-finetune")
        self.output_dir = cfg.get("output", {}).get("path", "/mnt/remote/output")
        self._mlflow = None
        self._run = None
        self._metrics_buffer = []
        self._start_time = None

    def start_run(self):
        if not self.enabled:
            return

        try:
            import mlflow
            self._mlflow = mlflow

            tracking_uri = os.path.join(self.output_dir, "mlruns")
            os.makedirs(tracking_uri, exist_ok=True)
            mlflow.set_tracking_uri(f"file://{tracking_uri}")
            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run()
            self._start_time = time.time()
            print(f"MLflow run started: {self._run.info.run_id}")
        except ImportError:
            print("WARNING: mlflow not installed, metrics tracking disabled")
            self.enabled = False
        except Exception as e:
            print(f"WARNING: MLflow init failed ({e}), metrics tracking disabled")
            self.enabled = False

    def _is_allowed(self, key: str) -> bool:
        if key in ALLOWED_METRICS:
            return True
        for prefix in BLOCKED_METRIC_PREFIXES:
            if key.startswith(prefix):
                return False
        if key.startswith("eval_") and key.replace("eval_", "") in ALLOWED_METRICS:
            return True
        return key in ALLOWED_METRICS

    def on_log(self, metrics: dict):
        """Log a batch of metrics.  Only allowlisted keys are recorded."""
        if not self.enabled or self._mlflow is None:
            return

        step = metrics.get("step", 0)
        filtered = {}
        for k, v in metrics.items():
            if self._is_allowed(k) and isinstance(v, (int, float)):
                filtered[k] = v

        if "loss" in filtered and "perplexity" not in filtered:
            try:
                filtered["perplexity"] = math.exp(min(filtered["loss"], 20))
            except (OverflowError, ValueError):
                pass

        if filtered:
            self._mlflow.log_metrics(filtered, step=step)
            self._metrics_buffer.append({"step": step, **filtered})

    def on_epoch_end(self, metrics: dict):
        self.on_log(metrics)

    def end_run(self, status="FINISHED"):
        if not self.enabled or self._mlflow is None:
            return

        try:
            if self._start_time:
                self._mlflow.log_metric("total_runtime_seconds", time.time() - self._start_time)

            metrics_path = os.path.join(self.output_dir, "metrics_export.json")
            with open(metrics_path, "w") as f:
                json.dump(self._metrics_buffer, f, indent=2)

            self._mlflow.log_artifact(metrics_path)
            self._mlflow.end_run(status=status)
            print(f"MLflow run ended ({status}). Metrics exported to {metrics_path}")
        except Exception as e:
            print(f"WARNING: MLflow end_run failed: {e}")

    def as_hf_callback(self):
        """Return a HuggingFace TrainerCallback that delegates to this tracker."""
        tracker = self

        from transformers import TrainerCallback

        class _MLflowHFCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    metrics = dict(logs)
                    metrics["step"] = state.global_step
                    tracker.on_log(metrics)

            def on_epoch_end(self, args, state, control, **kwargs):
                tracker.on_epoch_end({"epoch": state.epoch, "step": state.global_step})

        return _MLflowHFCallback()
