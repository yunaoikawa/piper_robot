"""Safety-first selection of a grasp-window observation method."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rollout.grasp_window import GraspWindowMethod


METHODS: tuple[GraspWindowMethod, ...] = (
    "WHITE_WINDOW",
    "MASK_GEOMETRY",
    "HYBRID",
)


@dataclass(frozen=True)
class AblationSampleResult:
    sample_id: str
    expected_window_ready: bool
    predictions: dict[str, bool]
    inference_ms: dict[str, float]


@dataclass(frozen=True)
class MethodMetrics:
    method: str
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    recall: float
    mean_inference_ms: float


@dataclass(frozen=True)
class AblationSelection:
    selected_method: str
    metrics: tuple[MethodMetrics, ...]
    selection_reason: str

    def to_dict(self) -> dict:
        return {
            "selected_method": self.selected_method,
            "selection_reason": self.selection_reason,
            "metrics": [asdict(value) for value in self.metrics],
        }


def evaluate_methods(
    samples: list[AblationSampleResult],
) -> tuple[MethodMetrics, ...]:
    if not samples:
        raise ValueError("ablation requires at least one sample")
    metrics = []
    for method in METHODS:
        fp = fn = tp = tn = 0
        times = []
        for sample in samples:
            if method not in sample.predictions:
                raise ValueError(f"sample {sample.sample_id} lacks {method}")
            predicted = bool(sample.predictions[method])
            expected = bool(sample.expected_window_ready)
            tp += int(predicted and expected)
            fp += int(predicted and not expected)
            tn += int(not predicted and not expected)
            fn += int(not predicted and expected)
            times.append(float(sample.inference_ms.get(method, 0.0)))
        metrics.append(
            MethodMetrics(
                method=method,
                false_positives=fp,
                false_negatives=fn,
                true_positives=tp,
                true_negatives=tn,
                recall=tp / max(tp + fn, 1),
                mean_inference_ms=sum(times) / max(len(times), 1),
            )
        )
    return tuple(metrics)


def select_method(samples: list[AblationSampleResult]) -> AblationSelection:
    metrics = evaluate_methods(samples)
    safe = [metric for metric in metrics if metric.false_positives == 0]
    if not safe:
        # HYBRID is the fail-closed conjunction and is the only permissible
        # fallback when the labelled set exposes false positives everywhere.
        selected = next(value for value in metrics if value.method == "HYBRID")
        reason = "no zero-false-positive method; fail-closed HYBRID selected"
    else:
        selected = min(
            safe,
            key=lambda value: (
                -value.recall,
                0 if value.method == "HYBRID" else 1,
                value.mean_inference_ms,
                METHODS.index(value.method),
            ),
        )
        reason = (
            "zero false positives, then maximum recall, then fail-closed "
            "conjunction, then minimum latency"
        )
    return AblationSelection(selected.method, metrics, reason)
