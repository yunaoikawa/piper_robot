"""LLMSkillAuthor — the opt-in LLM-in-the-loop half of the hybrid controller.

The DeterministicPlanner refines skill parameters with fixed, hand-written rules and
is what the reproducible daily-CAD ablation uses. This module is the escalation path:
when the deterministic rules stop making progress, an LLM (Claude) reads the *real*
logged failures for an episode and proposes a parameter patch to the SkillLibrary,
which is then clamped to safe physical ranges and re-tried in MuJoCo.

It is strictly opt-in and never required:
  * ``available()`` is False unless the ``anthropic`` SDK is installed and a credential
    is resolvable, so importing this module has no effect on the ablation.
  * It only ever proposes numbers for the existing tunable parameters; the values are
    clamped to safe bounds before they touch the sim, so a bad suggestion cannot break
    the physics or fabricate a result — the success signal still comes from the real
    SceneVerifier end-state, exactly as in the deterministic path.

Run the worked example against a real ablation log:

    python -m wetrobo.agent.llm_agent --log runs/ablation_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

from wetrobo.skills.library import DEFAULT_PARAMS

# Safe physical ranges for each tunable parameter (metres, or settle steps). A proposed
# value outside its range is clamped, and unknown keys are ignored — the LLM can only
# move the knobs that already exist, within limits the sim tolerates.
PARAM_BOUNDS = {
    "approach_z": (0.15, 0.30),
    "lift_z": (0.22, 0.40),
    "transit_z": (0.30, 0.45),
    "place_z": (0.28, 0.40),
    "grasp_tol_m": (0.02, 0.06),
    "grasp_settle": (30, 120),
}

DEFAULT_MODEL = os.environ.get("WETROBO_LLM_MODEL", "claude-opus-4-8")

_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": "One or two sentences on why these parameter changes "
                           "should fix the observed failures.",
        },
        "patch": {
            "type": "array",
            "description": "Parameter changes to apply. Only include a knob you want "
                           "to move.",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "enum": list(PARAM_BOUNDS)},
                    "value": {"type": "number"},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rationale", "patch"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You tune the motor-skill parameters of a simulated 5-DOF lab robot performing a "
    "flask -> incubator pick-and-place in MuJoCo. You are given the current parameters, "
    "their physical meaning and safe ranges, and the real per-attempt failures logged "
    "from the last episode. Propose the smallest parameter change likely to fix the "
    "dominant failure mode. Move a knob only when the failures justify it."
)


def summarize_failures(records: list[dict], *, condition: str | None = None) -> dict:
    """Reduce raw logged attempts to a compact, per-outcome failure summary.

    Uses the real JSONL attempt records; nothing is synthesised. If ``condition`` is
    given (``"cad"``/``"vision"``) only that arm's attempts are counted."""
    attempts = [r for r in records if r.get("kind") == "attempt"]
    if condition is not None:
        attempts = [r for r in attempts if r.get("condition") == condition]
    counts: dict[str, int] = {}
    grasp_dists: list[float] = []
    stage_counts: dict[str, int] = {}
    for r in attempts:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        stage = r.get("stage_reached")
        if stage:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        d = r.get("grasp_dist_m")
        if r["outcome"] == "grasp_miss" and d is not None:
            grasp_dists.append(float(d))
    summary = {"n_attempts": len(attempts), "outcome_counts": counts}
    if stage_counts:
        summary["furthest_stage_counts"] = stage_counts
    if grasp_dists:
        summary["grasp_miss_dist_m"] = {
            "min": round(min(grasp_dists), 4),
            "max": round(max(grasp_dists), 4),
            "mean": round(sum(grasp_dists) / len(grasp_dists), 4),
        }
    return summary


class LLMSkillAuthor:
    """Optional LLM proposer of SkillLibrary parameter patches."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self._api_key = api_key
        self._client = None

    def available(self) -> bool:
        """True only if the ``anthropic`` SDK is importable and a credential is set.

        Opt-in gate: with no SDK or key this returns False and the caller falls back to
        the deterministic planner, so the ablation never depends on the LLM."""
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(
            self._api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def build_prompt(self, task: str, current_params: dict, failures: dict) -> str:
        knobs = {k: {"value": current_params.get(k, DEFAULT_PARAMS.get(k)),
                     "safe_range": list(PARAM_BOUNDS[k])} for k in PARAM_BOUNDS}
        return (
            f"Task: {task}\n\n"
            f"Current parameters (with safe ranges):\n{json.dumps(knobs, indent=2)}\n\n"
            f"Failures from the last episode (real MuJoCo rollouts):\n"
            f"{json.dumps(failures, indent=2)}\n\n"
            "Propose a parameter patch as JSON."
        )

    def propose_param_patch(self, task: str, current_params: dict,
                            failures: dict) -> dict:
        """Ask the LLM for a parameter patch. Returns the parsed structured response
        ``{"rationale": str, "patch": [{"key", "value"}, ...]}``.

        Raises RuntimeError if the author is not available — callers should guard with
        ``available()`` first."""
        if not self.available():
            raise RuntimeError(
                "LLMSkillAuthor unavailable: install `anthropic` and set "
                "ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)."
            )
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _PATCH_SCHEMA}},
            messages=[{"role": "user",
                       "content": self.build_prompt(task, current_params, failures)}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    @staticmethod
    def apply(skills, patch: dict) -> dict:
        """Clamp the proposed patch to safe ranges and write it into the SkillLibrary.

        Returns the applied changes ``{key: (old, new)}``. Values out of range are
        clamped; unknown keys are dropped — the LLM cannot inject arbitrary state."""
        applied: dict[str, tuple[float, float]] = {}
        for entry in patch.get("patch", []):
            key = entry.get("key")
            if key not in PARAM_BOUNDS:
                continue
            lo, hi = PARAM_BOUNDS[key]
            new = float(min(max(entry["value"], lo), hi))
            old = float(skills.params.get(key, DEFAULT_PARAMS.get(key)))
            if new != old:
                skills.params[key] = new
                applied[key] = (old, new)
        if applied:
            skills.save()
        return applied


def _main() -> int:
    from wetrobo.episode_log import EpisodeLog
    from wetrobo.skills.library import SkillLibrary

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="runs/ablation_test.jsonl",
                    help="real ablation JSONL to summarise failures from")
    ap.add_argument("--condition", default="vision", choices=["cad", "vision"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    records = EpisodeLog.read(args.log)
    failures = summarize_failures(records, condition=args.condition)
    if failures["n_attempts"] == 0:
        print(f"no '{args.condition}' attempts in {args.log}; run the ablation first")
        return 1

    task = "flask -> incubator pick-and-place (5-DOF right arm, transparent flask)"
    skills = SkillLibrary()
    author = LLMSkillAuthor(model=args.model)

    print(f"Real logged failures ({args.condition}):\n{json.dumps(failures, indent=2)}\n")
    if not author.available():
        print("LLMSkillAuthor is OPT-IN and currently unavailable "
              "(need `anthropic` installed + ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN).")
        print("The deterministic planner runs the ablation without it. The prompt that "
              "would be sent:\n")
        print(author.build_prompt(task, skills.params, failures))
        return 0

    patch = author.propose_param_patch(task, skills.params, failures)
    print(f"LLM rationale: {patch.get('rationale')}")
    applied = LLMSkillAuthor.apply(skills, patch)
    print(f"applied (clamped) changes: {applied or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
