"""sfs_reward.py — RLVR (RL with Verifiable Rewards) reward for AQUA-NL.

WHY THIS EXISTS
---------------
AQUA-NL fine-tunes Qwen3-8B (LoRA) to generate speech-quality descriptions that
state *measured* numbers ("The SNR is 16.10 dB. The F0 mean is 121.00 Hz.").
We already have a fully deterministic verifier — the Signal Faithfulness Score
(`src/sfs.py`): it regex-parses the numeric claims out of generated text and
scores each claim against ground-truth SP features inside per-feature tolerances.
A deterministic, ground-truth-checked scalar is exactly the signal RL-with-
verifiable-rewards (RLVR / GRPO) wants. This module turns SFS into a per-sample
reward and wraps it for TRL's GRPOTrainer.

REWARD DESIGN
-------------
    reward = f1_weight       * SFS_F1(text vs gt_features)
           - rep_penalty     * rep_n(text, n)          # n-gram repetition fraction
           - nonascii_penalty* nonascii_frac(text)     # foreign-token injection

Two deliberate choices:

1. SFS-**F1**, NOT recall.  Recall = (mentioned GT features / |GT features|).
   It is trivially hackable by *number-spamming*: emit every plausible number
   for every feature and recall saturates at 1.0 regardless of correctness,
   because recall only asks "was the feature mentioned", not "was it right".
   RL will find and exploit that. F1 = harmonic mean of precision and recall,
   and precision = (correct claims / all claims) *punishes wrong and extra
   numbers*. A spammer's precision collapses, dragging F1 down. So F1 rewards
   "state the numbers you can get RIGHT", which is the actual task. (This is
   the documented research finding for this project: optimize F1, not recall.)

2. An explicit anti-degeneration penalty.  SFS only parses NUMBERS, so it is
   BLIND to fluency collapse: a checkpoint that emits "</sec></sec></sec>..."
   tag-spam, a repetition loop, or Chinese-character injection can still contain
   parseable numbers and therefore score non-zero SFS while being unreadable
   garbage (this is the documented v11/v12 section-path failure). Under RL that
   blindness is dangerous — the policy can drift toward high-SFS-but-degenerate
   text. We subtract the same two cheap, model-free degeneration signals
   `ckpt_selection.py` already uses for checkpoint selection: n-gram repetition
   fraction and non-ASCII character fraction. The bet is that RL on
   (SFS-F1 minus degeneration) can fix the degeneration that SFT alone could not,
   because the reward actively discourages it every step.

This module is intentionally dependency-light: it imports ONLY `sfs.py`,
`ckpt_selection.py`, and the stdlib. It does **not** import `trl`, so the reward
is fully unit-testable on CPU with no GPU / no RL library installed. The TRL glue
(`make_sfs_reward_func`) returns a plain closure with the GRPO reward signature;
`grpo_train.py` is where `trl` actually gets imported.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

# Both live in src/; conftest.py puts src/ on sys.path for tests, and
# grpo_train.py / inference do the same `sys.path.insert(0, src)`.
from sfs import HybridClaimParser, SFSScorer
from ckpt_selection import nonascii_frac, rep_n

# A single parser + scorer instance is reused across calls — both are stateless
# (the parser only compiles regexes at import time; the scorer holds constant
# tolerance tables), so sharing them avoids recompiling/reallocating per sample.
_PARSER = HybridClaimParser()
_SCORER = SFSScorer()


__all__ = [
    "sfs_f1",
    "sfs_reward",
    "make_sfs_reward_func",
    "extract_completion_text",
]


# ── Core scalar reward ────────────────────────────────────────────────────────
def sfs_f1(generated_text: str, gt_features: Mapping[str, Any]) -> float:
    """SFS-F1 of one generated description against its ground-truth feature dict.

    Reuses the project's `HybridClaimParser` (tagged spans first, regex fallback)
    and `SFSScorer` — the *same* code path the deterministic evaluation uses, so
    the RL reward is exactly the metric the paper reports. No reimplementation.

    `gt_features` is the per-clip ground-truth dict in the shape `SFSScorer.score`
    expects: scalar features keyed by name (e.g. {"snr": 16.1, "f0_mean": 121.0})
    plus optional "overlap_segments": [(start_s, end_s), ...] for the IoU path.

    Returns the F1 in [0, 1]. Empty / unparseable text → 0.0 (precision 0).
    """
    if not generated_text:
        return 0.0
    claims = _PARSER.parse(generated_text)
    result = _SCORER.score(claims, dict(gt_features))
    return float(result["f1"])


def sfs_reward(
    generated_text: str,
    gt_features: Mapping[str, Any],
    *,
    f1_weight: float = 1.0,
    rep_penalty: float = 0.5,
    nonascii_penalty: float = 1.0,
    rep_n: int = 4,
) -> float:
    """Verifiable RL reward for ONE generated description.

        reward = f1_weight * SFS_F1(text vs gt)
               - rep_penalty * rep_n_fraction(text, n=rep_n)
               - nonascii_penalty * nonascii_fraction(text)

    See the module docstring for why F1 (not recall) and why the degeneration
    penalty. The reward is a single float; it is NOT clamped, so a heavily
    degenerate completion with no correct claims can go negative — which is the
    point: GRPO advantages are relative within a group, and a clearly-bad
    completion should sit below a merely-mediocre one.

    Args:
        generated_text:    the model's completion (decoded string).
        gt_features:        per-clip ground-truth feature dict (SFSScorer shape).
        f1_weight:          weight on the faithfulness term (default 1.0).
        rep_penalty:        weight on the n-gram repetition penalty (default 0.5).
        nonascii_penalty:   weight on the non-ASCII fraction penalty (default 1.0).
        rep_n:              n for the repetition n-gram (default 4, matches
                            ckpt_selection's default and the templated-prose
                            separation point).

    Returns:
        float reward.
    """
    text = generated_text or ""
    f1 = sfs_f1(text, gt_features)
    # rep_n() and nonascii_frac() are imported from ckpt_selection — the exact
    # same degeneration signals used for checkpoint selection, kept in one place.
    rep_term = _rep_fraction(text, rep_n)
    nonascii_term = nonascii_frac(text)
    return (
        f1_weight * f1
        - rep_penalty * rep_term
        - nonascii_penalty * nonascii_term
    )


# `rep_n` is both a kwarg name (the n-gram size) and the imported function name.
# Alias the function so the kwarg can shadow it inside sfs_reward without losing
# access to the callable.
def _rep_fraction(text: str, n: int) -> float:
    return rep_n(text, n)


# ── TRL GRPO batch wrapper ────────────────────────────────────────────────────
def extract_completion_text(completion: Any) -> str:
    """Normalize a TRL completion into a plain string.

    TRL passes completions in one of two shapes depending on whether the dataset
    is "prompt-completion" (plain text) or "conversational" (chat messages):

      - str:                      "The SNR is 16.10 dB. ..."         → returned as-is
      - list[dict] (chat turns):  [{"role": "assistant", "content": "..."}]
                                  → the content of the LAST turn is returned
      - dict (single message):    {"role": ..., "content": "..."}    → its content

    Anything else is coerced with str(). Returns "" for None / empty.
    """
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return str(completion.get("content", "") or "")
    # A list of chat-message dicts — take the last assistant-ish turn's content.
    if isinstance(completion, Sequence):
        if not completion:
            return ""
        last = completion[-1]
        if isinstance(last, Mapping):
            return str(last.get("content", "") or "")
        return str(last)
    return str(completion)


def _resolve_gt_list(
    prompts: Sequence[Any] | None,
    completions: Sequence[Any],
    gt_lookup: Any,
    kwargs: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Figure out the per-completion ground-truth feature dicts.

    The GT lookup mechanism is intentionally explicit and supports the two
    natural ways TRL surfaces per-sample side information:

    1. PROMPT-KEYED dict (`gt_lookup` is a Mapping): map each prompt (or each
       value of a `keys`/`clip_ids`/`id` kwargs column) to its GT dict. This is
       the right shape when the dataset rows carry a stable clip id and you build
       `{clip_id_or_prompt: gt_features}` once.

    2. ALIGNED LIST passed through kwargs: TRL forwards every non-standard
       dataset column to the reward function as a kwarg list aligned with
       `completions`. So a dataset column named `gt_features` (or `gt`) arrives
       as `kwargs["gt_features"]`, one GT dict per completion. If `gt_lookup` is
       itself a list/sequence (not a Mapping), it is treated as that aligned list.

    Resolution order: an aligned list in kwargs ("gt_features" / "gt") wins;
    else a sequence `gt_lookup`; else a Mapping `gt_lookup` keyed by a kwargs id
    column ("clip_ids"/"keys"/"ids"/"id"/"clip_id") or, failing that, by prompt.

    Returns a list of GT dicts the same length as `completions`. Missing entries
    become `{}` (which yields F1 0.0 — an honest "no GT, no credit").
    """
    n = len(completions)

    # (2) aligned list passed via kwargs.
    for key in ("gt_features", "gt", "ground_truth"):
        if key in kwargs and isinstance(kwargs[key], Sequence) and not isinstance(kwargs[key], (str, bytes)):
            aligned = list(kwargs[key])
            return _pad_to(aligned, n)

    # gt_lookup itself given as an aligned sequence.
    if isinstance(gt_lookup, Sequence) and not isinstance(gt_lookup, (str, bytes)):
        return _pad_to(list(gt_lookup), n)

    # (1) Mapping lookup, keyed by an explicit id column or by the prompt text.
    if isinstance(gt_lookup, Mapping):
        keys = _per_sample_keys(prompts, kwargs, n)
        return [dict(gt_lookup.get(k, {})) if k is not None else {} for k in keys]

    # Callable lookup: gt_lookup(prompt) -> gt dict.
    if callable(gt_lookup):
        keys = _per_sample_keys(prompts, kwargs, n)
        out = []
        for k in keys:
            try:
                out.append(dict(gt_lookup(k) or {}))
            except Exception:
                out.append({})
        return out

    # Nothing usable — empty GT for every sample (rewards collapse to the
    # negative degeneration terms only, which is at least not silently wrong).
    return [{} for _ in range(n)]


def _per_sample_keys(
    prompts: Sequence[Any] | None,
    kwargs: Mapping[str, Any],
    n: int,
) -> list[Any]:
    """Per-sample key for a Mapping/callable GT lookup: prefer an explicit id
    column forwarded through kwargs, else the prompt text."""
    for key in ("clip_ids", "keys", "ids", "id", "clip_id", "clip_stem", "filename"):
        if key in kwargs and isinstance(kwargs[key], Sequence) and not isinstance(kwargs[key], (str, bytes)):
            return _pad_to(list(kwargs[key]), n, fill=None)
    if prompts is not None:
        return _pad_to(list(prompts), n, fill=None)
    return [None] * n


def _pad_to(seq: list, n: int, fill: Any = None) -> list:
    """Truncate or pad `seq` to length n."""
    if len(seq) == n:
        return seq
    if len(seq) > n:
        return seq[:n]
    return seq + [fill] * (n - len(seq))


def make_sfs_reward_func(
    gt_lookup: Any = None,
    *,
    f1_weight: float = 1.0,
    rep_penalty: float = 0.5,
    nonascii_penalty: float = 1.0,
    rep_n: int = 4,
) -> Callable[..., list[float]]:
    """Build a TRL-GRPO-compatible batch reward function.

    The returned closure has TRL's reward signature

        reward_func(prompts, completions, **kwargs) -> list[float]

    and maps every completion to `sfs_reward(...)` against its ground-truth
    feature dict. Completions may be plain strings or chat-message lists — both
    are handled by `extract_completion_text`.

    Ground-truth resolution (see `_resolve_gt_list` for the full contract):
      - Pass GT as a dataset column so TRL forwards it via kwargs
        (`gt_features=[...]` aligned with completions) — simplest and recommended.
      - OR pass a list here as `gt_lookup` aligned with the batch.
      - OR pass a Mapping `gt_lookup` keyed by clip id (forwarded via a
        `clip_ids=[...]` kwargs column) or by prompt text.

    The reward weights are bound at construction time so the GRPO loop just calls
    `reward_func(prompts, completions, **kwargs)`.

    NOTE: TRL also supports passing the function's `__name__` into logs; the
    closure is named for that.
    """

    def reward_func(prompts=None, completions=None, **kwargs) -> list[float]:
        if completions is None:
            completions = []
        gt_list = _resolve_gt_list(prompts, completions, gt_lookup, kwargs)
        rewards: list[float] = []
        for completion, gt in zip(completions, gt_list):
            text = extract_completion_text(completion)
            rewards.append(
                sfs_reward(
                    text,
                    gt or {},
                    f1_weight=f1_weight,
                    rep_penalty=rep_penalty,
                    nonascii_penalty=nonascii_penalty,
                    rep_n=rep_n,
                )
            )
        return rewards

    reward_func.__name__ = "sfs_reward_func"
    return reward_func
